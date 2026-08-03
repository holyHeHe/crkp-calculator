#!/usr/bin/env python3
"""Natural-prevalence calibration analysis: raw, nested Platt, and nested isotonic probabilities."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.cli import load_cfg, load_development, parser
from crkp_ml.config import output_dir
from crkp_ml.metrics import (
    add_auc_cluster_cis,
    calibration_curve_bins,
    evaluate_predictions,
)
from crkp_ml.models import parse_pipeline_label
from crkp_ml.validation import nested_calibrated_oof, oof_predict_raw


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = load_cfg(args.config)
    cohort = load_development(cfg)
    out = output_dir(cfg, "07_probability_calibration")

    metric_rows = []
    prediction_frames = []
    bin_frames = []
    plot_store = {}

    for model_index, label in enumerate(cfg["analysis"]["calibration_models"]):
        model_name, sampling_name = parse_pipeline_label(label)
        raw_probability, raw_fold, _ = oof_predict_raw(
            cohort.X,
            cohort.y,
            cohort.groups,
            model_name,
            sampling_name,
            cfg,
        )
        for method_index, method in enumerate(["raw", "platt", "isotonic"]):
            if method == "raw":
                probability = raw_probability
                fold = raw_fold
                prediction = (probability >= 0.5).astype(int)
            else:
                result = nested_calibrated_oof(
                    cohort.X,
                    cohort.y,
                    cohort.groups,
                    model_name,
                    sampling_name,
                    cfg,
                    calibration_method=method,
                    threshold_objective="default_0.5",
                )
                probability = result.calibrated_probability
                fold = result.fold
                prediction = result.prediction

            row_label = f"{label}_{method}"
            metrics = evaluate_predictions(
                cohort.y,
                probability,
                prediction=prediction,
                threshold=0.5,
                model=label,
                analysis=f"natural_prevalence_{method}",
            )
            metrics.update(
                {
                    "probability_method": method,
                    "development_prevalence": float(cohort.y.mean()),
                }
            )
            metrics = add_auc_cluster_cis(
                metrics,
                cohort.y,
                probability,
                cohort.groups,
                cfg,
                seed_offset=30000 + model_index * 1000 + method_index * 100,
            )
            metric_rows.append(metrics)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_index": np.arange(len(cohort.y)),
                        cfg["data"]["group_column"]: cohort.groups,
                        "y_true": cohort.y,
                        "model": label,
                        "probability_method": method,
                        "outer_fold": fold,
                        "probability_CR": probability,
                        "prediction_CR": prediction,
                        "threshold": 0.5,
                    }
                )
            )
            bins = calibration_curve_bins(cohort.y, probability, n_bins=10)
            bins.insert(0, "probability_method", method)
            bins.insert(0, "model", label)
            bin_frames.append(bins)
            plot_store[(label, method)] = bins

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
        for method in ["raw", "platt", "isotonic"]:
            bins = plot_store[(label, method)]
            plt.plot(
                bins["mean_predicted_probability"],
                bins["observed_event_rate"],
                marker="o",
                label=method,
            )
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed resistance proportion")
        plt.title(f"Calibration: {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"calibration_{label.replace('/', '_')}.png", dpi=300)
        plt.close()

    metrics_frame = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    bins = pd.concat(bin_frames, ignore_index=True)
    metrics_frame.to_csv(out / "natural_prevalence_calibration_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "natural_prevalence_oof_predictions.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(out / "calibration_curve_bins.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "probability_calibration_results.xlsx", engine="openpyxl") as writer:
        metrics_frame.to_excel(writer, sheet_name="calibration_metrics", index=False)
        bins.to_excel(writer, sheet_name="calibration_bins", index=False)
    print(f"Saved calibration outputs to {out}")


if __name__ == "__main__":
    main()
