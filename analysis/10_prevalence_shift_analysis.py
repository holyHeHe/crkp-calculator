#!/usr/bin/env python3
"""Fourteen-ratio performance sensitivity and five-ratio calibration/prior-correction analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.calibration import prevalence_prior_correction
from crkp_ml.cli import load_cfg, load_development, parser
from crkp_ml.config import output_dir, resolve_path
from crkp_ml.metrics import (
    add_auc_cluster_cis,
    calibration_curve_bins,
    evaluate_predictions,
)
from crkp_ml.validation import nested_calibrated_oof, oof_predict_raw


def ratio_indices(y, ratio: int, positive_order, negative_order):
    n_positive = len(positive_order)
    n_negative = len(negative_order)
    if n_negative >= ratio * n_positive:
        selected_positive = positive_order
        selected_negative = negative_order[: ratio * n_positive]
    else:
        selected_positive_count = n_negative // ratio
        selected_positive = positive_order[:selected_positive_count]
        selected_negative = negative_order[: selected_positive_count * ratio]
    indices = np.concatenate([selected_positive, selected_negative])
    return np.sort(indices)


def load_or_generate_natural_predictions(cfg, cohort):
    calibration_path = (
        resolve_path(cfg, cfg["output"]["root"])
        / "07_probability_calibration"
        / "natural_prevalence_oof_predictions.csv"
    )
    if calibration_path.exists():
        frame = pd.read_csv(calibration_path)
        wanted = frame[frame["model"] == "ENN-BLSMOTE-XGBoost"].copy()
        raw = wanted[wanted["probability_method"] == "raw"].sort_values("row_index")
        platt = wanted[wanted["probability_method"] == "platt"].sort_values("row_index")
        if len(raw) == len(cohort.y) and len(platt) == len(cohort.y):
            return raw["probability_CR"].to_numpy(), platt["probability_CR"].to_numpy()

    raw_probability, _, _ = oof_predict_raw(
        cohort.X,
        cohort.y,
        cohort.groups,
        "XGBoost",
        "ENN-BLSMOTE",
        cfg,
    )
    platt_result = nested_calibrated_oof(
        cohort.X,
        cohort.y,
        cohort.groups,
        "XGBoost",
        "ENN-BLSMOTE",
        cfg,
        calibration_method="platt",
        threshold_objective="default_0.5",
    )
    return raw_probability, platt_result.calibrated_probability


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = load_cfg(args.config)
    cohort = load_development(cfg)
    out = output_dir(cfg, "10_prevalence_shift_analysis")
    raw_probability, locked_platt_probability = load_or_generate_natural_predictions(cfg, cohort)

    rng = np.random.default_rng(int(cfg["validation"]["cv_random_state"]) + 4101)
    positive_order = np.flatnonzero(cohort.y.to_numpy() == 1)
    negative_order = np.flatnonzero(cohort.y.to_numpy() == 0)
    rng.shuffle(positive_order)
    rng.shuffle(negative_order)

    all_ratios = [int(value) for value in cfg["analysis"]["prevalence_ratios_all"]]
    calibration_ratios = set(int(value) for value in cfg["analysis"]["prevalence_ratios_calibration"])
    source_prevalence = float(cohort.y.mean())

    metric_rows = []
    prediction_frames = []
    audit_rows = []
    bin_frames = []
    for ratio_index, ratio in enumerate(all_ratios):
        indices = ratio_indices(cohort.y.to_numpy(), ratio, positive_order, negative_order)
        y_subset = cohort.y.iloc[indices].reset_index(drop=True)
        groups_subset = cohort.groups.iloc[indices].reset_index(drop=True)
        raw_subset = raw_probability[indices]
        locked_subset = locked_platt_probability[indices]
        target_prevalence = float(y_subset.mean())
        methods = {
            "raw": raw_subset,
            "locked_nested_platt": locked_subset,
        }
        if ratio in calibration_ratios:
            methods["prior_corrected_platt"] = prevalence_prior_correction(
                locked_subset,
                source_prevalence=source_prevalence,
                target_prevalence=target_prevalence,
            )

        audit_rows.append(
            {
                "CR_CS_ratio": f"1:{ratio}",
                "n_records": len(indices),
                "n_unique_patients": groups_subset.nunique(),
                "n_CR": int(y_subset.sum()),
                "n_CS": int((1 - y_subset).sum()),
                "CR_prevalence": target_prevalence,
                "duplicates_introduced_by_ratio_construction": 0,
            }
        )
        for method_index, (method, probability) in enumerate(methods.items()):
            prediction = (probability >= 0.5).astype(int)
            metrics = evaluate_predictions(
                y_subset,
                probability,
                prediction=prediction,
                threshold=0.5,
                model="ENN-BLSMOTE-XGBoost",
                analysis=method,
            )
            metrics.update(
                {
                    "CR_CS_ratio": f"1:{ratio}",
                    "probability_method": method,
                    "source_development_prevalence": source_prevalence,
                }
            )
            metrics = add_auc_cluster_cis(
                metrics,
                y_subset,
                probability,
                groups_subset,
                cfg,
                seed_offset=60000 + ratio_index * 1000 + method_index * 100,
            )
            metric_rows.append(metrics)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "original_row_index": indices,
                        cfg["data"]["group_column"]: groups_subset,
                        "y_true": y_subset,
                        "CR_CS_ratio": f"1:{ratio}",
                        "probability_method": method,
                        "probability_CR": probability,
                        "prediction_CR": prediction,
                        "threshold": 0.5,
                    }
                )
            )
            bins = calibration_curve_bins(y_subset, probability, 10)
            bins.insert(0, "probability_method", method)
            bins.insert(0, "CR_CS_ratio", f"1:{ratio}")
            bin_frames.append(bins)

    metrics_frame = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    bins = pd.concat(bin_frames, ignore_index=True)
    selected = metrics_frame[
        metrics_frame["CR_CS_ratio"].isin([f"1:{r}" for r in calibration_ratios])
    ].copy()
    all_performance = metrics_frame[
        metrics_frame["probability_method"].isin(["raw", "locked_nested_platt"])
    ].copy()

    metrics_frame.to_csv(out / "all_ratio_metrics.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out / "selected_five_ratio_calibration_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "ratio_predictions.csv", index=False, encoding="utf-8-sig")
    bins.to_csv(out / "ratio_calibration_bins.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "prevalence_shift_results.xlsx", engine="openpyxl") as writer:
        all_performance.to_excel(writer, sheet_name="14_ratio_performance", index=False)
        selected.to_excel(writer, sheet_name="5_ratio_calibration", index=False)
        audit.to_excel(writer, sheet_name="cohort_audit", index=False)
        bins.to_excel(writer, sheet_name="calibration_bins", index=False)
    print(f"Saved prevalence-shift outputs to {out}")


if __name__ == "__main__":
    main()
