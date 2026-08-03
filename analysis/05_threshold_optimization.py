#!/usr/bin/env python3
"""Benchmark calibrated XGBoost threshold optimization against ENN-BLSMOTE resampling."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.cli import load_cfg, load_development, parser
from crkp_ml.config import output_dir
from crkp_ml.metrics import (
    add_auc_cluster_cis,
    evaluate_predictions,
    fbeta_from_counts,
    paired_cluster_bootstrap_difference,
)
from crkp_ml.validation import (
    nested_calibrated_oof,
    oof_predict_raw,
    resampled_inner_recall_targets,
)


def prediction_metric(name):
    def metric(y, pred):
        tn, fp, fn, tp = confusion_matrix(y, np.asarray(pred, dtype=int), labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        precision = tp / (tp + fp) if tp + fp else np.nan
        return float(
            {
                "Sensitivity": sensitivity,
                "PPV": precision,
                "Specificity": specificity,
                "F1": fbeta_from_counts(tp, fp, fn, 1),
                "F2": fbeta_from_counts(tp, fp, fn, 2),
                "VME": 1 - sensitivity,
                "ME": 1 - specificity,
            }[name]
        )
    return metric


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = load_cfg(args.config)
    cohort = load_development(cfg)
    out = output_dir(cfg, "05_threshold_optimization")
    calibration = str(cfg["thresholds"]["calibration_method"])

    settings = [
        ("default_0.5", {}),
        ("f2", {}),
        ("youden", {}),
        (
            "sensitivity_constraint",
            {"min_sensitivity": float(cfg["thresholds"]["primary_min_sensitivity"])},
        ),
    ]
    settings.extend(
        (
            "cost_sensitive",
            {"cost_vme": float(ratio), "cost_me": 1.0, "label_suffix": f"VME{ratio:g}_ME1"},
        )
        for ratio in cfg["thresholds"]["cost_vme_to_me_ratios"]
    )

    metrics_rows = []
    prediction_frames = []
    threshold_frames = []
    result_store = {}

    for index, (objective, options) in enumerate(settings):
        suffix = options.pop("label_suffix", None)
        label = f"XGBoost_{objective}" + (f"_{suffix}" if suffix else "")
        print(f"Running calibrated baseline: {label}")
        result = nested_calibrated_oof(
            cohort.X,
            cohort.y,
            cohort.groups,
            "XGBoost",
            "none",
            cfg,
            calibration_method=calibration,
            threshold_objective=objective,
            **options,
        )
        metrics = evaluate_predictions(
            cohort.y,
            result.calibrated_probability,
            prediction=result.prediction,
            threshold="outer-fold-specific",
            model=label,
            analysis="calibrated_baseline_threshold_optimization",
        )
        metrics.update(
            {
                "threshold_objective": objective,
                "calibration_method": calibration,
                "mean_selected_threshold": result.thresholds["selected_threshold"].mean(),
                "median_selected_threshold": result.thresholds["selected_threshold"].median(),
            }
        )
        metrics = add_auc_cluster_cis(
            metrics,
            cohort.y,
            result.calibrated_probability,
            cohort.groups,
            cfg,
            seed_offset=2000 + index * 100,
        )
        metrics_rows.append(metrics)
        result_store[label] = result
        threshold_frame = result.thresholds.copy()
        threshold_frame.insert(0, "operating_point", label)
        threshold_frames.append(threshold_frame)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": np.arange(len(cohort.y)),
                    cfg["data"]["group_column"]: cohort.groups,
                    "y_true": cohort.y,
                    "model": label,
                    "outer_fold": result.fold,
                    "raw_probability_CR": result.raw_probability,
                    "calibrated_probability_CR": result.calibrated_probability,
                    "prediction_CR": result.prediction,
                }
            )
        )

    # Resampled comparator at its conventional raw probability threshold of 0.50.
    enn_probability, enn_fold, _ = oof_predict_raw(
        cohort.X,
        cohort.y,
        cohort.groups,
        "XGBoost",
        "ENN-BLSMOTE",
        cfg,
    )
    enn_prediction = (enn_probability >= 0.5).astype(int)
    enn_metrics = evaluate_predictions(
        cohort.y,
        enn_probability,
        prediction=enn_prediction,
        threshold=0.5,
        model="ENN-BLSMOTE-XGBoost_default_0.5",
        analysis="resampling_comparator",
    )
    enn_metrics.update(
        {
            "threshold_objective": "resampled_default_0.5",
            "calibration_method": "raw",
            "mean_selected_threshold": 0.5,
            "median_selected_threshold": 0.5,
        }
    )
    enn_metrics = add_auc_cluster_cis(
        enn_metrics,
        cohort.y,
        enn_probability,
        cohort.groups,
        cfg,
        seed_offset=8000,
    )
    metrics_rows.append(enn_metrics)
    prediction_frames.append(
        pd.DataFrame(
            {
                "row_index": np.arange(len(cohort.y)),
                cfg["data"]["group_column"]: cohort.groups,
                "y_true": cohort.y,
                "model": "ENN-BLSMOTE-XGBoost_default_0.5",
                "outer_fold": enn_fold,
                "raw_probability_CR": enn_probability,
                "calibrated_probability_CR": np.nan,
                "prediction_CR": enn_prediction,
            }
        )
    )

    # Approximate matched-recall comparison: target is estimated only within each outer training fold.
    recall_targets = resampled_inner_recall_targets(cohort.X, cohort.y, cohort.groups, cfg)
    matched = nested_calibrated_oof(
        cohort.X,
        cohort.y,
        cohort.groups,
        "XGBoost",
        "none",
        cfg,
        calibration_method=calibration,
        threshold_objective="matched_recall",
        target_recall_by_fold=recall_targets,
    )
    matched_label = "XGBoost_calibrated_approx_matched_recall"
    matched_metrics = evaluate_predictions(
        cohort.y,
        matched.calibrated_probability,
        prediction=matched.prediction,
        threshold="outer-fold-specific",
        model=matched_label,
        analysis="approx_matched_recall_comparison",
    )
    matched_metrics.update(
        {
            "threshold_objective": "matched_recall",
            "calibration_method": calibration,
            "mean_selected_threshold": matched.thresholds["selected_threshold"].mean(),
            "median_selected_threshold": matched.thresholds["selected_threshold"].median(),
        }
    )
    matched_metrics = add_auc_cluster_cis(
        matched_metrics,
        cohort.y,
        matched.calibrated_probability,
        cohort.groups,
        cfg,
        seed_offset=9000,
    )
    metrics_rows.append(matched_metrics)
    threshold_frame = matched.thresholds.copy()
    threshold_frame.insert(0, "operating_point", matched_label)
    threshold_frames.append(threshold_frame)
    prediction_frames.append(
        pd.DataFrame(
            {
                "row_index": np.arange(len(cohort.y)),
                cfg["data"]["group_column"]: cohort.groups,
                "y_true": cohort.y,
                "model": matched_label,
                "outer_fold": matched.fold,
                "raw_probability_CR": matched.raw_probability,
                "calibrated_probability_CR": matched.calibrated_probability,
                "prediction_CR": matched.prediction,
            }
        )
    )

    # Paired patient-cluster bootstrap of threshold metrics for the approximate matched-recall pair.
    paired_rows = []
    n_bootstrap = int(cfg["validation"]["bootstrap_replicates"])
    seed = int(cfg["validation"]["bootstrap_random_state"])
    for offset, metric_name in enumerate(["Sensitivity", "PPV", "Specificity", "F1", "F2", "VME", "ME"]):
        comparison = paired_cluster_bootstrap_difference(
            cohort.y,
            matched.prediction,
            enn_prediction,
            cohort.groups,
            prediction_metric(metric_name),
            n_bootstrap,
            seed + 12000 + offset,
        )
        paired_rows.append(
            {
                "comparison": "calibrated XGBoost minus ENN-BLSMOTE-XGBoost",
                "metric": metric_name,
                **comparison,
                "note": "Positive favors XGBoost except VME and ME, where negative favors XGBoost.",
            }
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    thresholds = pd.concat(threshold_frames, ignore_index=True)
    paired = pd.DataFrame(paired_rows)
    metrics_frame.to_csv(out / "threshold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "threshold_oof_predictions.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(out / "selected_thresholds_by_fold.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out / "approx_matched_recall_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "threshold_optimization_results.xlsx", engine="openpyxl") as writer:
        metrics_frame.to_excel(writer, sheet_name="operating_points", index=False)
        thresholds.to_excel(writer, sheet_name="fold_thresholds", index=False)
        paired.to_excel(writer, sheet_name="matched_recall_comparison", index=False)
    print(f"Saved threshold-optimization outputs to {out}")


if __name__ == "__main__":
    main()
