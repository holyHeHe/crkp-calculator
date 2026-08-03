#!/usr/bin/env python3
"""Same-center internal temporal validation with locked preprocessing, calibration, and thresholds."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.cli import load_cfg, load_development, load_temporal_aligned, parser
from crkp_ml.config import output_dir
from crkp_ml.data import Cohort, align_temporal_features, cohort_audit, filter_by_date
from crkp_ml.metrics import add_auc_cluster_cis, evaluate_predictions
from crkp_ml.models import parse_pipeline_label
from crkp_ml.validation import fit_development_and_apply_temporal


def subset_cohort(cohort: Cohort, mask) -> Cohort:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    return Cohort(
        frame=cohort.frame.iloc[indices].reset_index(drop=True),
        X=cohort.X.iloc[indices].reset_index(drop=True),
        y=cohort.y.iloc[indices].reset_index(drop=True),
        groups=cohort.groups.iloc[indices].reset_index(drop=True),
        time=cohort.time.iloc[indices].reset_index(drop=True),
        feature_columns=list(cohort.feature_columns),
    )


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = load_cfg(args.config)
    development = load_development(cfg)
    temporal = load_temporal_aligned(cfg, development)
    temporal, _, temporal_extra = align_temporal_features(development, temporal)
    temporal = filter_by_date(
        temporal,
        cfg["data"].get("temporal_start"),
        cfg["data"].get("temporal_end"),
    )

    overlap_patients = set(development.groups).intersection(set(temporal.groups))
    if cfg["data"].get("exclude_patient_overlap_in_temporal_validation", True):
        temporal = subset_cohort(temporal, ~temporal.groups.isin(overlap_patients))

    out = output_dir(cfg, "08_temporal_validation")
    metric_rows = []
    prediction_frames = []
    threshold_rows = []

    for model_index, label in enumerate(["XGBoost", "ENN-BLSMOTE-XGBoost"]):
        model_name, sampling_name = parse_pipeline_label(label)
        fitted = fit_development_and_apply_temporal(
            development.X,
            development.y,
            development.groups,
            temporal.X,
            model_name,
            sampling_name,
            cfg,
            calibration_method=str(cfg["thresholds"]["calibration_method"]),
            threshold_objective="sensitivity_constraint",
            min_sensitivity=float(cfg["thresholds"]["primary_min_sensitivity"]),
        )
        raw_probability = fitted["temporal_raw"]
        calibrated_probability = fitted["temporal_calibrated"]
        selected_threshold = float(fitted["threshold"])

        operating_points = [
            ("raw_default_0.5", raw_probability, (raw_probability >= 0.5).astype(int), 0.5),
            ("platt_default_0.5", calibrated_probability, (calibrated_probability >= 0.5).astype(int), 0.5),
            (
                "platt_historical_sensitivity_constraint",
                calibrated_probability,
                fitted["temporal_prediction"],
                selected_threshold,
            ),
        ]
        for point_index, (point_name, probability, prediction, threshold) in enumerate(operating_points):
            metrics = evaluate_predictions(
                temporal.y,
                probability,
                prediction=prediction,
                threshold=threshold,
                model=label,
                analysis=point_name,
            )
            metrics.update(
                {
                    "validation_type": "same-center internal temporal validation",
                    "development_n": len(development.y),
                    "temporal_n": len(temporal.y),
                    "temporal_time_min": temporal.time.min(),
                    "temporal_time_max": temporal.time.max(),
                    "patient_overlap_excluded": bool(
                        cfg["data"].get("exclude_patient_overlap_in_temporal_validation", True)
                    ),
                    "n_overlap_patients_identified": len(overlap_patients),
                }
            )
            metrics = add_auc_cluster_cis(
                metrics,
                temporal.y,
                probability,
                temporal.groups,
                cfg,
                seed_offset=40000 + model_index * 1000 + point_index * 100,
            )
            metric_rows.append(metrics)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_index": np.arange(len(temporal.y)),
                        cfg["data"]["group_column"]: temporal.groups,
                        cfg["data"]["time_column"]: temporal.time,
                        "y_true": temporal.y,
                        "model": label,
                        "operating_point": point_name,
                        "probability_CR": probability,
                        "prediction_CR": prediction,
                        "threshold": threshold,
                    }
                )
            )
        threshold_rows.append(
            {
                "model": label,
                "calibration_method": cfg["thresholds"]["calibration_method"],
                "threshold_objective": "sensitivity_constraint",
                "historical_selected_threshold": selected_threshold,
                **fitted["threshold_selected_row"],
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    development_audit = cohort_audit(
        development, cfg["data"]["group_column"], cfg["data"]["time_column"]
    )
    temporal_audit = cohort_audit(
        temporal, cfg["data"]["group_column"], cfg["data"]["time_column"]
    )
    extra_features = pd.DataFrame({"temporal_extra_feature_not_used": temporal_extra})

    metrics_frame.to_csv(out / "temporal_validation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "temporal_validation_predictions.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "temporal_validation_results.xlsx", engine="openpyxl") as writer:
        metrics_frame.to_excel(writer, sheet_name="metrics", index=False)
        thresholds.to_excel(writer, sheet_name="locked_thresholds", index=False)
        development_audit.to_excel(writer, sheet_name="development_audit", index=False)
        temporal_audit.to_excel(writer, sheet_name="temporal_audit", index=False)
        extra_features.to_excel(writer, sheet_name="unused_temporal_columns", index=False)
    print(f"Saved internal temporal-validation outputs to {out}")


if __name__ == "__main__":
    main()
