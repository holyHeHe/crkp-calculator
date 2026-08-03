#!/usr/bin/env python3
"""Sensitivity analysis excluding treatment-pathway, device-duration, and all duration features."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.cli import load_cfg, load_development, parser
from crkp_ml.config import output_dir
from crkp_ml.data import define_feature_sets
from crkp_ml.metrics import add_auc_cluster_cis, evaluate_predictions
from crkp_ml.models import parse_pipeline_label
from crkp_ml.validation import oof_predict_raw


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = load_cfg(args.config)
    cohort = load_development(cfg)
    out = output_dir(cfg, "06_feature_restriction_analysis")

    feature_sets, removed = define_feature_sets(
        cohort.X,
        cfg["features"]["treatment_pathway_duration"],
        cfg["features"]["device_duration"],
    )
    metric_rows = []
    prediction_frames = []
    audit_rows = []

    for feature_index, (feature_set_name, columns) in enumerate(feature_sets.items()):
        for model_index, label in enumerate(cfg["analysis"]["feature_restriction_models"]):
            model_name, sampling_name = parse_pipeline_label(label)
            print(f"Running {feature_set_name} | {label} | {len(columns)} features")
            X_subset = cohort.X[columns].copy()
            probability, fold, _ = oof_predict_raw(
                X_subset,
                cohort.y,
                cohort.groups,
                model_name,
                sampling_name,
                cfg,
            )
            prediction = (probability >= 0.5).astype(int)
            metrics = evaluate_predictions(
                cohort.y,
                probability,
                prediction=prediction,
                threshold=0.5,
                model=label,
                analysis=feature_set_name,
            )
            metrics.update(
                {
                    "feature_set": feature_set_name,
                    "n_features": len(columns),
                    "base_learner": model_name,
                    "sampling": sampling_name,
                }
            )
            metrics = add_auc_cluster_cis(
                metrics,
                cohort.y,
                probability,
                cohort.groups,
                cfg,
                seed_offset=20000 + feature_index * 1000 + model_index * 100,
            )
            metric_rows.append(metrics)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_index": np.arange(len(cohort.y)),
                        cfg["data"]["group_column"]: cohort.groups,
                        "y_true": cohort.y,
                        "feature_set": feature_set_name,
                        "model": label,
                        "outer_fold": fold,
                        "probability_CR": probability,
                        "prediction_CR": prediction,
                        "threshold": 0.5,
                    }
                )
            )
            audit_rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": label,
                    "n_features": len(columns),
                    "features": " | ".join(columns),
                }
            )

    removed_rows = []
    for group_name, features in removed.items():
        for feature in features:
            removed_rows.append({"removed_group": group_name, "feature": feature})

    metrics_frame = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    removed_frame = pd.DataFrame(removed_rows)
    metrics_frame.to_csv(out / "feature_restriction_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "feature_restriction_oof_predictions.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "feature_restriction_results.xlsx", engine="openpyxl") as writer:
        metrics_frame.to_excel(writer, sheet_name="metrics", index=False)
        audit.to_excel(writer, sheet_name="feature_sets", index=False)
        removed_frame.to_excel(writer, sheet_name="removed_features", index=False)
    print(f"Saved feature-restriction outputs to {out}")


if __name__ == "__main__":
    main()
