#!/usr/bin/env python3
"""Evaluate oversampling, undersampling, hybrid sampling, and the GBDT extension."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.cli import load_cfg, load_development, parser
from crkp_ml.config import output_dir
from crkp_ml.io import write_excel
from crkp_ml.metrics import add_auc_cluster_cis, evaluate_predictions
from crkp_ml.models import (
    HYBRID_SAMPLING_NAMES,
    OVERSAMPLING_NAMES,
    UNDERSAMPLING_NAMES,
    pipeline_label,
)
from crkp_ml.validation import oof_predict_raw


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = load_cfg(args.config)
    cohort = load_development(cfg)
    out = output_dir(cfg, "03_sampling_experiments")

    primary_learners = list(cfg["analysis"]["primary_sampling_learners"])
    configured_sampling = cfg["analysis"].get("sampling_strategies")
    sampling_names = (
        list(configured_sampling)
        if configured_sampling
        else OVERSAMPLING_NAMES + UNDERSAMPLING_NAMES + HYBRID_SAMPLING_NAMES
    )
    jobs = [(model, sampling) for model in primary_learners for sampling in sampling_names]
    if cfg["analysis"].get("include_gbdt_sampling_extension", True):
        jobs.extend(("GBDT", sampling) for sampling in sampling_names)

    metric_rows = []
    prediction_frames = []
    errors = []
    for job_index, (model_name, sampling_name) in enumerate(jobs):
        label = pipeline_label(model_name, sampling_name)
        print(f"Running sampling pipeline: {label}")
        try:
            probability, fold, _ = oof_predict_raw(
                cohort.X, cohort.y, cohort.groups, model_name, sampling_name, cfg
            )
            prediction = (probability >= 0.5).astype(int)
            metrics = evaluate_predictions(
                cohort.y,
                probability,
                prediction=prediction,
                threshold=0.5,
                model=label,
                analysis="sampling_grouped_OOF",
            )
            metrics.update({"base_learner": model_name, "sampling": sampling_name})
            metrics = add_auc_cluster_cis(
                metrics,
                cohort.y,
                probability,
                cohort.groups,
                cfg,
                seed_offset=1000 + job_index * 100,
            )
            metric_rows.append(metrics)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_index": np.arange(len(cohort.y)),
                        cfg["data"]["group_column"]: cohort.groups,
                        "y_true": cohort.y,
                        "model": label,
                        "base_learner": model_name,
                        "sampling": sampling_name,
                        "outer_fold": fold,
                        "probability_CR": probability,
                        "prediction_CR": prediction,
                        "threshold": 0.5,
                    }
                )
            )
        except Exception as error:
            errors.append(
                {
                    "model": label,
                    "base_learner": model_name,
                    "sampling": sampling_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(f"FAILED {label}: {error}")

    metrics_frame = pd.DataFrame(metric_rows)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames else pd.DataFrame()
    )
    error_frame = pd.DataFrame(errors)
    metrics_frame.to_csv(out / "sampling_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "sampling_oof_predictions.csv", index=False, encoding="utf-8-sig")
    write_excel(
        out / "sampling_results.xlsx",
        {
            "metrics": metrics_frame,
            "oversampling": metrics_frame[metrics_frame["sampling"].isin(OVERSAMPLING_NAMES)],
            "undersampling": metrics_frame[metrics_frame["sampling"].isin(UNDERSAMPLING_NAMES)],
            "hybrid": metrics_frame[metrics_frame["sampling"].isin(HYBRID_SAMPLING_NAMES)],
            "errors": error_frame,
        },
    )
    print(f"Saved sampling outputs to {out}")


if __name__ == "__main__":
    main()
