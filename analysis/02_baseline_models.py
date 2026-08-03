#!/usr/bin/env python3
"""Evaluate the seven baseline algorithms with patient-grouped five-fold OOF prediction."""
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
from crkp_ml.validation import oof_predict_raw


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = load_cfg(args.config)
    cohort = load_development(cfg)
    out = output_dir(cfg, "02_baseline_models")

    metric_rows = []
    prediction_frames = []
    audit_frames = []
    for index, model_name in enumerate(cfg["analysis"]["baseline_models"]):
        print(f"Running baseline: {model_name}")
        probability, fold, audit = oof_predict_raw(
            cohort.X, cohort.y, cohort.groups, model_name, "none", cfg
        )
        prediction = (probability >= 0.5).astype(int)
        metrics = evaluate_predictions(
            cohort.y,
            probability,
            prediction=prediction,
            threshold=0.5,
            model=model_name,
            analysis="baseline_grouped_OOF",
        )
        metrics = add_auc_cluster_cis(
            metrics,
            cohort.y,
            probability,
            cohort.groups,
            cfg,
            seed_offset=index * 100,
        )
        metric_rows.append(metrics)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": np.arange(len(cohort.y)),
                    cfg["data"]["group_column"]: cohort.groups,
                    "y_true": cohort.y,
                    "model": model_name,
                    "sampling": "none",
                    "outer_fold": fold,
                    "probability_CR": probability,
                    "prediction_CR": prediction,
                    "threshold": 0.5,
                }
            )
        )
        audit.insert(0, "model", model_name)
        audit_frames.append(audit)

    metrics_frame = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    audits = pd.concat(audit_frames, ignore_index=True)
    metrics_frame.to_csv(out / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "baseline_oof_predictions.csv", index=False, encoding="utf-8-sig")
    write_excel(
        out / "baseline_results.xlsx",
        {"metrics": metrics_frame, "fold_audit": audits},
    )
    print(f"Saved baseline outputs to {out}")


if __name__ == "__main__":
    main()
