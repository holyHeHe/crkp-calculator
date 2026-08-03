#!/usr/bin/env python3
"""Audit the development cohort and document fold-specific preprocessing inputs.

This script does not export patient-level data. It writes aggregate cohort, missingness,
feature-type, and duplicate audits used to document the leakage-safe preprocessing workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.cli import load_cfg, load_development, parser
from crkp_ml.config import output_dir
from crkp_ml.data import cohort_audit, feature_type_table, missingness_table
from crkp_ml.io import save_json, write_excel


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = load_cfg(args.config)
    cohort = load_development(cfg)
    out = output_dir(cfg, "01_data_preprocessing")

    audit = cohort_audit(
        cohort,
        cfg["data"]["group_column"],
        cfg["data"]["time_column"],
    )
    missingness = missingness_table(cohort)
    feature_types = feature_type_table(cohort.X)
    workflow = [
        {
            "step": 1,
            "operation": "Define patient-grouped training and held-out folds",
            "fit_on": "Split before all data-dependent preprocessing",
        },
        {
            "step": 2,
            "operation": "Continuous median imputation and missingness indicators",
            "fit_on": "Training fold only",
        },
        {
            "step": 3,
            "operation": "Continuous 1st/99th percentile winsorization",
            "fit_on": "Training fold only",
        },
        {
            "step": 4,
            "operation": "Continuous Z-score standardization",
            "fit_on": "Training fold only",
        },
        {
            "step": 5,
            "operation": "Binary imputation / categorical one-hot encoding",
            "fit_on": "Training fold only; unknown held-out categories ignored",
        },
        {
            "step": 6,
            "operation": "Sampling, model fitting, calibration, and threshold selection",
            "fit_on": "Training fold only; held-out fold remains unsampled",
        },
    ]

    import pandas as pd

    write_excel(
        out / "data_preprocessing_audit.xlsx",
        {
            "cohort_audit": audit,
            "missingness": missingness,
            "feature_types": feature_types,
            "workflow": pd.DataFrame(workflow),
        },
    )
    save_json(
        {
            "n_records": len(cohort.frame),
            "n_unique_patients": int(cohort.groups.nunique()),
            "n_features": len(cohort.feature_columns),
            "CR_prevalence": float(cohort.y.mean()),
            "important_note": (
                "Repeated patient records are retained as separate eligible episodes and are "
                "kept within the same patient-grouped validation fold. The manuscript must not "
                "describe all records as independent patients unless a one-record-per-patient "
                "cohort is created and every analysis is rerun."
            ),
        },
        out / "audit_summary.json",
    )
    print(f"Saved audit outputs to {out}")


if __name__ == "__main__":
    main()
