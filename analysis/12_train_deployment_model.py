#!/usr/bin/env python3
"""Train the locked research deployment bundle using anchor-compatible binary inputs.

The bundle is for research deployment only. Reported performance must come from grouped OOF and
internal temporal analyses, not from apparent fit on the complete development cohort.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.calibration import fit_calibrator
from crkp_ml.cli import load_binary_development, load_cfg, parser
from crkp_ml.config import output_dir
from crkp_ml.io import save_json
from crkp_ml.metrics import evaluate_predictions
from crkp_ml.models import make_pipeline, parse_pipeline_label
from crkp_ml.selection import greedy_grouped_sfs
from crkp_ml.thresholds import select_threshold
from crkp_ml.validation import oof_predict_raw


def main() -> None:
    command = parser(__doc__)
    command.add_argument(
        "--run-sfs",
        action="store_true",
        help="Re-run grouped sequential forward selection instead of using configured final features.",
    )
    args = command.parse_args()
    cfg = load_cfg(args.config)
    cohort = load_binary_development(cfg)
    out = output_dir(cfg, "12_train_deployment_model")

    label = str(cfg["analysis"]["deployment_model"])
    model_name, sampling_name = parse_pipeline_label(label)
    run_sfs = bool(
        args.run_sfs
        or cfg["analysis"].get(
            "run_grouped_sfs",
            cfg["analysis"].get("run_nested_sfs", False),
        )
    )

    if run_sfs:
        candidate_features = list(cohort.X.columns)
        features, sfs_history = greedy_grouped_sfs(
            cohort.X,
            cohort.y,
            cohort.groups,
            candidate_features,
            model_name,
            sampling_name,
            cfg,
            max_features=int(cfg["analysis"]["sfs_max_features"]),
        )
        sfs_history.to_csv(out / "grouped_SFS_history.csv", index=False, encoding="utf-8-sig")
        feature_source = "grouped SFS rerun on the development cohort; do not use this step as an unbiased performance estimate"
    else:
        features = [str(feature).strip() for feature in cfg["features"]["deployment_features"]]
        feature_source = "final feature list supplied in the release configuration"

    missing = [feature for feature in features if feature not in cohort.X.columns]
    if missing:
        raise ValueError(
            f"Deployment features are missing from sheet {cfg['data']['binary_sheet']}: {missing}"
        )
    unsafe = [
        feature for feature in features
        if feature.startswith("Days of ") or feature.endswith("_days")
    ]
    if unsafe:
        raise ValueError(
            "Deployment inputs must be available at or before the prediction anchor. "
            f"Duration-style features were found: {unsafe}"
        )

    X_selected = cohort.X[features].copy()
    raw_oof, fold, _ = oof_predict_raw(
        X_selected,
        cohort.y,
        cohort.groups,
        model_name,
        sampling_name,
        cfg,
    )
    calibration_method = str(cfg["thresholds"]["calibration_method"])
    calibrator = fit_calibrator(raw_oof, cohort.y, calibration_method)
    calibrated_oof = calibrator.transform(raw_oof)

    threshold_objective = str(cfg["analysis"]["deployment_threshold_objective"])
    threshold, threshold_grid, selected_threshold = select_threshold(
        cohort.y,
        calibrated_oof,
        cfg,
        objective=threshold_objective,
        min_sensitivity=float(cfg["thresholds"]["primary_min_sensitivity"]),
        cost_vme=5.0,
        cost_me=1.0,
    )
    oof_prediction = (calibrated_oof >= threshold).astype(int)
    diagnostics = evaluate_predictions(
        cohort.y,
        calibrated_oof,
        prediction=oof_prediction,
        threshold=threshold,
        model=label,
        analysis="deployment_feature_set_grouped_OOF_diagnostic",
    )

    final_pipeline = make_pipeline(model_name, sampling_name, X_selected, cfg)
    final_pipeline.fit(X_selected, cohort.y)
    bundle = {
        "pipeline": final_pipeline,
        "calibrator": calibrator,
        "threshold": float(threshold),
        "features": features,
        "model_label": label,
        "base_learner": model_name,
        "sampling": sampling_name,
        "calibration_method": calibration_method,
        "threshold_objective": threshold_objective,
        "development_prevalence_CR": float(cohort.y.mean()),
        "prediction_anchor": "culture-specimen collection time (T0)",
        "intended_use": (
            "Early antimicrobial-stewardship reassessment after rapid K. pneumoniae "
            "identification and before phenotypic AST results."
        ),
        "probability_label": (
            "Platt-calibrated carbapenem-resistance probability referenced to the "
            "development-cohort prevalence and case mix."
        ),
        "warning": (
            "Research use only. This probability is not prevalence-invariant and does not "
            "replace phenotypic AST. Local calibration assessment, prospective validation, "
            "and where necessary prevalence updating or recalibration are required before use."
        ),
    }
    joblib.dump(bundle, out / "crkp_deployment_bundle.joblib")

    metadata = {key: value for key, value in bundle.items() if key not in {"pipeline", "calibrator"}}
    metadata.update(
        {
            "feature_source": feature_source,
            "n_development_records": len(cohort.y),
            "n_unique_patients": int(cohort.groups.nunique()),
            "grouped_OOF_diagnostics": diagnostics,
            "selected_threshold_training_summary": selected_threshold,
        }
    )
    save_json(metadata, out / "crkp_deployment_metadata.json")
    threshold_grid.to_csv(out / "deployment_threshold_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "row_index": np.arange(len(cohort.y)),
            cfg["data"]["group_column"]: cohort.groups,
            "y_true": cohort.y,
            "outer_fold": fold,
            "raw_probability_CR": raw_oof,
            "calibrated_probability_CR": calibrated_oof,
            "prediction_CR": oof_prediction,
        }
    ).to_csv(out / "deployment_feature_set_oof_diagnostics.csv", index=False, encoding="utf-8-sig")

    model_card = f"""# CRKP research deployment bundle

- Model: {label}
- Features: {len(features)} anchor-compatible inputs
- Calibration: {calibration_method}
- Threshold objective: {threshold_objective}
- Selected threshold: {threshold:.6f}
- Development CR prevalence: {cohort.y.mean():.4%}
- Prediction anchor: culture-specimen collection time (T0)

The displayed probability is calibrated to the development cohort and is not prevalence-invariant.
The model is intended for antimicrobial-stewardship support after rapid species identification and
before phenotypic AST; it is not an autonomous prescribing system or a replacement for AST.
"""
    (out / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    print(f"Saved deployment bundle to {out}")


if __name__ == "__main__":
    main()
