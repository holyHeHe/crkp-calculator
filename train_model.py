from __future__ import annotations

import json
import platform
from pathlib import Path

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from model_components import QuantileClipper

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "feike3.xlsx"
MODEL_DIR = BASE_DIR / "model"
SHEET_NAME = "Sheet2"
TARGET_COLUMN = "数据类别"
GROUP_COLUMN = "patient_SN"
RANDOM_STATE = 2025
N_SPLITS = 5
TARGET_SENSITIVITY = 0.85

FEATURES = [
    "Days of Indwelling Urinary Catheterization",
    "Vascular System Disease",
    "Respiratory System Disease",
    "Days of Carbapenems Use",
    "ICU Admission",
    "Metabolic Abnormality",
    "Respiratory Tract Infection",
    "Urinary System Disease",
    "Albumin",
    "Age",
    "Digestive System Disease",
    "Days of β-Lactamase Inhibitor Combinations Use",
]


def build_pipeline() -> ImbPipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("winsor", QuantileClipper(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )
    return ImbPipeline(
        steps=[
            ("preprocess", preprocess),
            ("enn", EditedNearestNeighbours(n_neighbors=3)),
            ("blsmote", BorderlineSMOTE(random_state=RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )


def probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p)).reshape(-1, 1)


def select_sensitivity_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float,
) -> float:
    thresholds = np.unique(np.r_[0.0, probabilities, 1.0])
    best: tuple[float, float] | None = None
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if sensitivity >= target_sensitivity:
            candidate = (specificity, float(threshold))
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise RuntimeError("No threshold met the target sensitivity.")
    return best[1]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}. Put feike3.xlsx next to train_model.py."
        )

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df.columns = [str(column).strip() for column in df.columns]
    required = set(FEATURES + [TARGET_COLUMN, GROUP_COLUMN])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    valid = target.notna()
    X = df.loc[valid, FEATURES].copy()
    y = target.loc[valid].astype(int).to_numpy()
    groups = df.loc[valid, GROUP_COLUMN].astype(str).to_numpy()
    source_prevalence = float(y.mean())

    print(f"Records: {len(y)}")
    print(f"Unique patients: {len(np.unique(groups))}")
    print(f"Natural CR prevalence: {source_prevalence:.6f}")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_raw = np.full(len(y), np.nan, dtype=float)

    for fold, (train_idx, validation_idx) in enumerate(cv.split(X, y, groups), start=1):
        fold_pipeline = build_pipeline()
        fold_pipeline.fit(X.iloc[train_idx], y[train_idx])
        oof_raw[validation_idx] = fold_pipeline.predict_proba(X.iloc[validation_idx])[:, 1]
        print(f"Completed fold {fold}/{N_SPLITS}")

    if np.isnan(oof_raw).any():
        raise RuntimeError("Incomplete out-of-fold predictions.")

    # Platt scaling on held-out, natural-prevalence predictions.
    calibrator = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        max_iter=10_000,
        random_state=RANDOM_STATE,
    )
    calibrator.fit(probability_to_logit(oof_raw), y)
    oof_calibrated = calibrator.predict_proba(probability_to_logit(oof_raw))[:, 1]

    operating_threshold = select_sensitivity_threshold(
        y,
        oof_calibrated,
        target_sensitivity=TARGET_SENSITIVITY,
    )

    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": final_pipeline,
        "calibrator": calibrator,
        "calibrator_input": "logit_raw_probability",
        "features": FEATURES,
        "calibration_method": "Platt scaling",
        "calibration_prevalence": source_prevalence,
        "calibration_reference": (
            "Natural prevalence of the 2014-2024 development cohort: "
            f"{int(y.sum())}/{len(y)} ({source_prevalence:.4%})"
        ),
        "operating_threshold": float(operating_threshold),
        "threshold_objective": (
            "Highest specificity among thresholds targeting sensitivity >=0.85 "
            "in patient-grouped out-of-fold development predictions"
        ),
        "model_name": (
            "Platt-calibrated ENN-BLSMOTE-XGBoost with 12 SFS-selected predictors"
        ),
        "prediction_anchor": "Time of culture-specimen collection (T0)",
        "intended_use": (
            "Antimicrobial-stewardship reassessment after identification as "
            "Klebsiella pneumoniae and before phenotypic AST results"
        ),
    }
    joblib.dump(bundle, MODEL_DIR / "model.joblib")

    metadata = {
        "records": int(len(y)),
        "unique_patients": int(len(np.unique(groups))),
        "resistant_records": int(y.sum()),
        "susceptible_records": int((1 - y).sum()),
        "natural_prevalence": source_prevalence,
        "raw_oof_brier": float(brier_score_loss(y, oof_raw)),
        "calibrated_oof_brier_fit_summary": float(
            brier_score_loss(y, oof_calibrated)
        ),
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_coefficient": float(calibrator.coef_[0, 0]),
        "operating_threshold": float(operating_threshold),
        "note": (
            "The calibrator is fitted from patient-grouped out-of-fold predictions. "
            "Use nested cross-validation or an untouched temporal cohort for unbiased "
            "manuscript calibration performance estimates."
        ),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "imbalanced_learn": imblearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    with (MODEL_DIR / "deployment_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved: {MODEL_DIR / 'model.joblib'}")
    print(f"Saved: {MODEL_DIR / 'deployment_metadata.json'}")


if __name__ == "__main__":
    main()
