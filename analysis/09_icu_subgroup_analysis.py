#!/usr/bin/env python3
"""ICU-only refitting, patient-cluster uncertainty, and within-ICU SHAP associations."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.cli import load_cfg, load_development, parser
from crkp_ml.config import output_dir
from crkp_ml.data import define_feature_sets, find_icu_column
from crkp_ml.metrics import (
    add_auc_cluster_cis,
    evaluate_predictions,
    fbeta_from_counts,
    paired_cluster_bootstrap_difference,
)
from crkp_ml.models import parse_pipeline_label
from crkp_ml.shap_tools import oof_tree_shap
from crkp_ml.validation import oof_predict_raw


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
    command = parser(__doc__)
    command.add_argument("--skip-shap", action="store_true", help="Skip the ICU-only SHAP step.")
    args = command.parse_args()
    cfg = load_cfg(args.config)
    cohort = load_development(cfg)
    out = output_dir(cfg, "09_icu_subgroup_analysis")

    icu_column = find_icu_column(cohort.X.columns, cfg["data"]["icu_columns"])
    icu_flag = pd.to_numeric(cohort.X[icu_column], errors="coerce").fillna(0).astype(int)
    composition_rows = []
    for value, label in [(1, "ICU"), (0, "non-ICU")]:
        mask = icu_flag == value
        composition_rows.append(
            {
                "subgroup": label,
                "n_records": int(mask.sum()),
                "n_unique_patients": int(cohort.groups[mask].nunique()),
                "n_CR": int(cohort.y[mask].sum()),
                "n_CS": int((1 - cohort.y[mask]).sum()),
                "CR_prevalence": float(cohort.y[mask].mean()),
            }
        )

    icu_indices = np.flatnonzero((icu_flag == 1).to_numpy())
    X_icu = cohort.X.iloc[icu_indices].drop(columns=[icu_column]).reset_index(drop=True)
    y_icu = cohort.y.iloc[icu_indices].reset_index(drop=True)
    groups_icu = cohort.groups.iloc[icu_indices].reset_index(drop=True)

    metric_rows = []
    prediction_frames = []
    store = {}
    for model_index, label in enumerate(cfg["analysis"]["icu_models"]):
        model_name, sampling_name = parse_pipeline_label(label)
        print(f"Running ICU-only refit: {label}")
        probability, fold, _ = oof_predict_raw(
            X_icu, y_icu, groups_icu, model_name, sampling_name, cfg
        )
        prediction = (probability >= 0.5).astype(int)
        metrics = evaluate_predictions(
            y_icu,
            probability,
            prediction=prediction,
            threshold=0.5,
            model=label,
            analysis="within_ICU_refit",
        )
        metrics.update(
            {
                "n_unique_patients": int(groups_icu.nunique()),
                "ICU_indicator_removed": True,
            }
        )
        metrics = add_auc_cluster_cis(
            metrics,
            y_icu,
            probability,
            groups_icu,
            cfg,
            seed_offset=50000 + model_index * 1000,
        )
        metric_rows.append(metrics)
        store[label] = (probability, prediction)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "subgroup_row_index": np.arange(len(y_icu)),
                    "original_row_index": icu_indices,
                    cfg["data"]["group_column"]: groups_icu,
                    "y_true": y_icu,
                    "model": label,
                    "outer_fold": fold,
                    "probability_CR": probability,
                    "prediction_CR": prediction,
                    "threshold": 0.5,
                }
            )
        )

    paired_rows = []
    if "XGBoost" in store and "ENN-BLSMOTE-XGBoost" in store:
        x_probability, x_prediction = store["XGBoost"]
        e_probability, e_prediction = store["ENN-BLSMOTE-XGBoost"]
        n_bootstrap = int(cfg["validation"]["bootstrap_replicates"])
        seed = int(cfg["validation"]["bootstrap_random_state"])
        for offset, (metric_name, first, second, metric_function) in enumerate(
            [
                ("ROC_AUC", x_probability, e_probability, roc_auc_score),
                ("PR_AUC", x_probability, e_probability, average_precision_score),
            ]
        ):
            comparison = paired_cluster_bootstrap_difference(
                y_icu, first, second, groups_icu, metric_function, n_bootstrap, seed + 52000 + offset
            )
            paired_rows.append({"metric": metric_name, **comparison})
        for offset, metric_name in enumerate(["Sensitivity", "PPV", "Specificity", "F1", "F2", "VME", "ME"], start=10):
            comparison = paired_cluster_bootstrap_difference(
                y_icu,
                x_prediction,
                e_prediction,
                groups_icu,
                prediction_metric(metric_name),
                n_bootstrap,
                seed + 52000 + offset,
            )
            paired_rows.append({"metric": metric_name, **comparison})

    shap_summaries = {}
    if not args.skip_shap:
        feature_sets, _ = define_feature_sets(
            X_icu,
            cfg["features"]["treatment_pathway_duration"],
            cfg["features"]["device_duration"],
        )
        X_shap = X_icu[feature_sets["D_no_duration"]]
        for label in cfg["analysis"]["icu_models"]:
            model_name, sampling_name = parse_pipeline_label(label)
            print(f"Running ICU-only OOF SHAP: {label}")
            shap_result = oof_tree_shap(
                X_shap, y_icu, groups_icu, model_name, sampling_name, cfg
            )
            summary = shap_result["summary"].copy()
            summary.insert(0, "model", label)
            summary.insert(1, "feature_set", "ICU_only_no_duration")
            shap_summaries[label] = summary
            plt.figure()
            shap.summary_plot(
                shap_result["shap_values"],
                shap_result["transformed_features"],
                feature_names=shap_result["feature_names"],
                plot_type="bar",
                max_display=int(cfg["analysis"]["shap_max_display"]),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(out / f"ICU_only_SHAP_{label}.png", dpi=300, bbox_inches="tight")
            plt.close()

    metrics_frame = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    paired = pd.DataFrame(paired_rows)
    composition = pd.DataFrame(composition_rows)
    metrics_frame.to_csv(out / "icu_refit_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "icu_refit_oof_predictions.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out / "icu_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "icu_subgroup_results.xlsx", engine="openpyxl") as writer:
        composition.to_excel(writer, sheet_name="cohort_composition", index=False)
        metrics_frame.to_excel(writer, sheet_name="ICU_refit_metrics", index=False)
        paired.to_excel(writer, sheet_name="paired_comparisons", index=False)
        for label, summary in shap_summaries.items():
            summary.to_excel(writer, sheet_name=f"SHAP_{label}"[:31], index=False)
    print(f"Saved ICU subgroup outputs to {out}")


if __name__ == "__main__":
    main()
