#!/usr/bin/env python3
"""Paired patient-cluster bootstrap comparisons with BH-FDR and Bonferroni adjustment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.cli import load_cfg, load_development, parser
from crkp_ml.config import output_dir, resolve_path
from crkp_ml.metrics import fbeta_from_counts, paired_cluster_bootstrap_difference


def threshold_metric(name):
    def metric(y, pred):
        tn, fp, fn, tp = confusion_matrix(y, np.asarray(pred, dtype=int), labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        precision = tp / (tp + fp) if tp + fp else np.nan
        mapping = {
            "Sensitivity": sensitivity,
            "PPV": precision,
            "Specificity": specificity,
            "F1": fbeta_from_counts(tp, fp, fn, beta=1),
            "F2": fbeta_from_counts(tp, fp, fn, beta=2),
            "VME": 1 - sensitivity,
            "ME": 1 - specificity,
        }
        return float(mapping[name])
    return metric


def adjust_family(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    valid = frame["bootstrap_p"].notna()
    frame["BH_adjusted_p"] = np.nan
    frame["Bonferroni_adjusted_p"] = np.nan
    if valid.any():
        p_values = frame.loc[valid, "bootstrap_p"].to_numpy()
        frame.loc[valid, "BH_adjusted_p"] = multipletests(p_values, method="fdr_bh")[1]
        frame.loc[valid, "Bonferroni_adjusted_p"] = multipletests(p_values, method="bonferroni")[1]
    return frame


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = load_cfg(args.config)
    cohort = load_development(cfg)
    out = output_dir(cfg, "04_multiple_comparisons")

    baseline_path = resolve_path(cfg, cfg["output"]["root"]) / "02_baseline_models" / "baseline_oof_predictions.csv"
    sampling_path = resolve_path(cfg, cfg["output"]["root"]) / "03_sampling_experiments" / "sampling_oof_predictions.csv"
    if not baseline_path.exists() or not sampling_path.exists():
        raise FileNotFoundError(
            "Run 02_baseline_models.py and 03_sampling_experiments.py before multiple comparisons."
        )

    baseline = pd.read_csv(baseline_path)
    sampling = pd.read_csv(sampling_path)
    group_col = cfg["data"]["group_column"]
    n_bootstrap = int(cfg["validation"]["bootstrap_replicates"])
    seed = int(cfg["validation"]["bootstrap_random_state"])

    baseline_map = {
        name: group.sort_values("row_index")
        for name, group in baseline.groupby("model")
    }
    sampling_map = {
        name: group.sort_values("row_index")
        for name, group in sampling.groupby("model")
    }
    rows = []

    def compare(family, candidate, reference, metric_name, value_column, metric_function, seed_offset):
        candidate_frame = baseline_map.get(candidate, sampling_map.get(candidate))
        reference_frame = baseline_map.get(reference, sampling_map.get(reference))
        if candidate_frame is None or reference_frame is None:
            return
        if not np.array_equal(candidate_frame["row_index"].to_numpy(), reference_frame["row_index"].to_numpy()):
            raise ValueError(f"OOF rows are not aligned for {candidate} vs {reference}")
        result = paired_cluster_bootstrap_difference(
            candidate_frame["y_true"].to_numpy(),
            candidate_frame[value_column].to_numpy(),
            reference_frame[value_column].to_numpy(),
            candidate_frame[group_col].astype(str).to_numpy(),
            metric_function,
            n_bootstrap,
            seed + seed_offset,
        )
        rows.append(
            {
                "family": family,
                "candidate": candidate,
                "reference": reference,
                "metric": metric_name,
                "difference_candidate_minus_reference": result["difference"],
                "difference_CI_low": result["CI_low"],
                "difference_CI_high": result["CI_high"],
                "bootstrap_p": result["p"],
                "n_bootstrap": result["n"],
            }
        )

    # Family 1: baseline discrimination versus XGBoost.
    for index, candidate in enumerate([m for m in baseline_map if m != "XGBoost"]):
        compare("baseline_discrimination", candidate, "XGBoost", "ROC_AUC", "probability_CR", roc_auc_score, 1000 + index * 10)
        compare("baseline_discrimination", candidate, "XGBoost", "PR_AUC", "probability_CR", average_precision_score, 1001 + index * 10)

    # Family 2: each sampling pipeline versus its unsampled learner for discrimination.
    for index, (candidate, candidate_frame) in enumerate(sampling_map.items()):
        reference = str(candidate_frame["base_learner"].iloc[0])
        if reference not in baseline_map:
            continue
        compare("sampling_vs_unsampled_discrimination", candidate, reference, "ROC_AUC", "probability_CR", roc_auc_score, 5000 + index * 10)
        compare("sampling_vs_unsampled_discrimination", candidate, reference, "PR_AUC", "probability_CR", average_precision_score, 5001 + index * 10)

    # Family 3: key XGBoost versus ENN-BLSMOTE-XGBoost default-threshold comparison.
    candidate = "ENN-BLSMOTE-XGBoost"
    reference = "XGBoost"
    compare("key_pipeline_comparison", candidate, reference, "ROC_AUC", "probability_CR", roc_auc_score, 9000)
    compare("key_pipeline_comparison", candidate, reference, "PR_AUC", "probability_CR", average_precision_score, 9001)
    compare("key_pipeline_comparison", candidate, reference, "Brier", "probability_CR", brier_score_loss, 9002)
    for offset, metric_name in enumerate(["Sensitivity", "PPV", "Specificity", "F1", "F2", "VME", "ME"], start=10):
        compare(
            "key_pipeline_comparison",
            candidate,
            reference,
            metric_name,
            "prediction_CR",
            threshold_metric(metric_name),
            9000 + offset,
        )

    comparison_frame = pd.DataFrame(rows)
    adjusted = pd.concat(
        [adjust_family(group) for _, group in comparison_frame.groupby("family")],
        ignore_index=True,
    )
    key = adjusted[adjusted["family"].isin(["baseline_discrimination", "key_pipeline_comparison"])].copy()
    adjusted.to_csv(out / "all_adjusted_comparisons.csv", index=False, encoding="utf-8-sig")
    key.to_csv(out / "reviewer_key_comparisons.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out / "multiple_comparisons.xlsx", engine="openpyxl") as writer:
        key.to_excel(writer, sheet_name="key_comparisons", index=False)
        adjusted.to_excel(writer, sheet_name="all_comparisons", index=False)
    print(f"Saved multiplicity-adjusted comparisons to {out}")


if __name__ == "__main__":
    main()
