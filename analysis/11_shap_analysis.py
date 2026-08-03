#!/usr/bin/env python3
"""Out-of-fold SHAP analysis for full and duration-restricted tree models."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crkp_ml.cli import load_cfg, load_development, parser
from crkp_ml.config import output_dir
from crkp_ml.data import define_feature_sets
from crkp_ml.models import parse_pipeline_label
from crkp_ml.shap_tools import oof_tree_shap


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = load_cfg(args.config)
    cohort = load_development(cfg)
    out = output_dir(cfg, "11_shap_analysis")

    feature_sets, _ = define_feature_sets(
        cohort.X,
        cfg["features"]["treatment_pathway_duration"],
        cfg["features"]["device_duration"],
    )
    summaries = {}
    for feature_set_name in cfg["analysis"]["shap_feature_sets"]:
        if feature_set_name not in feature_sets:
            raise ValueError(f"Unknown SHAP feature set: {feature_set_name}")
        X_subset = cohort.X[feature_sets[feature_set_name]].copy()
        for label in cfg["analysis"]["shap_models"]:
            model_name, sampling_name = parse_pipeline_label(label)
            print(f"Running OOF SHAP: {feature_set_name} | {label}")
            result = oof_tree_shap(
                X_subset,
                cohort.y,
                cohort.groups,
                model_name,
                sampling_name,
                cfg,
            )
            summary = result["summary"].copy()
            summary.insert(0, "feature_set", feature_set_name)
            summary.insert(0, "model", label)
            key = f"{feature_set_name}_{label}"
            summaries[key] = summary
            summary.to_csv(
                out / f"SHAP_summary_{key}.csv", index=False, encoding="utf-8-sig"
            )

            plt.figure()
            shap.summary_plot(
                result["shap_values"],
                result["transformed_features"],
                feature_names=result["feature_names"],
                plot_type="bar",
                max_display=int(cfg["analysis"]["shap_max_display"]),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(out / f"SHAP_bar_{key}.png", dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure()
            shap.summary_plot(
                result["shap_values"],
                result["transformed_features"],
                feature_names=result["feature_names"],
                max_display=int(cfg["analysis"]["shap_max_display"]),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(out / f"SHAP_dot_{key}.png", dpi=300, bbox_inches="tight")
            plt.close()

    with pd.ExcelWriter(out / "SHAP_results.xlsx", engine="openpyxl") as writer:
        for key, summary in summaries.items():
            summary.to_excel(writer, sheet_name=key[:31], index=False)
    print(
        "Saved SHAP outputs. Interpret all SHAP values as predictive associations/model "
        f"contributions, not causal effects: {out}"
    )


if __name__ == "__main__":
    main()
