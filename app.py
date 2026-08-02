from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# Required so joblib can import the custom transformer stored in the pipeline.
from model_components import QuantileClipper  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
IMAGE_PATH = BASE_DIR / "assets" / "SFS12-2.jpg"

st.set_page_config(
    page_title="CRKP Resistance Prediction Tool",
    page_icon="🧫",
    layout="wide",
)

DISPLAY_NAMES = {
    "Days of Indwelling Urinary Catheterization":
        "Urinary-catheter duration",
    "Vascular System Disease":
        "Vascular system disease",
    "Respiratory System Disease":
        "Respiratory system disease",
    "Days of Carbapenems Use":
        "Carbapenem exposure duration",
    "ICU Admission":
        "ICU admission",
    "Metabolic Abnormality":
        "Metabolic abnormality",
    "Respiratory Tract Infection":
        "Respiratory tract infection",
    "Urinary System Disease":
        "Urinary system disease",
    "Albumin":
        "Albumin",
    "Age":
        "Age",
    "Digestive System Disease":
        "Digestive system disease",
    "Days of β-Lactamase Inhibitor Combinations Use":
        "BL/BLI exposure duration",
}

BINARY_FEATURES = {
    "Vascular System Disease",
    "Respiratory System Disease",
    "ICU Admission",
    "Metabolic Abnormality",
    "Respiratory Tract Infection",
    "Urinary System Disease",
    "Digestive System Disease",
}


@st.cache_resource
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def encode_yes_no(value: str) -> int:
    return 1 if value == "Yes" else 0


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + np.exp(-value))


def probability_to_logit(probability: float) -> np.ndarray:
    probability = float(np.clip(probability, 1e-8, 1.0 - 1e-8))
    return np.array(
        [[np.log(probability / (1.0 - probability))]],
        dtype=float,
    )


def adjust_for_target_prevalence(
    source_probability: float,
    source_prevalence: float,
    target_prevalence: float,
) -> float:
    """Prior-prevalence adjustment under the label-shift assumption."""
    probability = float(
        np.clip(source_probability, 1e-8, 1.0 - 1e-8)
    )
    source = float(
        np.clip(source_prevalence, 1e-8, 1.0 - 1e-8)
    )
    target = float(
        np.clip(target_prevalence, 1e-8, 1.0 - 1e-8)
    )

    positive = probability * (target / source)
    negative = (1.0 - probability) * (
        (1.0 - target) / (1.0 - source)
    )
    return positive / (positive + negative)


def format_patient_value(feature: str, value: float) -> str:
    if feature in BINARY_FEATURES:
        return "Yes" if int(value) == 1 else "No"
    if feature == "Albumin":
        return f"{float(value):.1f} g/L"
    if feature == "Age":
        return f"{int(value)} years"
    return f"{int(value)} days"


def calculate_patient_contributions(
    pipeline,
    calibrator,
    x_input: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, float, float]:
    """
    Calculate XGBoost SHAP contributions and map them to the
    Platt-calibrated log-odds scale.

    The XGBoost contributions sum to the raw model margin. Because the
    Platt calibrator was fitted as:
        calibrated_logit = intercept + coefficient * raw_margin
    each feature contribution can be multiplied by the Platt coefficient.
    """
    preprocess = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]

    transformed = preprocess.transform(x_input)
    dmatrix = xgb.DMatrix(
        transformed,
        feature_names=feature_names,
    )

    contribution_row = classifier.get_booster().predict(
        dmatrix,
        pred_contribs=True,
    )[0]

    raw_feature_contributions = contribution_row[:-1]
    raw_bias = float(contribution_row[-1])

    platt_coefficient = float(calibrator.coef_[0, 0])
    platt_intercept = float(calibrator.intercept_[0])

    calibrated_feature_contributions = (
        raw_feature_contributions * platt_coefficient
    )
    calibrated_baseline_log_odds = (
        platt_intercept + platt_coefficient * raw_bias
    )
    reconstructed_log_odds = (
        calibrated_baseline_log_odds
        + float(np.sum(calibrated_feature_contributions))
    )

    rows = []
    for feature, contribution in zip(
        feature_names,
        calibrated_feature_contributions,
    ):
        patient_value = x_input.iloc[0][feature]
        rows.append(
            {
                "Feature": DISPLAY_NAMES.get(feature, feature),
                "Patient value": format_patient_value(
                    feature,
                    patient_value,
                ),
                "Contribution": float(contribution),
                "Direction": (
                    "Increases estimated resistance"
                    if contribution >= 0
                    else "Decreases estimated resistance"
                ),
                "Absolute contribution": abs(float(contribution)),
            }
        )

    contribution_frame = pd.DataFrame(rows).sort_values(
        "Absolute contribution",
        ascending=False,
    )

    return (
        contribution_frame,
        calibrated_baseline_log_odds,
        reconstructed_log_odds,
    )


try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Failed to load the calibrated deployment model: {exc}")
    st.stop()

required_keys = {
    "pipeline",
    "calibrator",
    "features",
    "calibration_prevalence",
}
missing_keys = required_keys.difference(bundle)
if missing_keys:
    st.error(
        "The model bundle is incomplete. Missing items: "
        f"{sorted(missing_keys)}. Run train_model.py and replace "
        "model/model.joblib."
    )
    st.stop()

pipeline = bundle["pipeline"]
calibrator = bundle["calibrator"]
FEATURES = list(bundle["features"])
SOURCE_PREVALENCE = float(bundle["calibration_prevalence"])
OPERATING_THRESHOLD = bundle.get("operating_threshold")

if "target_prevalence" not in st.session_state:
    st.session_state.target_prevalence = SOURCE_PREVALENCE
if "use_local_adjustment" not in st.session_state:
    st.session_state.use_local_adjustment = False
if "prevalence_label" not in st.session_state:
    st.session_state.prevalence_label = (
        f"Development cohort ({SOURCE_PREVALENCE:.2%})"
    )
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


st.title("CRKP Carbapenem Resistance Risk Assessment Tool")
st.caption(
    "Deployed model: "
    f"{bundle.get('model_name', 'Platt-calibrated ENN–BLSMOTE–XGBoost')}"
)

with st.expander(
    "Prediction-time anchor and intended use",
    expanded=True,
):
    st.markdown(
        """
        **Prediction anchor (T0):** time of culture-specimen collection.

        Enter only information documented at or before T0. Antimicrobial-
        exposure and invasive-device durations must be truncated at T0.
        Information generated after specimen collection must not be entered.

        The calculator is intended for antimicrobial-stewardship reassessment
        after the isolate has been identified as *Klebsiella pneumoniae* and
        before phenotypic antimicrobial-susceptibility testing results are
        available. It does not replace AST, clinical judgment, or prescribing
        review.
        """
    )


# -------------------------------------------------------------------------
# 1. Probability-reference setting
# -------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("1. Select the probability reference")

    setting_left, setting_right = st.columns([2.2, 1])

    with setting_left:
        prevalence_mode = st.radio(
            "Probability reference",
            options=[
                "Use development-cohort prevalence",
                "Apply optional local-prevalence adjustment",
            ],
            index=(
                1
                if st.session_state.use_local_adjustment
                else 0
            ),
            horizontal=True,
        )

        if prevalence_mode == "Apply optional local-prevalence adjustment":
            with st.form("local_prevalence_form"):
                count_left, count_right = st.columns(2)

                with count_left:
                    resistant_count = st.number_input(
                        "Carbapenem-resistant eligible "
                        "K. pneumoniae isolate records",
                        min_value=0,
                        value=0,
                        step=1,
                    )

                with count_right:
                    total_count = st.number_input(
                        "Total eligible K. pneumoniae isolate records",
                        min_value=1,
                        value=100,
                        step=1,
                    )

                apply_prevalence = st.form_submit_button(
                    "Apply local-prevalence adjustment",
                    type="primary",
                )

            if apply_prevalence:
                if resistant_count <= 0:
                    st.error(
                        "The resistant count must be greater than zero. "
                        "Use the development-cohort setting when local "
                        "data are unavailable."
                    )
                elif resistant_count >= total_count:
                    st.error(
                        "The resistant count must be lower than the "
                        "total eligible count."
                    )
                else:
                    local_prevalence = (
                        float(resistant_count) / float(total_count)
                    )
                    st.session_state.target_prevalence = (
                        local_prevalence
                    )
                    st.session_state.use_local_adjustment = True
                    st.session_state.prevalence_label = (
                        f"Local input: {int(resistant_count)}/"
                        f"{int(total_count)} "
                        f"({local_prevalence:.2%})"
                    )
                    st.success(
                        "Local-prevalence adjustment applied. "
                        "This is a prior-prevalence adjustment, "
                        "not full local recalibration."
                    )
        else:
            st.session_state.target_prevalence = SOURCE_PREVALENCE
            st.session_state.use_local_adjustment = False
            st.session_state.prevalence_label = (
                f"Development cohort ({SOURCE_PREVALENCE:.2%})"
            )

    with setting_right:
        st.metric(
            "Current reference prevalence",
            f"{st.session_state.target_prevalence:.2%}",
        )
        st.caption(st.session_state.prevalence_label)

    st.caption(
        "The default output is Platt-calibrated to the natural prevalence "
        f"of the development cohort ({SOURCE_PREVALENCE:.2%}). "
        "Prevalence-only adjustment is optional and does not replace formal "
        "local validation or recalibration."
    )


# -------------------------------------------------------------------------
# 2. Patient data
# -------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("2. Enter patient information available at or before T0")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            days_catheter = st.number_input(
                "Days of indwelling urinary catheterization "
                "accumulated up to T0",
                min_value=0,
                max_value=365,
                value=0,
                help=(
                    "Count from catheter insertion through "
                    "culture-specimen collection only."
                ),
            )
            vascular = st.selectbox(
                "Vascular system disease documented at or before T0",
                ["No", "Yes"],
            )
            respiratory_system = st.selectbox(
                "Respiratory system disease documented at or before T0",
                ["No", "Yes"],
            )
            days_carbapenem = st.number_input(
                "Days of carbapenem exposure during the 90-day "
                "look-back ending at T0",
                min_value=0,
                max_value=90,
                value=0,
                help="Do not include exposure occurring after T0.",
            )
            icu = st.selectbox(
                "ICU admission documented at or before T0",
                ["No", "Yes"],
            )
            metabolic = st.selectbox(
                "Metabolic abnormality documented at or before T0",
                ["No", "Yes"],
            )

        with right:
            respiratory_infection = st.selectbox(
                "Respiratory tract infection documented at or before T0",
                ["No", "Yes"],
            )
            urinary_system = st.selectbox(
                "Urinary system disease documented at or before T0",
                ["No", "Yes"],
            )
            albumin = st.number_input(
                "Most recent albumin available at or before T0 (g/L)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=0.1,
            )
            age = st.number_input(
                "Age at T0 (years)",
                min_value=0,
                max_value=120,
                value=60,
            )
            digestive = st.selectbox(
                "Digestive system disease documented at or before T0",
                ["No", "Yes"],
            )
            days_bli = st.number_input(
                "Days of beta-lactamase inhibitor combination exposure "
                "during the 90-day look-back ending at T0",
                min_value=0,
                max_value=90,
                value=0,
                help="Do not include exposure occurring after T0.",
            )

        anchor_confirmed = st.checkbox(
            "I confirm that all entered information was available "
            "at or before T0."
        )

        submitted = st.form_submit_button(
            "Calculate calibrated resistance probability",
            type="primary",
        )

    if submitted:
        if not anchor_confirmed:
            st.warning(
                "Please confirm that all information was available "
                "at or before the prediction anchor."
            )
        else:
            input_data = {
                "Days of Indwelling Urinary Catheterization":
                    days_catheter,
                "Vascular System Disease":
                    encode_yes_no(vascular),
                "Respiratory System Disease":
                    encode_yes_no(respiratory_system),
                "Days of Carbapenems Use":
                    days_carbapenem,
                "ICU Admission":
                    encode_yes_no(icu),
                "Metabolic Abnormality":
                    encode_yes_no(metabolic),
                "Respiratory Tract Infection":
                    encode_yes_no(respiratory_infection),
                "Urinary System Disease":
                    encode_yes_no(urinary_system),
                "Albumin":
                    albumin,
                "Age":
                    age,
                "Digestive System Disease":
                    encode_yes_no(digestive),
                "Days of β-Lactamase Inhibitor Combinations Use":
                    days_bli,
            }

            try:
                x_input = pd.DataFrame(
                    [input_data],
                    columns=FEATURES,
                )

                raw_probability = float(
                    pipeline.predict_proba(x_input)[:, 1][0]
                )
                calibrated_probability = float(
                    calibrator.predict_proba(
                        probability_to_logit(raw_probability)
                    )[:, 1][0]
                )

                (
                    contribution_frame,
                    baseline_log_odds,
                    reconstructed_log_odds,
                ) = calculate_patient_contributions(
                    pipeline,
                    calibrator,
                    x_input,
                    FEATURES,
                )

                reconstructed_probability = sigmoid(
                    reconstructed_log_odds
                )

                if not np.isclose(
                    reconstructed_probability,
                    calibrated_probability,
                    atol=1e-5,
                ):
                    st.warning(
                        "The contribution decomposition did not exactly "
                        "reconstruct the calibrated probability. "
                        "The probability result remains available, but "
                        "the local explanation has been withheld."
                    )
                    contribution_frame = None

                st.session_state.last_prediction = {
                    "development_probability":
                        calibrated_probability,
                    "contribution_frame":
                        contribution_frame,
                    "baseline_log_odds":
                        baseline_log_odds,
                    "patient_input":
                        input_data,
                }

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# -------------------------------------------------------------------------
# 3. Prediction results
# -------------------------------------------------------------------------
last_prediction = st.session_state.last_prediction

if last_prediction is not None:
    with st.container(border=True):
        st.subheader("3. Prediction result")

        development_probability = float(
            last_prediction["development_probability"]
        )
        adjusted_probability = adjust_for_target_prevalence(
            development_probability,
            SOURCE_PREVALENCE,
            st.session_state.target_prevalence,
        )

        if st.session_state.use_local_adjustment:
            metric_1, metric_2, metric_3 = st.columns(3)
        else:
            metric_1, metric_3 = st.columns(2)
            metric_2 = None

        with metric_1:
            st.metric(
                "Development-cohort calibrated probability",
                f"{development_probability:.1%}",
            )
            st.caption(
                f"Platt-calibrated at prevalence "
                f"{SOURCE_PREVALENCE:.2%}."
            )

        if metric_2 is not None:
            with metric_2:
                st.metric(
                    "Local-prevalence-adjusted estimate",
                    f"{adjusted_probability:.1%}",
                )
                st.caption(
                    f"Adjusted to the entered prevalence "
                    f"{st.session_state.target_prevalence:.2%}; "
                    "not full local recalibration."
                )

        with metric_3:
            if OPERATING_THRESHOLD is not None:
                st.metric(
                    "Development operating threshold",
                    f"{float(OPERATING_THRESHOLD):.3f}",
                )
                st.caption(
                    "Threshold selected from development-cohort "
                    "out-of-fold predictions."
                )
            else:
                st.metric(
                    "Development operating threshold",
                    "Not available",
                )

        if OPERATING_THRESHOLD is not None:
            if development_probability >= float(OPERATING_THRESHOLD):
                st.warning(
                    "The development-calibrated estimate is above the "
                    "prespecified sensitivity-oriented screening threshold. "
                    "Clinical and microbiological review is required."
                )
            else:
                st.info(
                    "The development-calibrated estimate is below the "
                    "prespecified sensitivity-oriented screening threshold. "
                    "This does not exclude carbapenem resistance."
                )

        if st.session_state.use_local_adjustment:
            st.caption(
                "The binary screening interpretation above uses the "
                "development-calibrated probability and development threshold. "
                "No separate local alert threshold is shown because the "
                "prevalence-adjusted estimate has not been locally validated."
            )

        st.caption(
            "The displayed values are predictive estimates rather than "
            "confirmed microbiological diagnoses or causal effects."
        )


    # ---------------------------------------------------------------------
    # 4. Model explanation
    # ---------------------------------------------------------------------
    with st.container(border=True):
        st.subheader("4. Model explanation")

        local_column, global_column = st.columns([1.25, 1])

        with local_column:
            st.markdown("#### Patient-specific feature contributions")
            contribution_frame = last_prediction[
                "contribution_frame"
            ]

            if contribution_frame is not None:
                top_contributions = contribution_frame.head(10).copy()
                chart_frame = top_contributions[
                    ["Feature", "Contribution"]
                ].copy()

                chart_frame["Increases estimated resistance"] = np.where(
                    chart_frame["Contribution"] >= 0,
                    chart_frame["Contribution"],
                    np.nan,
                )
                chart_frame["Decreases estimated resistance"] = np.where(
                    chart_frame["Contribution"] < 0,
                    chart_frame["Contribution"],
                    np.nan,
                )

                chart_frame = chart_frame.drop(
                    columns=["Contribution"]
                ).set_index("Feature")

                st.bar_chart(
                    chart_frame,
                    horizontal=True,
                    height=430,
                )

                st.caption(
                    "Bars show each feature's additive contribution to the "
                    "Platt-calibrated model log-odds. Positive values increase "
                    "the estimated resistance score; negative values decrease "
                    "it. These values are not percentage-point changes and "
                    "should not be interpreted causally."
                )

                table_frame = top_contributions[
                    [
                        "Feature",
                        "Patient value",
                        "Direction",
                        "Contribution",
                    ]
                ].copy()
                table_frame["Contribution"] = table_frame[
                    "Contribution"
                ].round(3)

                with st.expander(
                    "View patient-specific contribution values"
                ):
                    st.dataframe(
                        table_frame,
                        hide_index=True,
                        use_container_width=True,
                    )
            else:
                st.info(
                    "Patient-specific contributions are unavailable for "
                    "this prediction."
                )

        with global_column:
            st.markdown("#### Overall feature importance")
            if IMAGE_PATH.exists():
                st.image(
                    str(IMAGE_PATH),
                    caption=(
                        "Population-level feature importance for the "
                        "12-feature model. This figure does not explain an "
                        "individual patient's prediction."
                    ),
                    use_container_width=True,
                )
            else:
                st.warning(
                    "Global feature-importance image not found at "
                    "assets/SFS12-2.jpg."
                )

        if st.session_state.use_local_adjustment:
            st.caption(
                "Under the optional prevalence-only adjustment, the baseline "
                "risk level changes while the patient-specific feature "
                "contribution bars remain those of the calibrated development "
                "model."
            )


# -------------------------------------------------------------------------
# 5. Model information
# -------------------------------------------------------------------------
with st.expander("Model information and limitations"):
    st.markdown(
        f"""
        **Probability definition.** The primary displayed probability is the
        post-hoc Platt-calibrated probability that the cultured
        *Klebsiella pneumoniae* isolate is carbapenem resistant, conditional
        on information available at or before T0.

        **Calibration reference.** The default probability is referenced to
        the natural prevalence of the 2014–2024 development cohort:
        **{SOURCE_PREVALENCE:.2%}**.

        **Optional local prevalence.** Entering local resistant and total
        eligible isolate counts performs a prior-prevalence adjustment rather
        than full local recalibration. Formal local validation and
        recalibration require individual local predictions and observed AST
        outcomes.

        **Treatment-pathway variables.** Prior antimicrobial exposure is a
        recognized predictor but may also reflect confounding by indication.
        The manuscript therefore reports sensitivity analyses excluding
        treatment-pathway and duration variables.

        **VME and ME.** Very-major-error and major-error rates are
        population-level validation metrics rather than patient-specific
        probabilities. They are reported in the manuscript at the
        prespecified operating point.
        """
    )

st.divider()
st.caption(
    "Research-use disclaimer: This calculator is intended for research "
    "demonstration and antimicrobial-stewardship support. It does not "
    "replace AST, microbiological review, clinical judgment, or diagnosis."
)