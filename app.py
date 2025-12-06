# app.py
import joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="CRKP Resistance Prediction Tool", page_icon="🧫", layout="wide")

# Note box
st.markdown("""
**Note:**
1. **This website aims to develop a tool that uses machine learning algorithms to assess whether Klebsiella pneumoniae in patients is carbapenem-resistant.**

2. **The risk of carbapenem resistance can be calculated by inputting 12 indicators, including:**
Days of Indwelling Urinary Catheterization, Vascular System Disease, Respiratory System Disease, Days of Carbapenems Use, ICU Admission, Metabolic Abnormality, Respiratory Tract Infection, Urinary System Disease, Albumin, Age, Digestive System Disease, Days of β-Lactamase Inhibitor Combinations Use.

3. **This tool is only for the preliminary assessment of carbapenem resistance in Klebsiella pneumoniae in patients and shall not replace clinical diagnosis.**
""")

st.title("CRKP Carbapenem Resistance Risk Assessment Tool")

# Load model
bundle = joblib.load("model/model.joblib")
pipe = bundle["pipeline"]
FEATURES = bundle["features"]

# Sidebar input fields
with st.sidebar:
    st.markdown("### Please fill in the following 12 indicators:")
    days_catheter = st.number_input("Days of Indwelling Urinary Catheterization", min_value=0, max_value=365, value=0)
    vascular = st.selectbox("Vascular System Disease", ["No", "Yes"])
    resp_sys = st.selectbox("Respiratory System Disease", ["No", "Yes"])
    days_carbapenem = st.number_input("Days of Carbapenems Use", min_value=0, max_value=365, value=0)
    icu = st.selectbox("ICU Admission", ["No", "Yes"])
    metabolic = st.selectbox("Metabolic Abnormality", ["No", "Yes"])
    resp_inf = st.selectbox("Respiratory Tract Infection", ["No", "Yes"])
    urinary = st.selectbox("Urinary System Disease", ["No", "Yes"])
    albumin = st.number_input("Albumin (g/L)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=60)
    digestive = st.selectbox("Digestive System Disease", ["No", "Yes"])
    days_beta = st.number_input("Days of β-Lactamase Inhibitor Combinations Use", min_value=0, max_value=365, value=0)

def bin_code(x): return 1 if x == "Yes" else 0

input_dict = {
    "Days of Indwelling Urinary Catheterization": days_catheter,
    "Vascular System Disease": bin_code(vascular),
    "Respiratory System Disease": bin_code(resp_sys),
    "Days of Carbapenems Use": days_carbapenem,
    "ICU Admission": bin_code(icu),
    "Metabolic Abnormality": bin_code(metabolic),
    "Respiratory Tract Infection": bin_code(resp_inf),
    "Urinary System Disease": bin_code(urinary),
    "Albumin": albumin,
    "Age": age,
    "Digestive System Disease": bin_code(digestive),
    "Days of β-Lactamase Inhibitor Combinations Use": days_beta,
}
X_input = pd.DataFrame([input_dict], columns=FEATURES)

# Center: feature importance image
st.image("assets/SFS12-2.jpg", caption="Feature importance identified by SFS (12 features)")

# Result box
if st.button("Calculate Resistance Probability"):
    proba = pipe.predict_proba(X_input)[:, 1][0]
    if proba > 0.5:
        st.markdown(f"<div style='color:red; font-size:24px; font-weight:bold;'>Predicted Probability: {proba*100:.2f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:green; font-size:24px; font-weight:bold;'>Predicted Probability: {proba*100:.2f}%</div>", unsafe_allow_html=True)

st.caption("Disclaimer: This tool is based on ENN‑BLSMOTE‑XGBoost and 12 features selected by SFS. It is intended for research demonstration only and should not be used as a substitute for clinical diagnosis.")
