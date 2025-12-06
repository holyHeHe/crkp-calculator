import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="CRKP Resistance Prediction Tool", page_icon="🧫", layout="wide")

# 页面说明
st.markdown("""
### 🧫 CRKP Carbapenem Resistance Risk Assessment Tool

**说明：**
1. 本工具用于评估患者感染的肺炎克雷伯菌是否具有碳青霉烯耐药性。
2. 通过输入 12 项临床指标，模型将给出耐药概率。
3. 本工具仅用于科研演示，不能替代临床诊断。
""")

# 模型加载
model_path = os.path.join("model", "model.joblib")
try:
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    FEATURES = bundle["features"]
except Exception as e:
    st.error("❌ 模型加载失败，请检查 model/model.joblib 是否存在且格式正确。")
    st.stop()

# 左侧输入栏
with st.sidebar:
    st.header("📝 请填写以下 12 项指标：")
    days_catheter = st.number_input("导尿管留置天数", min_value=0, max_value=365, value=0)
    vascular = st.selectbox("血管系统疾病", ["否", "是"])
    resp_sys = st.selectbox("呼吸系统疾病", ["否", "是"])
    days_carbapenem = st.number_input("碳青霉烯使用天数", min_value=0, max_value=365, value=0)
    icu = st.selectbox("是否入住 ICU", ["否", "是"])
    metabolic = st.selectbox("代谢异常", ["否", "是"])
    resp_inf = st.selectbox("呼吸道感染", ["否", "是"])
    urinary = st.selectbox("泌尿系统疾病", ["否", "是"])
    albumin = st.number_input("白蛋白 (g/L)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
    age = st.number_input("年龄 (岁)", min_value=0, max_value=120, value=60)
    digestive = st.selectbox("消化系统疾病", ["否", "是"])
    days_beta = st.number_input("β-内酰胺酶抑制剂使用天数", min_value=0, max_value=365, value=0)

def bin_code(x): return 1 if x == "是" else 0

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

# 中间展示特征图
image_path = os.path.join("assets", "SFS12-2.jpg")
if os.path.exists(image_path):
    st.image(image_path, caption="SFS 选出的 12 项特征重要性图")
else:
    st.warning("⚠️ 特征图未找到，请确认 assets/SFS12-2.jpg 是否已上传。")

# 右侧预测结果
st.subheader("📊 预测结果")
if st.button("点击计算耐药概率"):
    try:
        proba = pipe.predict_proba(X_input)[:, 1][0]
        color = "red" if proba > 0.5 else "green"
        st.markdown(
            f"<div style='color:{color}; font-size:24px; font-weight:bold;'>预测耐药概率：{proba*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error("❌ 预测失败，请检查输入格式或模型兼容性。")

st.caption("模型基于 ENN‑BLSMOTE‑XGBoost 与 SFS 特征选择，仅用于科研演示。")