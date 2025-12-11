# train_model.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import BorderlineSMOTE

# 1) Load dataset
df = pd.read_excel("feike3.xlsx", sheet_name="Sheet2")
df.columns = [str(c).strip() for c in df.columns]

# Target column (adjust index if needed)
y = df.iloc[:, 65]
mask = y.notna() & np.isfinite(y)
y = y[mask].astype(int)
X = df.iloc[:, 2:65][mask].copy()

# Selected features
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
    "Days of β-Lactamase Inhibitor Combinations Use"
]
X = X[FEATURES]

# 2) Define preprocessing (safe to serialize)
num_cols = list(X.select_dtypes(include=[np.number]).columns)
cat_cols = [c for c in X.columns if c not in num_cols]

numeric_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer([
    ("num", numeric_tf, num_cols),
    ("cat", categorical_tf, cat_cols),
])

# 3) Fit preprocessing and transform
X_proc = preprocess.fit_transform(X)

# 4) Apply resampling (do not serialize resamplers)
enn = EditedNearestNeighbours(n_neighbors=3)
X_enn, y_enn = enn.fit_resample(X_proc, y)

smote = BorderlineSMOTE(random_state=2025)
X_bal, y_bal = smote.fit_resample(X_enn, y_enn)

# 5) Train classifier
clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2025,
    n_jobs=-1,
    eval_metric="logloss",
    tree_method="hist"
)
clf.fit(X_bal, y_bal)

# 6) Save only stable artifacts (preprocessor + classifier + features)
os.makedirs("model", exist_ok=True)
joblib.dump(
    {"preprocess": preprocess, "clf": clf, "features": FEATURES},
    "model/model.joblib"
)

print("✅ Model retrained and saved to model/model.joblib")
