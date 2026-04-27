"""
train_model.py
==============
Run this in Google Colab to generate phishing_model.pkl

Steps:
  1. Upload cleaned_dataset.csv to Colab (same folder as this script)
  2. Run:  !pip install scikit-learn joblib -q
  3. Run:  !python train_model.py
  4. Download the generated phishing_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("cleaned_dataset.csv")

# Drop id column if present
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

X = df.drop("CLASS_LABEL", axis=1)
y = df["CLASS_LABEL"]

FEATURE_NAMES = list(X.columns)   # save for later use in app
print(f"Features: {len(FEATURE_NAMES)}  |  Samples: {len(df)}")

# ── 2. Reproduce exact same split as Person 2's preprocessing ─────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
print(f"Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")

# ── 3. Scale (same as Person 2's scaler) ──────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# ── 4. Train Random Forest ────────────────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_sc, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\n── Validation Results ──────────────────────────────────")
y_pred = model.predict(X_val_sc)
y_prob = model.predict_proba(X_val_sc)[:, 1]
print(classification_report(y_val, y_pred, target_names=["Legitimate", "Phishing"]))
print(f"ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")

print("\n── Test Results ────────────────────────────────────────")
y_pred_test = model.predict(X_test_sc)
y_prob_test = model.predict_proba(X_test_sc)[:, 1]
print(classification_report(y_test, y_pred_test, target_names=["Legitimate", "Phishing"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_test):.4f}")

# ── 6. Save model + scaler + feature names ────────────────────────────────────
bundle = {
    "model":         model,
    "scaler":        scaler,
    "feature_names": FEATURE_NAMES,
}
joblib.dump(bundle, "phishing_model.pkl")
print("\n✅ phishing_model.pkl saved successfully!")
print("   Download it and place it in the  model/  folder of the app.")
