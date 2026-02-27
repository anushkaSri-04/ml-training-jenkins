import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# =========================
# LOAD DATA
# =========================

dataset = pd.read_csv("data/clean_dataset.csv")


# =========================
# FEATURE ENGINEERING
# =========================

dataset["term_months"] = dataset["term"].str.extract(r'(\d+)')[0].astype(int)

dataset["income_log"] = np.log1p(dataset["annual_inc"])
dataset["loan_log"] = np.log1p(dataset["loan_amnt"])

dataset["loan_to_income"] = dataset["loan_amnt"] / (dataset["annual_inc"] + 1)
dataset["income_per_term"] = dataset["annual_inc"] / dataset["term_months"]

dataset["emp_length_num"] = (
    dataset["emp_length"]
        .str.replace("+", "", regex=False)
        .str.extract(r'(\d+)')[0]
        .astype(float)
        .fillna(0)
)

dataset["income_per_experience"] = dataset["annual_inc"] / (dataset["emp_length_num"] + 1)

dataset["loan_dti_interaction"] = dataset["loan_amnt"] * dataset["dti"]
dataset["installment_term_interaction"] = dataset["installment"] * dataset["term_months"]
dataset["income_installment_ratio"] = dataset["income_per_term"] / (dataset["installment"] + 1)

dataset = dataset.drop(columns=["term"])

dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.fillna(dataset.median(numeric_only=True))


# =========================
# TARGET + FEATURES
# =========================

y = dataset["eligible"]
X = dataset.drop("eligible", axis=1)

X = pd.get_dummies(X, drop_first=True)
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print("Final Shape:", X.shape)
print("Feature Engineering Completed Successfully.")


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# LOGISTIC REGRESSION
# =========================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)
log_probs = log_model.predict_proba(X_test_scaled)[:, 1]

print("\nLOGISTIC REGRESSION RESULTS")
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))
print("ROC-AUC:", roc_auc_score(y_test, log_probs))


# =========================
# XGBOOST
# =========================

scale_pos_weight = 0.8 * ((y_train == 0).sum() / (y_train == 1).sum())

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.4,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.1,
    reg_lambda=2,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]


# Threshold Optimization

best_thresh = 0
best_recall_good = 0

for t in np.arange(0.1, 0.9, 0.01):
    preds = (xgb_probs >= t).astype(int)
    recall_good = recall_score(y_test, preds, pos_label=1)
    recall_bad = recall_score(y_test, preds, pos_label=0)

    if recall_bad >= 0.20 and recall_good > best_recall_good:
        best_recall_good = recall_good
        best_thresh = t

print("Optimized Threshold:", best_thresh)

xgb_custom_pred = (xgb_probs >= best_thresh).astype(int)

print("\nXGBOOST RESULTS")
print(confusion_matrix(y_test, xgb_custom_pred))
print(classification_report(y_test, xgb_custom_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_probs))

approval_rate = (xgb_custom_pred == 1).mean() * 100
print("Approval Rate (%):", approval_rate)


# =========================
# LIGHTGBM
# =========================

lgb_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    min_child_samples=50,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
)

lgb_model.fit(X_train, y_train)

lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
lgb_pred = (lgb_probs >= 0.5).astype(int)

print("\nLIGHTGBM RESULTS")
print(confusion_matrix(y_test, lgb_pred))
print(classification_report(y_test, lgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, lgb_probs))


# =========================
# CATBOOST
# =========================

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    auto_class_weights="Balanced"
)

cat_model.fit(X_train, y_train)

cat_probs = cat_model.predict_proba(X_test)[:, 1]
cat_pred = (cat_probs >= 0.5).astype(int)

print("\nCATBOOST RESULTS")
print(confusion_matrix(y_test, cat_pred))
print(classification_report(y_test, cat_pred))
print("ROC-AUC:", roc_auc_score(y_test, cat_probs))


# =========================
# MODEL COMPARISON
# =========================

results = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost", "LightGBM", "CatBoost"],
    "Precision": [
        precision_score(y_test, log_pred),
        precision_score(y_test, xgb_custom_pred),
        precision_score(y_test, lgb_pred),
        precision_score(y_test, cat_pred)
    ],
    "Recall": [
        recall_score(y_test, log_pred),
        recall_score(y_test, xgb_custom_pred),
        recall_score(y_test, lgb_pred),
        recall_score(y_test, cat_pred)
    ],
    "F1 Score": [
        f1_score(y_test, log_pred),
        f1_score(y_test, xgb_custom_pred),
        f1_score(y_test, lgb_pred),
        f1_score(y_test, cat_pred)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, log_probs),
        roc_auc_score(y_test, xgb_probs),
        roc_auc_score(y_test, lgb_probs),
        roc_auc_score(y_test, cat_probs)
    ]
})

print("\nMODEL COMPARISON")
print(results)


# =========================
# SAVE BEST MODEL
# =========================

os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/xgboost_model.pkl")

print("\nModel saved successfully in models/xgboost_model.pkl")