import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/phase2/processed_behavior_v3"

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train_behavior.csv"
)

VAL_PATH = os.path.join(
    DATA_DIR,
    "val_behavior.csv"
)

TEST_PATH = os.path.join(
    DATA_DIR,
    "test_behavior.csv"
)

MODEL_DIR = "models/stage2"
RESULT_DIR = "results/stage2"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("VERISIGHT - STAGE 2 LIGHTGBM V3")
print("=" * 75)

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

print(
    f"Train: {train.shape[0]:,} rows, "
    f"{train.shape[1]} columns"
)

print(
    f"Val  : {val.shape[0]:,} rows, "
    f"{val.shape[1]} columns"
)

print(
    f"Test : {test.shape[0]:,} rows, "
    f"{test.shape[1]} columns"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "spam"

y_train = train[TARGET].astype(int)
y_val = val[TARGET].astype(int)
y_test = test[TARGET].astype(int)


# ============================================================
# DROP NON-FEATURE COLUMNS
# ============================================================

DROP_COLS = [
    TARGET,
    "user_id",
    "prod_id",
    "date",

    # This feature is constant zero in V3
    "user_product_previous_reviews"
]


def prepare_features(df):

    X = df.drop(
        columns=DROP_COLS,
        errors="ignore"
    )

    # Remove any accidental non-numeric columns
    non_numeric = X.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric:

        print(
            "Dropping non-numeric columns:",
            non_numeric
        )

        X = X.drop(
            columns=non_numeric
        )

    return X


X_train = prepare_features(train)
X_val = prepare_features(val)
X_test = prepare_features(test)


# ============================================================
# CHECK FEATURE CONSISTENCY
# ============================================================

if list(X_train.columns) != list(X_val.columns):
    raise RuntimeError(
        "Train and validation features do not match."
    )

if list(X_train.columns) != list(X_test.columns):
    raise RuntimeError(
        "Train and test features do not match."
    )


print(
    f"\nFinal model features: {X_train.shape[1]}"
)

print(
    "\nSpam distribution:"
)

print(
    f"Train spam: {y_train.sum():,} "
    f"/ {len(y_train):,}"
)

print(
    f"Val spam  : {y_val.sum():,} "
    f"/ {len(y_val):,}"
)

print(
    f"Test spam : {y_test.sum():,} "
    f"/ {len(y_test):,}"
)


# ============================================================
# MODEL
# ============================================================

print("\nTraining LightGBM...")

model = lgb.LGBMClassifier(

    objective="binary",

    n_estimators=1000,

    learning_rate=0.03,

    num_leaves=63,

    max_depth=-1,

    min_child_samples=30,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=0.1,

    random_state=42,

    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

model.fit(

    X_train,
    y_train,

    eval_set=[
        (X_val, y_val)
    ],

    eval_metric="auc",

    callbacks=[
        lgb.early_stopping(
            100,
            verbose=True
        )
    ]
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

test_prob = model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_pred
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_prob
)

pr_auc = average_precision_score(
    y_test,
    test_prob
)

cm = confusion_matrix(
    y_test,
    test_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 75)
print("V3 TEST PERFORMANCE")
print("=" * 75)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1       : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=[
            "Genuine",
            "Spam"
        ],
        zero_division=0
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 75)
print("TOP 20 FEATURES")
print("=" * 75)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "stage2_lightgbm_v3.txt"
)

model.booster_.save_model(
    model_path
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "version": "stage2_v3",

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1": float(f1),

    "roc_auc": float(roc_auc),

    "pr_auc": float(pr_auc),

    "confusion_matrix": cm.tolist(),

    "train_samples": int(len(train)),

    "validation_samples": int(len(val)),

    "test_samples": int(len(test)),

    "num_features": int(X_train.shape[1]),

    "best_iteration": int(
        model.best_iteration_
    )
}

results_path = os.path.join(
    RESULT_DIR,
    "stage2_lightgbm_v3_results.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


importance_path = os.path.join(
    RESULT_DIR,
    "stage2_lightgbm_v3_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)
print("SAVED")
print("=" * 75)

print(
    model_path
)

print(
    results_path
)

print(
    importance_path
)

print("\n" + "=" * 75)
print("V3 TRAINING COMPLETE")
print("=" * 75)