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
# CONFIGURATION
# ============================================================

DATA_DIR = "data/phase2/processed_behavior_v2"

MODEL_DIR = "models/stage2"
RESULT_DIR = "results/stage2"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("VERISIGHT - STAGE 2 LIGHTGBM")
print("=" * 70)

print("\nLoading datasets...")

train = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "train_behavior.csv"
    )
)

val = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "val_behavior.csv"
    )
)

test = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "test_behavior.csv"
    )
)

print(
    f"Train: {train.shape}"
)

print(
    f"Validation: {val.shape}"
)

print(
    f"Test: {test.shape}"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "spam"

# Metadata columns that should NOT be used as ML features
DROP_COLUMNS = [
    TARGET,
    "user_id",
    "prod_id",
    "date"
]


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    c for c in train.columns
    if c not in DROP_COLUMNS
]

print(
    f"\nNumber of ML features: "
    f"{len(feature_columns)}"
)

print("\nFeatures:")

for feature in feature_columns:
    print(
        f"  - {feature}"
    )


X_train = train[feature_columns]
y_train = train[TARGET]

X_val = val[feature_columns]
y_val = val[TARGET]

X_test = test[feature_columns]
y_test = test[TARGET]


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print("\nTrain:")
print(
    y_train.value_counts()
    .sort_index()
)

print("\nValidation:")
print(
    y_val.value_counts()
    .sort_index()
)

print("\nTest:")
print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# LIGHTGBM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)

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

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbosity=-1
)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_val, y_val)
    ],

    eval_metric="auc",

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=50,
            verbose=True
        )
    ]
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "stage2_lightgbm.txt"
)

model.booster_.save_model(
    model_path
)

print(
    f"\nModel saved to:\n{model_path}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    split_name
):

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 70)

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
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
            y,
            predictions,
            target_names=[
                "Genuine",
                "Spam"
            ],
            digits=4,
            zero_division=0
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# EVALUATE
# ============================================================

val_results = evaluate_model(
    model,
    X_val,
    y_val,
    "Validation"
)

test_results = evaluate_model(
    model,
    X_test,
    y_test,
    "Test"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(
        index=False
    )
)

importance_file = os.path.join(
    RESULT_DIR,
    "stage2_feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "LightGBM",
    "features": feature_columns,
    "validation": val_results,
    "test": test_results
}

results_file = os.path.join(
    RESULT_DIR,
    "stage2_lightgbm_results.json"
)

with open(
    results_file,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 LIGHTGBM COMPLETE")
print("=" * 70)

print("\nTEST PERFORMANCE")

print(
    f"Accuracy : "
    f"{test_results['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{test_results['precision']:.4f}"
)

print(
    f"Recall   : "
    f"{test_results['recall']:.4f}"
)

print(
    f"F1       : "
    f"{test_results['f1']:.4f}"
)

print(
    f"ROC-AUC  : "
    f"{test_results['roc_auc']:.4f}"
)

print(
    f"PR-AUC   : "
    f"{test_results['pr_auc']:.4f}"
)

print("\nSaved:")

print(model_path)
print(results_file)
print(importance_file)

print("\n" + "=" * 70)