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
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    "models/stage2/stage2_lightgbm_v3.txt"
)

VAL_PATH = (
    "data/phase2/processed_behavior_v3/"
    "val_behavior.csv"
)

TEST_PATH = (
    "data/phase2/processed_behavior_v3/"
    "test_behavior.csv"
)

OUTPUT_DIR = "results/stage2"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 75)
print("VERISIGHT - V3 VALIDATION THRESHOLD SELECTION")
print("=" * 75)

print("\nLoading model...")

model = lgb.Booster(
    model_file=MODEL_PATH
)


# ============================================================
# COMMON FEATURE PREPARATION
# ============================================================

TARGET = "spam"

DROP_COLS = [
    TARGET,
    "user_id",
    "prod_id",
    "date",
    "user_product_previous_reviews"
]


def prepare(df):

    X = df.drop(
        columns=DROP_COLS,
        errors="ignore"
    )

    return X


# ============================================================
# LOAD VALIDATION
# ============================================================

print("\nLoading validation dataset...")

val = pd.read_csv(
    VAL_PATH
)

X_val = prepare(val)

y_val = val[TARGET].astype(int)

print(
    f"Validation samples: {len(val):,}"
)

print(
    f"Validation spam: {y_val.sum():,}"
)


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

print("\nGenerating validation probabilities...")

val_prob = model.predict(
    X_val
)

print(
    f"Probability range: "
    f"{val_prob.min():.4f} - "
    f"{val_prob.max():.4f}"
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

thresholds = np.arange(
    0.05,
    0.61,
    0.01
)

results = []

for threshold in thresholds:

    pred = (
        val_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        pred
    )

    precision = precision_score(
        y_val,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        pred,
        zero_division=0
    )

    results.append({
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_spam": int(pred.sum())
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# BEST F1 THRESHOLD
# ============================================================

best = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(
    best["threshold"]
)


print("\n" + "=" * 75)
print("VALIDATION THRESHOLD RESULTS")
print("=" * 75)

print(
    f"Best threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy       : "
    f"{best['accuracy']:.4f}"
)

print(
    f"Precision      : "
    f"{best['precision']:.4f}"
)

print(
    f"Recall         : "
    f"{best['recall']:.4f}"
)

print(
    f"F1             : "
    f"{best['f1']:.4f}"
)

print(
    f"Predicted spam : "
    f"{int(best['predicted_spam']):,}"
)


# ============================================================
# SHOW NEARBY THRESHOLDS
# ============================================================

print("\nNearby thresholds:")

nearby = results_df[
    (
        results_df["threshold"]
        >= best_threshold - 0.05
    )
    &
    (
        results_df["threshold"]
        <= best_threshold + 0.05
    )
]

print(
    nearby.to_string(
        index=False
    )
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_path = (
    f"{OUTPUT_DIR}/"
    "stage2_v3_selected_threshold.json"
)

with open(
    threshold_path,
    "w"
) as f:

    json.dump(
        {
            "selection_dataset":
                "validation",
            "selection_metric":
                "F1",
            "threshold":
                best_threshold,
            "validation_accuracy":
                float(best["accuracy"]),
            "validation_precision":
                float(best["precision"]),
            "validation_recall":
                float(best["recall"]),
            "validation_f1":
                float(best["f1"])
        },
        f,
        indent=2
    )


csv_path = (
    f"{OUTPUT_DIR}/"
    "stage2_v3_validation_thresholds.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


print("\nSaved:")
print(threshold_path)
print(csv_path)


# ============================================================
# NOW EVALUATE TEST ONCE
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)

print(
    "Using locked validation threshold:"
)

print(
    f"{best_threshold:.2f}"
)


print("\nLoading test dataset...")

test = pd.read_csv(
    TEST_PATH
)

X_test = prepare(test)

y_test = test[TARGET].astype(int)

print(
    f"Test samples: {len(test):,}"
)

print(
    f"Test spam: {y_test.sum():,}"
)


# ============================================================
# TEST PREDICTION
# ============================================================

test_prob = model.predict(
    X_test
)

test_pred = (
    test_prob >= best_threshold
).astype(int)


# ============================================================
# FINAL METRICS
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

cm = confusion_matrix(
    y_test,
    test_pred
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print("\n" + "=" * 75)
print("FINAL STAGE 2 TEST PERFORMANCE")
print("=" * 75)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print(
    f"\nTrue Negatives : {tn:,}"
)

print(
    f"False Positives: {fp:,}"
)

print(
    f"False Negatives: {fn:,}"
)

print(
    f"True Positives : {tp:,}"
)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

final_results = {

    "model":
        "LightGBM V3",

    "threshold":
        best_threshold,

    "threshold_selected_on":
        "validation",

    "selection_metric":
        "F1",

    "test_accuracy":
        float(accuracy),

    "test_precision":
        float(precision),

    "test_recall":
        float(recall),

    "test_f1":
        float(f1),

    "confusion_matrix":
        cm.tolist(),

    "true_negatives":
        int(tn),

    "false_positives":
        int(fp),

    "false_negatives":
        int(fn),

    "true_positives":
        int(tp)
}


final_path = (
    f"{OUTPUT_DIR}/"
    "stage2_v3_final_evaluation.json"
)

with open(
    final_path,
    "w"
) as f:

    json.dump(
        final_results,
        f,
        indent=2
    )


print("\nSaved:")
print(final_path)

print("\n" + "=" * 75)
print("FINAL EVALUATION COMPLETE")
print("=" * 75)