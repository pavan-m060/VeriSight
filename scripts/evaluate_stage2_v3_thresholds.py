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
# LOAD
# ============================================================

print("=" * 75)
print("VERISIGHT - STAGE 2 V3 THRESHOLD ANALYSIS")
print("=" * 75)

model = lgb.Booster(
    model_file=MODEL_PATH
)

df = pd.read_csv(
    TEST_PATH
)

print(
    f"\nTest samples: {len(df):,}"
)


# ============================================================
# PREPARE FEATURES
# ============================================================

TARGET = "spam"

DROP_COLS = [
    TARGET,
    "user_id",
    "prod_id",
    "date",
    "user_product_previous_reviews"
]

X = df.drop(
    columns=DROP_COLS,
    errors="ignore"
)

y = df[TARGET].astype(int)

print(
    f"Features: {X.shape[1]}"
)

print(
    f"Spam: {(y == 1).sum():,}"
)

print(
    f"Genuine: {(y == 0).sum():,}"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating probabilities...")

spam_prob = model.predict(X)

print(
    f"Probability range: "
    f"{spam_prob.min():.4f} - "
    f"{spam_prob.max():.4f}"
)


# ============================================================
# THRESHOLDS
# ============================================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]


results = []


print("\n" + "=" * 75)
print("THRESHOLD ANALYSIS")
print("=" * 75)

print(
    f"{'Threshold':>10}"
    f"{'Accuracy':>11}"
    f"{'Precision':>12}"
    f"{'Recall':>11}"
    f"{'F1':>11}"
    f"{'Pred Spam':>12}"
)

print("-" * 75)


for threshold in thresholds:

    pred = (
        spam_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y,
        pred
    )

    precision = precision_score(
        y,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        pred,
        zero_division=0
    )

    predicted_spam = pred.sum()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_spam": predicted_spam
    })

    print(
        f"{threshold:10.2f}"
        f"{accuracy:11.4f}"
        f"{precision:12.4f}"
        f"{recall:11.4f}"
        f"{f1:11.4f}"
        f"{predicted_spam:12,}"
    )


# ============================================================
# BEST F1
# ============================================================

results_df = pd.DataFrame(
    results
)

best = results_df.loc[
    results_df["f1"].idxmax()
]

threshold = float(
    best["threshold"]
)

pred = (
    spam_prob >= threshold
).astype(int)

cm = confusion_matrix(
    y,
    pred
)

tn, fp, fn, tp = cm.ravel()


print("\n" + "=" * 75)
print("BEST THRESHOLD BY F1")
print("=" * 75)

print(
    f"Threshold : {threshold:.2f}"
)

print(
    f"Accuracy  : {best['accuracy']:.4f}"
)

print(
    f"Precision : {best['precision']:.4f}"
)

print(
    f"Recall    : {best['recall']:.4f}"
)

print(
    f"F1        : {best['f1']:.4f}"
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
# SAVE
# ============================================================

csv_path = (
    f"{OUTPUT_DIR}/"
    "stage2_v3_threshold_analysis.csv"
)

json_path = (
    f"{OUTPUT_DIR}/"
    "stage2_v3_threshold_summary.json"
)

results_df.to_csv(
    csv_path,
    index=False
)

summary = {
    "best_threshold": threshold,
    "accuracy": float(
        best["accuracy"]
    ),
    "precision": float(
        best["precision"]
    ),
    "recall": float(
        best["recall"]
    ),
    "f1": float(
        best["f1"]
    ),
    "confusion_matrix": cm.tolist()
}

with open(
    json_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print("\nSaved:")
print(csv_path)
print(json_path)

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)