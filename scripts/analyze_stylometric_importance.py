# ==========================================================
# VeriSight
# Stage 1 - Stylometric Feature Importance Analysis
# ==========================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# ==========================================================
# Paths
# ==========================================================

X_FILE = "features/stylometry_stage1_v2.npy"
Y_FILE = "features/y_stylometry_stage1_v2.npy"

OUTPUT_FILE = "features/stylometric_feature_importance.csv"

# ==========================================================
# Feature Names
# ==========================================================

FEATURE_NAMES = [
    "word_count",
    "sentence_count",
    "character_count",
    "avg_word_length",
    "avg_sentence_length",
    "sentence_length_std",
    "vocabulary_richness",
    "repeated_word_ratio",
    "punctuation_count",
    "punctuation_ratio",
    "comma_count",
    "period_count",
    "question_count",
    "exclamation_count",
    "digit_count",
    "digit_ratio",
    "uppercase_ratio",
    "stopword_ratio",
    "long_word_ratio",
    "short_word_ratio",
    "first_person_ratio",
    "second_person_ratio",
    "third_person_ratio",
    "avg_chars_per_word"
]

# ==========================================================
# Start
# ==========================================================

print("=" * 70)
print("VeriSight - Stylometric Feature Importance Analysis")
print("=" * 70)

# ==========================================================
# Load
# ==========================================================

print("\nLoading stylometric features...")

X = np.load(X_FILE)
y = np.load(Y_FILE)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================================
# Split
# ==========================================================

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# ==========================================================
# Random Forest
# ==========================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(
    X_train,
    y_train
)

# ==========================================================
# Accuracy
# ==========================================================

predictions = rf.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nRandom Forest Accuracy: "
    f"{accuracy * 100:.2f}%"
)

# ==========================================================
# Gini Feature Importance
# ==========================================================

print("\nCalculating Random Forest feature importance...")

gini_importance = rf.feature_importances_

# ==========================================================
# Permutation Importance
# ==========================================================

print("\nCalculating permutation importance...")
print("This may take some time...")

perm = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_mean = perm.importances_mean

permutation_std = perm.importances_std

# ==========================================================
# Create Results DataFrame
# ==========================================================

results = pd.DataFrame({

    "feature": FEATURE_NAMES,

    "gini_importance":
        gini_importance,

    "permutation_importance":
        permutation_mean,

    "permutation_std":
        permutation_std
})

# ==========================================================
# Sort by Permutation Importance
# ==========================================================

results = results.sort_values(
    by="permutation_importance",
    ascending=False
)

# ==========================================================
# Save
# ==========================================================

os.makedirs(
    "features",
    exist_ok=True
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==========================================================
# Print Results
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)

# ==========================================================
# Top Features
# ==========================================================

print("\n" + "=" * 70)
print("TOP 10 STYLOMETRIC FEATURES")
print("=" * 70)

top10 = results.head(10)

for rank, (_, row) in enumerate(
    top10.iterrows(),
    start=1
):

    print(
        f"{rank:2d}. "
        f"{row['feature']:28s} "
        f"Permutation = "
        f"{row['permutation_importance']:.6f}"
    )

# ==========================================================
# Negative Importance Warning
# ==========================================================

negative = results[
    results["permutation_importance"] < 0
]

if len(negative) > 0:

    print("\n" + "=" * 70)
    print("FEATURES WITH NEGATIVE PERMUTATION IMPORTANCE")
    print("=" * 70)

    for feature in negative["feature"]:
        print("-", feature)

# ==========================================================
# Final
# ==========================================================

print("\n" + "=" * 70)
print("Analysis completed")
print("=" * 70)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 70)