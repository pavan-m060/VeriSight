# ==========================================================
# VeriSight
# compare_stylometry_gemini_v3.py
#
# Compare:
#   Human training reviews
#   AI training reviews
#   Unseen Gemini reviews
#
# Uses the same 24 stylometric features.
# ==========================================================

import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_FILE = (
    PROJECT_ROOT /
    "features" /
    "stylometry_stage1_v3.npy"
)

Y_FILE = (
    PROJECT_ROOT /
    "features" /
    "y_stylometry_stage1_v3.npy"
)

GEMINI_FILE = (
    PROJECT_ROOT /
    "features" /
    "stylometry_gemini_v3.npy"
)


# ==========================================================
# FEATURE NAMES
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
# LOAD
# ==========================================================

print("=" * 80)
print("VERISIGHT - STYLOMETRIC DISTRIBUTION ANALYSIS")
print("=" * 80)

X = np.load(X_FILE)
y = np.load(Y_FILE)
G = np.load(GEMINI_FILE)

print("\nTraining feature shape:", X.shape)
print("Training label shape  :", y.shape)
print("Gemini feature shape  :", G.shape)


# ==========================================================
# SPLIT TRAINING DATA
# ==========================================================

human = X[y == 0]
ai = X[y == 1]

print("\nHuman samples:", len(human))
print("AI samples   :", len(ai))
print("Gemini       :", len(G))


# ==========================================================
# BUILD COMPARISON TABLE
# ==========================================================

rows = []

for i, feature in enumerate(FEATURE_NAMES):

    human_mean = human[:, i].mean()
    ai_mean = ai[:, i].mean()
    gemini_mean = G[:, i].mean()

    human_median = np.median(human[:, i])
    ai_median = np.median(ai[:, i])
    gemini_median = np.median(G[:, i])

    rows.append({
        "feature": feature,

        "human_mean": human_mean,
        "ai_mean": ai_mean,
        "gemini_mean": gemini_mean,

        "human_median": human_median,
        "ai_median": ai_median,
        "gemini_median": gemini_median
    })


result = pd.DataFrame(rows)


# ==========================================================
# PRINT FULL TABLE
# ==========================================================

pd.set_option(
    "display.max_rows",
    30
)

pd.set_option(
    "display.max_columns",
    10
)

pd.set_option(
    "display.width",
    220
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.6f}"
)

print("\n" + "=" * 80)
print("FEATURE MEANS AND MEDIANS")
print("=" * 80)

print(
    result.to_string(index=False)
)


# ==========================================================
# DISTANCE FROM HUMAN / AI MEANS
# ==========================================================

# Training standard deviation
training_std = X.std(
    axis=0
)

# Avoid division by zero
training_std[
    training_std == 0
] = 1.0

human_mean = human.mean(axis=0)
ai_mean = ai.mean(axis=0)
gemini_mean = G.mean(axis=0)


human_distance = (
    np.abs(gemini_mean - human_mean)
    / training_std
)

ai_distance = (
    np.abs(gemini_mean - ai_mean)
    / training_std
)

distance_table = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "Gemini_vs_Human_z": human_distance,
    "Gemini_vs_AI_z": ai_distance
})

distance_table[
    "closer_to"
] = np.where(
    human_distance < ai_distance,
    "Human",
    "AI"
)

distance_table[
    "distance_difference"
] = (
    ai_distance -
    human_distance
)


# ==========================================================
# SORT BY STRONGEST DIFFERENCE
# ==========================================================

distance_table = (
    distance_table
    .sort_values(
        "distance_difference",
        key=lambda x: np.abs(x),
        ascending=False
    )
)


print("\n" + "=" * 80)
print("FEATURES WHERE GEMINI IS CLOSER TO HUMAN OR AI")
print("=" * 80)

print(
    distance_table.to_string(
        index=False
    )
)


# ==========================================================
# CENTROID DISTANCE
# ==========================================================

# Standardize everything using training distribution

X_standardized = (
    X / training_std
)

human_centroid = (
    human_mean / training_std
)

ai_centroid = (
    ai_mean / training_std
)

gemini_standardized = (
    gemini_mean / training_std
)


human_centroid_distance = np.linalg.norm(
    gemini_standardized -
    human_centroid
)

ai_centroid_distance = np.linalg.norm(
    gemini_standardized -
    ai_centroid
)


print("\n" + "=" * 80)
print("OVERALL CENTROID DISTANCE")
print("=" * 80)

print(
    f"Gemini -> Human centroid: "
    f"{human_centroid_distance:.6f}"
)

print(
    f"Gemini -> AI centroid    : "
    f"{ai_centroid_distance:.6f}"
)


if human_centroid_distance < ai_centroid_distance:

    print(
        "\nGemini stylometric centroid "
        "is closer to HUMAN."
    )

else:

    print(
        "\nGemini stylometric centroid "
        "is closer to AI."
    )


# ==========================================================
# UPPERCASE SPECIFIC CHECK
# ==========================================================

uppercase_index = 16

human_upper = human[:, uppercase_index]
ai_upper = ai[:, uppercase_index]
gemini_upper = G[:, uppercase_index]

print("\n" + "=" * 80)
print("UPPERCASE RATIO")
print("=" * 80)

print(
    f"Human mean   : "
    f"{human_upper.mean():.6f}"
)

print(
    f"AI mean      : "
    f"{ai_upper.mean():.6f}"
)

print(
    f"Gemini mean  : "
    f"{gemini_upper.mean():.6f}"
)

print(
    f"Human median : "
    f"{np.median(human_upper):.6f}"
)

print(
    f"AI median    : "
    f"{np.median(ai_upper):.6f}"
)

print(
    f"Gemini median: "
    f"{np.median(gemini_upper):.6f}"
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

OUTPUT_FILE = (
    PROJECT_ROOT /
    "features" /
    "stylometry_gemini_distribution_analysis.csv"
)

distance_table.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nAnalysis saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)