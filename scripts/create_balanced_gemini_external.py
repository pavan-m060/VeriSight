import pandas as pd
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "external"
    / "gemini_external_final.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "external"
    / "gemini_external_balanced.csv"
)

# ==========================================================
# Settings
# ==========================================================

SOURCES = [
    "Amazon",
    "GooglePlay",
    "TripAdvisor",
    "Yelp"
]

RATINGS = [1, 2, 3, 4, 5]

SAMPLES_PER_SOURCE = 90
SAMPLES_PER_RATING = 72

# ==========================================================
# Load
# ==========================================================

print("=" * 65)
print("Creating Balanced Gemini External Dataset")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal Shape:")
print(df.shape)

# ==========================================================
# Select balanced data
# ==========================================================

selected = []

# We want 80 of each rating overall.
# At the same time, 100 reviews per source.
#
# First select 20 reviews for every
# source + rating combination.
#
# 4 sources × 5 ratings × 20 = 400

SAMPLES_PER_SOURCE_RATING = 16

for source in SOURCES:

    for rating in RATINGS:

        subset = df[
            (df["source"] == source)
            & (df["rating"] == rating)
        ]

        available = len(subset)

        print(
            f"{source:12s} | "
            f"{rating}★ | "
            f"Available: {available:3d} | "
            f"Taking: {SAMPLES_PER_SOURCE_RATING}"
        )

        if available < SAMPLES_PER_SOURCE_RATING:

            raise ValueError(
                f"Not enough reviews for "
                f"{source} {rating}★. "
                f"Available: {available}"
            )

        sample = subset.sample(
            n=SAMPLES_PER_SOURCE_RATING,
            random_state=42
        )

        selected.append(sample)

# ==========================================================
# Combine
# ==========================================================

balanced = pd.concat(
    selected,
    ignore_index=True
)

# ==========================================================
# Shuffle
# ==========================================================

balanced = balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ==========================================================
# Validation
# ==========================================================

print("\n" + "=" * 65)
print("BALANCED DATASET")
print("=" * 65)

print("\nShape:")
print(balanced.shape)

print("\nSource Distribution:")
print(balanced["source"].value_counts())

print("\nRating Distribution:")
print(
    balanced["rating"]
    .value_counts()
    .sort_index()
)

print("\nSource + Rating Distribution:")
print(
    balanced
    .groupby(["source", "rating"])
    .size()
    .unstack(fill_value=0)
)

print("\nDuplicate Reviews:")
print(
    balanced["review"].duplicated().sum()
)

print("\nGenerator Distribution:")
print(
    balanced["generator_model"].value_counts()
)

print("\nLabel Distribution:")
print(
    balanced["label"].value_counts()
)

# ==========================================================
# Save
# ==========================================================

balanced.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 65)
print("Balanced Gemini Dataset Created")
print("=" * 65)

print("\nFinal Shape:")
print(balanced.shape)

print("\nSaved to:")
print(OUTPUT_FILE)