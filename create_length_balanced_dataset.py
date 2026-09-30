# ==========================================================
# VeriSight - Create Length Balanced Training Dataset
# ==========================================================

import pandas as pd
import numpy as np
from pathlib import Path

from features.stylometric_features import (
    extract_stylometric_features
)


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "stage1"
    / "stage1_final_v2.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "stage1"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "stage1_length_balanced.csv"
)


# ==========================================================
# Configuration
# ==========================================================

RANDOM_STATE = 42

# Number of samples per class per bucket.
#
# We deliberately give more representation to the
# short-review range because unseen Gemini is concentrated
# there.

SAMPLES_PER_BUCKET = 1000


# ==========================================================
# Length Buckets
# ==========================================================

BUCKETS = [
    (0, 20, "0-20"),
    (21, 40, "21-40"),
    (41, 60, "41-60"),
    (61, 80, "61-80"),
    (81, 100, "81-100"),
    (101, 120, "101-120"),
    (121, 150, "121-150"),
    (151, 200, "151-200"),
    (201, 100000, "200+")
]


# ==========================================================
# Calculate Word Counts
# ==========================================================

def calculate_word_counts(df):

    print("\nCalculating word counts...")

    word_counts = []

    total = len(df)

    for i, review in enumerate(
        df["review"].astype(str)
    ):

        features = extract_stylometric_features(
            review
        )

        word_counts.append(
            features["word_count"]
        )

        if (i + 1) % 5000 == 0:

            print(
                f"Processed "
                f"{i + 1:,} / {total:,}"
            )

    df = df.copy()

    df["word_count"] = word_counts

    return df


# ==========================================================
# Assign Length Bucket
# ==========================================================

def assign_bucket(word_count):

    for minimum, maximum, name in BUCKETS:

        if minimum <= word_count <= maximum:

            return name

    return "200+"


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("VeriSight - Length Balanced Dataset Creation")
    print("=" * 70)


    # ======================================================
    # Load
    # ======================================================

    print("\nLoading Stage 1 dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Original dataset: "
        f"{len(df):,}"
    )

    print("\nOriginal class distribution:")

    print(
        df["label"].value_counts()
    )


    # ======================================================
    # Word Count
    # ======================================================

    df = calculate_word_counts(
        df
    )


    # ======================================================
    # Length Bucket
    # ======================================================

    df["length_bucket"] = (
        df["word_count"]
        .apply(assign_bucket)
    )


    # ======================================================
    # Show Original Distribution
    # ======================================================

    print("\n" + "=" * 70)
    print("ORIGINAL LENGTH DISTRIBUTION")
    print("=" * 70)

    original_distribution = pd.crosstab(
        df["label"],
        df["length_bucket"]
    )

    print(
        original_distribution.to_string()
    )


    # ======================================================
    # Balance Each Bucket
    # ======================================================

    print("\n" + "=" * 70)
    print("CREATING BALANCED DATASET")
    print("=" * 70)

    balanced_parts = []


    for _, _, bucket_name in BUCKETS:

        print(
            f"\nBucket: {bucket_name}"
        )

        bucket_df = df[
            df["length_bucket"]
            == bucket_name
        ]

        # --------------------------------------------------
        # Human
        # --------------------------------------------------

        human = bucket_df[
            bucket_df["label"] == 0
        ]

        # --------------------------------------------------
        # AI
        # --------------------------------------------------

        ai = bucket_df[
            bucket_df["label"] == 1
        ]

        print(
            f"Human available: "
            f"{len(human):,}"
        )

        print(
            f"AI available: "
            f"{len(ai):,}"
        )

        # --------------------------------------------------
        # Determine sample count
        # --------------------------------------------------

        available = min(
            len(human),
            len(ai),
            SAMPLES_PER_BUCKET
        )

        if available == 0:

            print(
                "SKIPPED - no samples for "
                "both classes."
            )

            continue

        # --------------------------------------------------
        # Sample Human
        # --------------------------------------------------

        human_sample = human.sample(
            n=available,
            random_state=RANDOM_STATE
        )

        # --------------------------------------------------
        # Sample AI
        # --------------------------------------------------

        ai_sample = ai.sample(
            n=available,
            random_state=RANDOM_STATE
        )

        balanced_parts.append(
            human_sample
        )

        balanced_parts.append(
            ai_sample
        )

        print(
            f"Selected: "
            f"{available:,} Human + "
            f"{available:,} AI"
        )


    # ======================================================
    # Combine
    # ======================================================

    balanced_df = pd.concat(
        balanced_parts,
        ignore_index=True
    )


    # ======================================================
    # Shuffle
    # ======================================================

    balanced_df = (
        balanced_df
        .sample(
            frac=1,
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )


    # ======================================================
    # Final Distribution
    # ======================================================

    print("\n" + "=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print(
        f"\nTotal samples: "
        f"{len(balanced_df):,}"
    )

    print("\nClass distribution:")

    print(
        balanced_df["label"]
        .value_counts()
    )

    print("\nLength distribution:")

    final_distribution = pd.crosstab(
        balanced_df["label"],
        balanced_df["length_bucket"]
    )

    print(
        final_distribution.to_string()
    )


    # ======================================================
    # Percentage Distribution
    # ======================================================

    print("\n" + "=" * 70)
    print("FINAL LENGTH PERCENTAGES")
    print("=" * 70)

    percentage_distribution = (
        final_distribution
        .div(
            final_distribution.sum(axis=1),
            axis=0
        )
        * 100
    )

    print(
        percentage_distribution
        .round(2)
        .to_string()
    )


    # ======================================================
    # Generator Distribution
    # ======================================================

    if "generator_model" in balanced_df.columns:

        print("\n" + "=" * 70)
        print("AI GENERATOR DISTRIBUTION")
        print("=" * 70)

        ai_generator_distribution = (
            balanced_df[
                balanced_df["label"] == 1
            ]["generator_model"]
            .value_counts()
        )

        print(
            ai_generator_distribution
        )


    # ======================================================
    # Save
    # ======================================================

    # We don't need these analysis columns for the actual
    # feature generation later.

    balanced_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )


    print("\n" + "=" * 70)
    print("LENGTH BALANCED DATASET CREATED")
    print("=" * 70)

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()