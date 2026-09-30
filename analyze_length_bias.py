# ==========================================================
# VeriSight - Length Bias Analysis
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

STAGE1_FILE = (
    BASE_DIR
    / "data"
    / "stage1"
    / "stage1_final_v2.csv"
)

GEMINI_FILE = (
    BASE_DIR
    / "data"
    / "external"
    / "gemini_unseen_balanced.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "length_bias"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Extract Word Counts
# ==========================================================

def get_word_counts(df, group_name):

    print(
        f"\nProcessing {group_name}: "
        f"{len(df):,} reviews"
    )

    records = []

    for review in df["review"].astype(str):

        features = extract_stylometric_features(
            review
        )

        records.append({
            "group": group_name,
            "word_count": features["word_count"],
            "sentence_count": features["sentence_count"],
            "character_count": features["character_count"],
            "avg_sentence_length":
                features["avg_sentence_length"]
        })

    return pd.DataFrame(records)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("VeriSight - Length Bias Analysis")
    print("=" * 70)


    # ======================================================
    # Load Stage 1
    # ======================================================

    print("\nLoading Stage 1 dataset...")

    stage1 = pd.read_csv(
        STAGE1_FILE
    )

    print(
        f"Stage 1 reviews: "
        f"{len(stage1):,}"
    )


    # ======================================================
    # Load Gemini
    # ======================================================

    print("\nLoading Gemini dataset...")

    gemini = pd.read_csv(
        GEMINI_FILE
    )

    print(
        f"Gemini reviews: "
        f"{len(gemini):,}"
    )


    # ======================================================
    # Create Groups
    # ======================================================

    groups = []


    # ------------------------------------------------------
    # Human
    # ------------------------------------------------------

    human = stage1[
        stage1["label"] == 0
    ].copy()

    human = human.sample(
        n=min(320, len(human)),
        random_state=42
    )

    groups.append(
        get_word_counts(
            human,
            "Human"
        )
    )


    # ------------------------------------------------------
    # AI Generators
    # ------------------------------------------------------

    if "generator_model" in stage1.columns:

        generators = (
            stage1[
                stage1["label"] == 1
            ]["generator_model"]
            .dropna()
            .unique()
        )

        for generator in generators:

            generator_df = stage1[
                (stage1["label"] == 1)
                &
                (
                    stage1["generator_model"]
                    == generator
                )
            ].copy()

            if len(generator_df) == 0:
                continue

            generator_df = generator_df.sample(
                n=min(
                    320,
                    len(generator_df)
                ),
                random_state=42
            )

            groups.append(
                get_word_counts(
                    generator_df,
                    str(generator)
                )
            )


    # ------------------------------------------------------
    # Gemini
    # ------------------------------------------------------

    gemini_ai = gemini[
        gemini["label"] == 1
    ].copy()

    gemini_ai = gemini_ai.sample(
        n=min(
            320,
            len(gemini_ai)
        ),
        random_state=42
    )

    groups.append(
        get_word_counts(
            gemini_ai,
            "Gemini"
        )
    )


    # ======================================================
    # Combine
    # ======================================================

    all_data = pd.concat(
        groups,
        ignore_index=True
    )


    # ======================================================
    # Statistics
    # ======================================================

    print("\n" + "=" * 70)
    print("LENGTH STATISTICS")
    print("=" * 70)

    statistics = (
        all_data
        .groupby("group")
        [
            [
                "word_count",
                "sentence_count",
                "character_count",
                "avg_sentence_length"
            ]
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
    )

    print(
        statistics.round(2).to_string()
    )


    # ======================================================
    # Simpler Summary
    # ======================================================

    print("\n" + "=" * 70)
    print("WORD COUNT SUMMARY")
    print("=" * 70)

    word_summary = (
        all_data
        .groupby("group")["word_count"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
        .round(2)
    )

    print(
        word_summary.to_string()
    )


    # ======================================================
    # Length Buckets
    # ======================================================

    print("\n" + "=" * 70)
    print("WORD COUNT BUCKETS")
    print("=" * 70)

    bins = [
        0,
        20,
        40,
        60,
        80,
        100,
        120,
        150,
        200,
        float("inf")
    ]

    labels = [
        "0-20",
        "21-40",
        "41-60",
        "61-80",
        "81-100",
        "101-120",
        "121-150",
        "151-200",
        "200+"
    ]

    all_data["length_bucket"] = pd.cut(
        all_data["word_count"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    bucket_table = pd.crosstab(
        all_data["group"],
        all_data["length_bucket"]
    )

    print(
        bucket_table.to_string()
    )


    # ======================================================
    # Percentage Distribution
    # ======================================================

    print("\n" + "=" * 70)
    print("WORD COUNT BUCKET PERCENTAGES")
    print("=" * 70)

    bucket_percent = (
        bucket_table
        .div(
            bucket_table.sum(axis=1),
            axis=0
        )
        * 100
    )

    print(
        bucket_percent.round(2).to_string()
    )


    # ======================================================
    # Compare Human vs Training AI
    # ======================================================

    print("\n" + "=" * 70)
    print("HUMAN vs TRAINING AI")
    print("=" * 70)

    human_words = all_data[
        all_data["group"] == "Human"
    ]["word_count"]

    training_ai_words = all_data[
        ~all_data["group"].isin(
            ["Human", "Gemini"]
        )
    ]["word_count"]

    print(
        f"\nHuman mean word count: "
        f"{human_words.mean():.2f}"
    )

    print(
        f"Training AI mean word count: "
        f"{training_ai_words.mean():.2f}"
    )

    print(
        f"\nHuman median word count: "
        f"{human_words.median():.2f}"
    )

    print(
        f"Training AI median word count: "
        f"{training_ai_words.median():.2f}"
    )


    # ======================================================
    # Compare Training AI vs Gemini
    # ======================================================

    print("\n" + "=" * 70)
    print("TRAINING AI vs GEMINI")
    print("=" * 70)

    gemini_words = all_data[
        all_data["group"] == "Gemini"
    ]["word_count"]

    print(
        f"\nTraining AI mean word count: "
        f"{training_ai_words.mean():.2f}"
    )

    print(
        f"Gemini mean word count: "
        f"{gemini_words.mean():.2f}"
    )

    print(
        f"\nTraining AI median word count: "
        f"{training_ai_words.median():.2f}"
    )

    print(
        f"Gemini median word count: "
        f"{gemini_words.median():.2f}"
    )


    # ======================================================
    # Save Detailed Data
    # ======================================================

    raw_file = (
        OUTPUT_DIR
        / "length_bias_raw.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "length_bias_summary.csv"
    )

    bucket_file = (
        OUTPUT_DIR
        / "length_bias_buckets.csv"
    )

    all_data.to_csv(
        raw_file,
        index=False
    )

    word_summary.to_csv(
        summary_file
    )

    bucket_percent.to_csv(
        bucket_file
    )


    # ======================================================
    # Final
    # ======================================================

    print("\n" + "=" * 70)
    print("LENGTH BIAS ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nSaved:")

    print(
        raw_file
    )

    print(
        summary_file
    )

    print(
        bucket_file
    )


if __name__ == "__main__":
    main()