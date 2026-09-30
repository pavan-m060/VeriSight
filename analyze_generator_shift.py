# ==========================================================
# VeriSight - Generator Shift Analysis
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
    / "generator_shift"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Extract Features
# ==========================================================

def extract_features(df, name):

    print(
        f"\nProcessing {name}: "
        f"{len(df):,} reviews"
    )

    features = []

    for i, review in enumerate(
        df["review"].astype(str)
    ):

        result = extract_stylometric_features(
            review
        )

        features.append(result)

        if (i + 1) % 1000 == 0:

            print(
                f"  {i + 1:,} / "
                f"{len(df):,}"
            )

    feature_df = pd.DataFrame(
        features
    )

    feature_df["group"] = name

    return feature_df


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("VeriSight - Generator Shift Analysis")
    print("=" * 70)


    # ======================================================
    # Load Stage 1
    # ======================================================

    print("\nLoading Stage 1 dataset...")

    stage1 = pd.read_csv(
        STAGE1_FILE
    )

    print(
        f"Stage 1: "
        f"{len(stage1):,} reviews"
    )

    print("\nGenerator distribution:")

    if "generator_model" in stage1.columns:

        print(
            stage1[
                "generator_model"
            ].value_counts()
        )

    else:

        print(
            "generator_model column "
            "not found."
        )


    # ======================================================
    # Load Gemini
    # ======================================================

    print("\nLoading Gemini dataset...")

    gemini = pd.read_csv(
        GEMINI_FILE
    )

    print(
        f"Gemini dataset: "
        f"{len(gemini):,} reviews"
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

    # Use maximum 320 for fair comparison
    human = human.sample(
        n=min(320, len(human)),
        random_state=42
    )

    groups.append(
        extract_features(
            human,
            "Human"
        )
    )


    # ------------------------------------------------------
    # AI Generators
    # ------------------------------------------------------

    if "generator_model" in stage1.columns:

        generator_names = (
            stage1[
                stage1["label"] == 1
            ]["generator_model"]
            .dropna()
            .unique()
        )

        for generator in generator_names:

            generator_df = stage1[
                (stage1["label"] == 1) &
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
                extract_features(
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
        n=min(320, len(gemini_ai)),
        random_state=42
    )

    groups.append(
        extract_features(
            gemini_ai,
            "Gemini"
        )
    )


    # ======================================================
    # Combine
    # ======================================================

    all_features = pd.concat(
        groups,
        ignore_index=True
    )


    # ======================================================
    # Feature Names
    # ======================================================

    feature_columns = [
        column
        for column in all_features.columns
        if column != "group"
    ]


    # ======================================================
    # Calculate Means
    # ======================================================

    mean_table = (
        all_features
        .groupby("group")[feature_columns]
        .mean()
    )


    # ======================================================
    # Calculate Standard Deviation
    # ======================================================

    std_table = (
        all_features
        .groupby("group")[feature_columns]
        .std()
    )


    # ======================================================
    # Print Mean Table
    # ======================================================

    print("\n" + "=" * 70)
    print("MEAN STYLOMETRIC FEATURES")
    print("=" * 70)

    print(
        mean_table.round(4).to_string()
    )


    # ======================================================
    # Gemini vs Training AI
    # ======================================================

    training_ai = all_features[
        all_features["group"] != "Gemini"
    ]

    training_ai = training_ai[
        training_ai["group"] != "Human"
    ]

    training_ai_mean = (
        training_ai[feature_columns]
        .mean()
    )

    gemini_mean = (
        mean_table.loc[
            "Gemini"
        ]
    )


    # ======================================================
    # Difference
    # ======================================================

    comparison = pd.DataFrame({

        "training_ai_mean":
            training_ai_mean,

        "gemini_mean":
            gemini_mean

    })


    comparison[
        "absolute_difference"
    ] = (
        comparison["gemini_mean"]
        -
        comparison["training_ai_mean"]
    ).abs()


    comparison[
        "percentage_difference"
    ] = (
        comparison["absolute_difference"]
        /
        (
            comparison["training_ai_mean"]
            .abs()
            + 1e-8
        )
    ) * 100


    comparison = comparison.sort_values(
        "percentage_difference",
        ascending=False
    )


    # ======================================================
    # Print Comparison
    # ======================================================

    print("\n" + "=" * 70)
    print("TRAINING AI vs GEMINI")
    print("=" * 70)

    print(
        comparison.round(4).to_string()
    )


    # ======================================================
    # Save Results
    # ======================================================

    mean_file = (
        OUTPUT_DIR
        / "generator_means.csv"
    )

    std_file = (
        OUTPUT_DIR
        / "generator_std.csv"
    )

    comparison_file = (
        OUTPUT_DIR
        / "training_ai_vs_gemini.csv"
    )

    mean_table.to_csv(
        mean_file
    )

    std_table.to_csv(
        std_file
    )

    comparison.to_csv(
        comparison_file
    )


    # ======================================================
    # Final
    # ======================================================

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        "\nSaved:"
    )

    print(
        mean_file
    )

    print(
        std_file
    )

    print(
        comparison_file
    )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()