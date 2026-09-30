import os
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "external",
    "length_balanced",
    "gemini_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "external",
    "length_balanced"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "gemini_missed_reviews.csv"
)


def main():

    print("=" * 70)
    print("VeriSight - Gemini Failure Analysis")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nTotal reviews: {len(df)}"
    )

    # ------------------------------------------------------
    # Gemini reviews
    # ------------------------------------------------------

    gemini = df[
        df["label"] == 1
    ].copy()

    print(
        f"Gemini reviews: {len(gemini)}"
    )

    # ------------------------------------------------------
    # Missed Gemini
    # prediction = 0 means Human
    # ------------------------------------------------------

    missed = gemini[
        gemini["prediction"] == 0
    ].copy()

    detected = gemini[
        gemini["prediction"] == 1
    ].copy()

    print(
        f"\nGemini correctly detected: "
        f"{len(detected)}"
    )

    print(
        f"Gemini missed: "
        f"{len(missed)}"
    )

    # ------------------------------------------------------
    # Word count
    # ------------------------------------------------------

    missed["word_count"] = (
        missed["review"]
        .astype(str)
        .str.split()
        .str.len()
    )

    detected["word_count"] = (
        detected["review"]
        .astype(str)
        .str.split()
        .str.len()
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("WORD COUNT COMPARISON")
    print("=" * 70)

    print(
        "\nMissed Gemini:"
    )

    print(
        missed["word_count"]
        .describe()
        .round(2)
    )

    print(
        "\nDetected Gemini:"
    )

    print(
        detected["word_count"]
        .describe()
        .round(2)
    )

    # ------------------------------------------------------
    # Probability comparison
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROBABILITY COMPARISON")
    print("=" * 70)

    print(
        "\nMissed Gemini AI probability:"
    )

    print(
        missed["ai_probability"]
        .describe()
        .round(4)
    )

    print(
        "\nDetected Gemini AI probability:"
    )

    print(
        detected["ai_probability"]
        .describe()
        .round(4)
    )

    # ------------------------------------------------------
    # Length buckets
    # ------------------------------------------------------

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
        np.inf
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

    missed["length_bucket"] = pd.cut(
        missed["word_count"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    detected["length_bucket"] = pd.cut(
        detected["word_count"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    print("\n" + "=" * 70)
    print("MISSED GEMINI BY LENGTH")
    print("=" * 70)

    print(
        missed["length_bucket"]
        .value_counts()
        .sort_index()
    )

    print("\n" + "=" * 70)
    print("DETECTED GEMINI BY LENGTH")
    print("=" * 70)

    print(
        detected["length_bucket"]
        .value_counts()
        .sort_index()
    )

    # ------------------------------------------------------
    # Show hardest examples
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("HARDEST GEMINI EXAMPLES")
    print("=" * 70)

    hardest = missed.sort_values(
        "ai_probability",
        ascending=False
    )

    for index, row in hardest.head(20).iterrows():

        print("\n--------------------------------------------------")

        print(
            f"AI Probability: "
            f"{row['ai_probability']:.6f}"
        )

        print(
            f"Word Count: "
            f"{row['word_count']}"
        )

        print(
            "Review:"
        )

        print(
            row["review"]
        )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    missed.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        "\nSaved missed Gemini reviews:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()