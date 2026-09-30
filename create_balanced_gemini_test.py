# ==========================================================
# VeriSight - Create Balanced Gemini External Test Set
# ==========================================================
#
# Creates:
#
#     320 Human reviews
#     320 Gemini AI reviews
#     --------------------
#     640 total reviews
#
# This dataset is used ONLY for external evaluation.
#
# ==========================================================

import pandas as pd
from pathlib import Path


# ==========================================================
# Project Root
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent


# ==========================================================
# Input Files
# ==========================================================

GEMINI_FILE = (
    BASE_DIR
    / "data"
    / "external"
    / "gemini_external_balanced.csv"
)

HUMAN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "human_reviews.csv"
)


# ==========================================================
# Output
# ==========================================================

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "external"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "gemini_unseen_balanced.csv"
)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("VeriSight - Balanced Gemini External Test Set")
    print("=" * 70)


    # ======================================================
    # STEP 1 - Load Gemini
    # ======================================================

    print("\nLoading Gemini dataset...")

    if not GEMINI_FILE.exists():

        raise FileNotFoundError(
            f"\nGemini dataset not found:\n{GEMINI_FILE}"
        )

    gemini = pd.read_csv(
        GEMINI_FILE
    )

    print(
        f"Gemini reviews available: "
        f"{len(gemini):,}"
    )


    # ======================================================
    # STEP 2 - Load Human
    # ======================================================

    print("\nLoading Human dataset...")

    if not HUMAN_FILE.exists():

        raise FileNotFoundError(
            f"\nHuman dataset not found:\n{HUMAN_FILE}"
        )

    human = pd.read_csv(
        HUMAN_FILE
    )

    print(
        f"Human reviews available: "
        f"{len(human):,}"
    )


    # ======================================================
    # STEP 3 - Validate Gemini
    # ======================================================

    if "review" not in gemini.columns:

        raise ValueError(
            "Gemini dataset must contain "
            "'review' column."
        )


    # ======================================================
    # STEP 4 - Validate Human
    # ======================================================

    if "review" not in human.columns:

        raise ValueError(
            "Human dataset must contain "
            "'review' column."
        )


    # ======================================================
    # STEP 5 - Select 320 Gemini Reviews
    # ======================================================

    gemini_test = (
        gemini
        .sample(
            n=320,
            random_state=42
        )
        .copy()
    )


    # ======================================================
    # STEP 6 - Select 320 Human Reviews
    # ======================================================

    human_test = (
        human
        .sample(
            n=320,
            random_state=42
        )
        .copy()
    )


    # ======================================================
    # STEP 7 - Prepare Gemini
    # ======================================================

    gemini_test = pd.DataFrame({

        "review":
            gemini_test["review"].values,

        "label":
            1,

        "type":
            "AI",

        "generator_model":
            "Gemini",

        "source":
            gemini_test["source"].values
            if "source" in gemini_test.columns
            else "Gemini",

        "rating":
            gemini_test["rating"].values
            if "rating" in gemini_test.columns
            else None
    })


    # ======================================================
    # STEP 8 - Prepare Human
    # ======================================================

    human_test = pd.DataFrame({

        "review":
            human_test["review"].values,

        "label":
            0,

        "type":
            "Human",

        "generator_model":
            "Human",

        "source":
            human_test["source"].values
            if "source" in human_test.columns
            else "Human",

        "rating":
            human_test["rating"].values
            if "rating" in human_test.columns
            else None
    })


    # ======================================================
    # STEP 9 - Combine
    # ======================================================

    final_df = pd.concat(
        [
            human_test,
            gemini_test
        ],
        ignore_index=True
    )


    # ======================================================
    # STEP 10 - Clean
    # ======================================================

    final_df["review"] = (
        final_df["review"]
        .astype(str)
        .str.strip()
    )

    final_df = final_df[
        final_df["review"] != ""
    ].copy()


    # ======================================================
    # STEP 11 - Remove Duplicates
    # ======================================================

    final_df = (
        final_df
        .drop_duplicates(
            subset=["review"]
        )
        .reset_index(drop=True)
    )


    # ======================================================
    # STEP 12 - Shuffle
    # ======================================================

    final_df = (
        final_df
        .sample(
            frac=1,
            random_state=42
        )
        .reset_index(drop=True)
    )


    # ======================================================
    # STEP 13 - Final Validation
    # ======================================================

    print("\n" + "=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print(
        "\nShape:",
        final_df.shape
    )

    print(
        "\nLabel Distribution:"
    )

    print(
        final_df["label"]
        .value_counts()
    )

    print(
        "\nGenerator Distribution:"
    )

    print(
        final_df["generator_model"]
        .value_counts()
    )

    print(
        "\nType Distribution:"
    )

    print(
        final_df["type"]
        .value_counts()
    )


    # ======================================================
    # STEP 14 - Verify Balance
    # ======================================================

    human_count = (
        final_df["label"] == 0
    ).sum()

    ai_count = (
        final_df["label"] == 1
    ).sum()


    print(
        f"\nHuman reviews: {human_count}"
    )

    print(
        f"Gemini AI reviews: {ai_count}"
    )


    if human_count != ai_count:

        print(
            "\nWARNING: Dataset is not perfectly balanced."
        )

    else:

        print(
            "\nDataset is perfectly balanced."
        )


    # ======================================================
    # STEP 15 - Save
    # ======================================================

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )


    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )


    # ======================================================
    # Final
    # ======================================================

    print("\n" + "=" * 70)
    print("BALANCED GEMINI TEST SET CREATED")
    print("=" * 70)

    print(
        "\nFinal samples:",
        len(final_df)
    )

    print(
        "Human:",
        human_count
    )

    print(
        "Gemini:",
        ai_count
    )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()