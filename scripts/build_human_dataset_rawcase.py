# ==========================================================
# VeriSight
# build_human_dataset_rawcase.py
#
# Build Human Dataset while PRESERVING original capitalization
# ==========================================================

import os
import re
import json
import random
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

# ==========================================================
# RAW FILES
# ==========================================================

AMAZON_FILE = RAW_DATA / "amazon.csv"
TRIP_FILE = RAW_DATA / "tripadvisor.csv"
GOOGLEPLAY_FILE = RAW_DATA / "googleplay.csv"
YELP_FILE = RAW_DATA / "yelp_academic_dataset_review.json"

# ==========================================================
# OUTPUT
# ==========================================================

OUTPUT_FILE = PROCESSED_DATA / "human_reviews_rawcase.csv"

# ==========================================================
# REQUIRED SAMPLE COUNTS
# ==========================================================

SAMPLES = {
    "Amazon": 5000,
    "Yelp": 7000,
    "TripAdvisor": 4000,
    "GooglePlay": 4500
}

# ==========================================================
# POSSIBLE COLUMN NAMES
# ==========================================================

REVIEW_COLUMNS = [
    "review",
    "reviewText",
    "text",
    "content",
    "body",
    "comment"
]

RATING_COLUMNS = [
    "rating",
    "score",
    "stars"
]

CATEGORY_COLUMNS = [
    "category",
    "product_category",
    "app",
    "business_category"
]

# ==========================================================
# FIND COLUMN
# ==========================================================

def find_column(df, candidates):

    for col in candidates:

        if col in df.columns:
            return col

    return None


# ==========================================================
# CLEAN TEXT
#
# IMPORTANT:
# NO LOWERCASE HERE
# ==========================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize Unicode
    text = text.strip()

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text


# ==========================================================
# VALIDATE REVIEW
# ==========================================================

def validate_review(text):

    if not text:
        return False

    words = text.split()

    # Keep the same general filtering idea
    if len(words) < 8:
        return False

    if len(words) > 250:
        return False

    if len(text) < 60:
        return False

    if len(text) > 2500:
        return False

    return True


# ==========================================================
# PROCESS DATAFRAME
# ==========================================================

def process_dataframe(df, source):

    print("\n" + "=" * 60)
    print(f"Processing {source}")
    print("=" * 60)

    review_col = find_column(df, REVIEW_COLUMNS)

    if review_col is None:

        raise ValueError(
            f"{source}: Could not find review column."
        )

    rating_col = find_column(df, RATING_COLUMNS)
    category_col = find_column(df, CATEGORY_COLUMNS)

    # Copy original dataframe
    df = df.copy()

    # IMPORTANT:
    # Preserve original capitalization
    df["review"] = (
        df[review_col]
        .apply(clean_text)
    )

    # Rating
    if rating_col:

        df["rating"] = df[rating_col]

    else:

        df["rating"] = np.nan

    # Category
    if category_col:

        df["category"] = (
            df[category_col]
            .astype(str)
        )

    else:

        df["category"] = source

    df["source"] = source

    # Remove empty
    df = df[
        df["review"].str.strip() != ""
    ].copy()

    # Validate
    df = df[
        df["review"].apply(validate_review)
    ].copy()

    # Remove duplicate reviews
    df = df.drop_duplicates(
        subset=["review"]
    )

    print("Available after cleaning:", len(df))

    return df.reset_index(drop=True)


# ==========================================================
# LOAD CSV DATASET
# ==========================================================

def load_csv_dataset(file_path, source):

    print(f"\nLoading: {file_path}")

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    print("Raw shape:", df.shape)

    return process_dataframe(
        df,
        source
    )


# ==========================================================
# LOAD YELP JSON
# ==========================================================

def load_yelp():

    print("\n" + "=" * 60)
    print("Loading Yelp")
    print("=" * 60)

    reviews = []

    SAMPLE_LIMIT = 50000

    with open(
        YELP_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for i, line in enumerate(f):

            try:

                review = json.loads(line)

            except Exception:

                continue

            if len(reviews) < SAMPLE_LIMIT:

                reviews.append(review)

            else:

                j = random.randint(0, i)

                if j < SAMPLE_LIMIT:

                    reviews[j] = review

    df = pd.DataFrame(reviews)

    print("Yelp raw shape:", df.shape)

    return process_dataframe(
        df,
        "Yelp"
    )


# ==========================================================
# SAMPLE DATA
# ==========================================================

def sample_dataset(df, source):

    required = SAMPLES[source]

    if len(df) < required:

        raise ValueError(
            f"{source} has only {len(df)} usable reviews, "
            f"but {required} are required."
        )

    return df.sample(
        n=required,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("VERISIGHT - HUMAN DATASET RAW CASE VERSION")
    print("=" * 70)

    # ------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------

    amazon = load_csv_dataset(
        AMAZON_FILE,
        "Amazon"
    )

    tripadvisor = load_csv_dataset(
        TRIP_FILE,
        "TripAdvisor"
    )

    googleplay = load_csv_dataset(
        GOOGLEPLAY_FILE,
        "GooglePlay"
    )

    yelp = load_yelp()

    # ------------------------------------------------------
    # Sample
    # ------------------------------------------------------

    amazon = sample_dataset(
        amazon,
        "Amazon"
    )

    yelp = sample_dataset(
        yelp,
        "Yelp"
    )

    tripadvisor = sample_dataset(
        tripadvisor,
        "TripAdvisor"
    )

    googleplay = sample_dataset(
        googleplay,
        "GooglePlay"
    )

    # ------------------------------------------------------
    # Merge
    # ------------------------------------------------------

    human = pd.concat(
        [
            amazon,
            yelp,
            tripadvisor,
            googleplay
        ],
        ignore_index=True
    )

    # ------------------------------------------------------
    # Keep required columns
    # ------------------------------------------------------

    human = human[
        [
            "review",
            "rating",
            "source",
            "category"
        ]
    ].copy()

    # ------------------------------------------------------
    # Shuffle
    # ------------------------------------------------------

    human = human.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    human.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ------------------------------------------------------
    # Results
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("RAW-CASE HUMAN DATASET CREATED")
    print("=" * 70)

    print("\nShape:")
    print(human.shape)

    print("\nSource Distribution:")
    print(
        human["source"].value_counts()
    )

    print("\nFirst 10 Reviews:")
    print(
        human["review"]
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nIMPORTANT:")
    print("Original capitalization has been preserved.")
    print("No .lower() operation was applied.")


if __name__ == "__main__":
    main()