# ==========================================================
# VeriSight
# build_human_dataset.py
#
# Block 1
# Imports + Configuration + Logger
# ==========================================================

import os
import re
import json
import random
import logging
import warnings
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "data" / "reports"

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ==========================================================
# INPUT FILES
# ==========================================================

AMAZON_FILE = RAW_DATA / "amazon.csv"

TRIP_FILE = RAW_DATA / "tripadvisor.csv"

GOOGLEPLAY_FILE = RAW_DATA / "googleplay.csv"

YELP_FILE = RAW_DATA / "yelp_academic_dataset_review.json"

# ==========================================================
# OUTPUT FILES
# ==========================================================

OUTPUT_DATASET = PROCESSED_DATA / "human_reviews.csv"

STATISTICS_FILE = REPORTS / "dataset_statistics.csv"

RATING_FILE = REPORTS / "rating_distribution.csv"

CATEGORY_FILE = REPORTS / "category_distribution.csv"

SOURCE_FILE = REPORTS / "source_distribution.csv"

QUALITY_FILE = REPORTS / "quality_report.csv"

CLEANING_FILE = REPORTS / "cleaning_report.csv"

# ==========================================================
# SAMPLE SIZE
# ==========================================================

SAMPLES = {

    "Amazon":5000,

    "Yelp":7000,

    "TripAdvisor":4000,

    "GooglePlay":4500

}

# ==========================================================
# REVIEW FILTERS
# ==========================================================

MIN_WORDS = 8

MAX_WORDS = 250

MIN_CHARACTERS = 60

MAX_CHARACTERS = 2500

QUALITY_THRESHOLD = 45

# ==========================================================
# LOGGER
# ==========================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger("VeriSight")

# ==========================================================
# CLEANING REPORT
# ==========================================================

cleaning_report = []

def add_cleaning_report(

    dataset,

    loaded,

    empty,

    duplicate,

    short,

    long_reviews,

    final

):

    cleaning_report.append({

        "Dataset":dataset,

        "Loaded":loaded,

        "Removed Empty":empty,

        "Removed Duplicate":duplicate,

        "Removed Short":short,

        "Removed Long":long_reviews,

        "Final":final

    })

# ==========================================================
# COLUMN DETECTION
# ==========================================================
REVIEW_COLUMNS = [

    "review",

    "text",

    "content",

    "reviewText",

    "body",

    "comment",

    "summary"

]

RATING_COLUMNS = [

    "rating",

    "score",

    "stars"

]

CATEGORY_COLUMNS = [

    "category",

    "product_category",

    "type"

]

# ==========================================================
# DUPLICATE HASHES
# ==========================================================

GLOBAL_HASHES = set()

# ==========================================================
# END OF BLOCK 1
#
# Continue with BLOCK 2
# ==========================================================

# ==========================================================
# BLOCK 2
# Advanced Text Cleaning
# ==========================================================

# ----------------------------------------------------------
# Regular Expressions
# ----------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

HTML_PATTERN = re.compile(r"<.*?>")

EMAIL_PATTERN = re.compile(r"\S+@\S+")

MULTISPACE_PATTERN = re.compile(r"\s+")

PUNCT_PATTERN = re.compile(r"([!?.,])\1+")

REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]+")

# ----------------------------------------------------------
# HTML Removal
# ----------------------------------------------------------

def remove_html(text):

    return re.sub(HTML_PATTERN, " ", text)

# ----------------------------------------------------------
# URL Removal
# ----------------------------------------------------------

def remove_urls(text):

    return re.sub(URL_PATTERN, " ", text)

# ----------------------------------------------------------
# Email Removal
# ----------------------------------------------------------

def remove_emails(text):

    return re.sub(EMAIL_PATTERN, " ", text)

# ----------------------------------------------------------
# Emoji / Unicode Removal
# ----------------------------------------------------------

def remove_unicode(text):

    text = unicodedata.normalize("NFKD", text)

    text = text.encode("ascii", "ignore").decode("utf-8", "ignore")

    return text

# ----------------------------------------------------------
# Repeated Punctuation
# ----------------------------------------------------------

def normalize_punctuation(text):

    return re.sub(PUNCT_PATTERN, r"\1", text)

# ----------------------------------------------------------
# Repeated Characters
# Example:
# goooooood -> good
# niiiiiice -> nice
# ----------------------------------------------------------

def normalize_repeated_characters(text):

    return re.sub(REPEAT_CHAR_PATTERN, r"\1", text)

# ----------------------------------------------------------
# Remove Control Characters
# ----------------------------------------------------------

def remove_control_characters(text):

    return "".join(

        ch

        for ch in text

        if unicodedata.category(ch)[0] != "C"

    )

# ----------------------------------------------------------
# Lowercase
# ----------------------------------------------------------

def lowercase(text):

    return text.lower()

# ----------------------------------------------------------
# Remove Extra Spaces
# ----------------------------------------------------------

def normalize_spaces(text):

    text = re.sub(MULTISPACE_PATTERN, " ", text)

    return text.strip()

# ----------------------------------------------------------
# English Language Check
# ----------------------------------------------------------

def is_english(text, threshold=0.90):

    if not text:

        return False

    ascii_chars = sum(c.isascii() for c in text)

    ratio = ascii_chars / max(len(text), 1)

    return ratio >= threshold

# ----------------------------------------------------------
# Word Count
# ----------------------------------------------------------

def word_count(text):

    return len(text.split())

# ----------------------------------------------------------
# Character Count
# ----------------------------------------------------------

def character_count(text):

    return len(text)

# ----------------------------------------------------------
# Duplicate Normalization
# ----------------------------------------------------------

def normalize_duplicate(text):

    text = lowercase(text)

    text = re.sub(r"[^\w\s]", "", text)

    text = normalize_spaces(text)

    return text

# ----------------------------------------------------------
# Global Duplicate Check
# ----------------------------------------------------------

def is_duplicate(text):

    normalized = normalize_duplicate(text)

    if normalized in GLOBAL_HASHES:

        return True

    GLOBAL_HASHES.add(normalized)

    return False

# ----------------------------------------------------------
# Complete Cleaning Pipeline
# ----------------------------------------------------------

def clean_text(text):

    if pd.isna(text):

        return ""

    text = str(text)

    text = remove_html(text)

    text = remove_urls(text)

    text = remove_emails(text)

    text = remove_unicode(text)

    text = remove_control_characters(text)

    text = normalize_punctuation(text)

    text = normalize_repeated_characters(text)

    text = lowercase(text)

    text = normalize_spaces(text)

    return text

# ----------------------------------------------------------
# Review Validation
# ----------------------------------------------------------

def validate_review(text):

    if text == "":

        return False

    if not is_english(text):

        return False

    wc = word_count(text)

    cc = character_count(text)

    if wc < MIN_WORDS:

        return False

    if wc > MAX_WORDS:

        return False

    if cc < MIN_CHARACTERS:

        return False

    if cc > MAX_CHARACTERS:

        return False

    return True

# ==========================================================
# END OF BLOCK 2
#
# Continue with BLOCK 3
# ==========================================================

# ==========================================================
# BLOCK 3
# Quality Score + Sampling + Statistics
# ==========================================================

# ----------------------------------------------------------
# Vocabulary Diversity
# ----------------------------------------------------------

def lexical_diversity(text):

    words = text.split()

    if len(words) == 0:
        return 0

    return len(set(words)) / len(words)

# ----------------------------------------------------------
# Average Word Length
# ----------------------------------------------------------

def average_word_length(text):

    words = text.split()

    if len(words) == 0:
        return 0

    return np.mean([len(w) for w in words])

# ----------------------------------------------------------
# Sentence Count
# ----------------------------------------------------------

def sentence_count(text):

    sentences = re.split(r"[.!?]+", text)

    sentences = [s for s in sentences if s.strip()]

    return max(len(sentences), 1)

# ----------------------------------------------------------
# Readability Score
# ----------------------------------------------------------

def readability_score(text):

    words = word_count(text)

    sentences = sentence_count(text)

    avg = words / sentences

    score = 100 - abs(avg - 18) * 2

    return max(min(score, 100), 0)

# ----------------------------------------------------------
# Quality Score
# ----------------------------------------------------------

def quality_score(text):

    score = 0

    wc = word_count(text)

    diversity = lexical_diversity(text)

    readability = readability_score(text)

    avg_len = average_word_length(text)

    # Word count (40)

    if 25 <= wc <= 150:
        score += 40
    elif wc >= 15:
        score += 25

    # Vocabulary diversity (30)

    score += diversity * 30

    # Readability (20)

    score += readability * 0.20

    # Average word length (10)

    if 4 <= avg_len <= 7:
        score += 10
    elif 3 <= avg_len <= 8:
        score += 5

    return round(min(score, 100), 2)

# ----------------------------------------------------------
# Dataset Statistics
# ----------------------------------------------------------

def dataset_statistics(df):

    stats = {

        "Total Reviews": len(df),

        "Average Words":
            round(df["word_count"].mean(),2),

        "Average Characters":
            round(df["char_count"].mean(),2),

        "Average Quality":
            round(df["quality_score"].mean(),2),

        "Minimum Words":
            df["word_count"].min(),

        "Maximum Words":
            df["word_count"].max(),

        "Vocabulary Mean":
            round(df["lexical_diversity"].mean(),3)

    }

    return pd.DataFrame(

        list(stats.items()),

        columns=["Metric","Value"]

    )

# ----------------------------------------------------------
# Balanced Rating Sampling
# ----------------------------------------------------------

def balanced_sample(df, sample_size):

    if "rating" not in df.columns:

        return df.sample(

            min(sample_size, len(df)),

            random_state=RANDOM_STATE

        )

    groups = []

    ratings = sorted(df["rating"].dropna().unique())

    if len(ratings) == 0:

        return df.sample(

            min(sample_size, len(df)),

            random_state=RANDOM_STATE

        )

    per_rating = sample_size // len(ratings)

    for rating in ratings:

        subset = df[df["rating"] == rating]

        n = min(len(subset), per_rating)

        groups.append(

            subset.sample(

                n=n,

                random_state=RANDOM_STATE

            )

        )

    sampled = pd.concat(groups)

    remaining = sample_size - len(sampled)

    if remaining > 0:

        extra = df.drop(sampled.index)

        if len(extra):

            sampled = pd.concat([

                sampled,

                extra.sample(

                    min(remaining, len(extra)),

                    random_state=RANDOM_STATE

                )

            ])

    return sampled.sample(

        frac=1,

        random_state=RANDOM_STATE

    ).reset_index(drop=True)

# ----------------------------------------------------------
# Final Feature Engineering
# ----------------------------------------------------------

def add_features(df):

    df["word_count"] = df["review"].apply(word_count)

    df["char_count"] = df["review"].apply(character_count)

    df["quality_score"] = df["review"].apply(quality_score)

    df["lexical_diversity"] = df["review"].apply(lexical_diversity)

    return df

# ----------------------------------------------------------
# Remove Low Quality Reviews
# ----------------------------------------------------------

def filter_quality(df):

    before = len(df)

    df = df[df["quality_score"] >= QUALITY_THRESHOLD]

    logger.info(

        f"Removed {before-len(df)} low-quality reviews."

    )

    return df.reset_index(drop=True)

# ==========================================================
# END OF BLOCK 3
#
# Continue with BLOCK 4
# ==========================================================

# ==========================================================
# BLOCK 4
# Amazon + TripAdvisor Loaders
# ==========================================================

# ----------------------------------------------------------
# Find Column Name
# ----------------------------------------------------------

def find_column(df, candidates):

    for col in candidates:
        if col in df.columns:
            return col

    return None

# ----------------------------------------------------------
# Generic Dataset Processor
# ----------------------------------------------------------

def process_dataframe(df, source):

    logger.info(f"\nProcessing {source} dataset...")

    loaded = len(df)

    review_col = find_column(df, REVIEW_COLUMNS)
    rating_col = find_column(df, RATING_COLUMNS)
    category_col = find_column(df, CATEGORY_COLUMNS)

    if review_col is None:
        raise ValueError(
            f"{source}: Could not find review column."
        )

    df = df.copy()

    df["review"] = df[review_col].astype(str).apply(clean_text)

    if rating_col:
        df["rating"] = df[rating_col]
    else:
        df["rating"] = np.nan

    if category_col:
        df["category"] = df[category_col].astype(str)
    else:
        df["category"] = source

    df["source"] = source

    # Remove empty reviews

    before = len(df)

    df = df[df["review"].str.strip() != ""]

    empty_removed = before - len(df)

    # Validation

    before = len(df)

    df = df[df["review"].apply(validate_review)]

    removed = before - len(df)

    # Feature Engineering

    df = add_features(df)

    # Quality Filtering

    before = len(df)

    if source != "GooglePlay":
        df = filter_quality(df)

    quality_removed = before - len(df)

    add_cleaning_report(

        dataset=source,

        loaded=loaded,

        empty=empty_removed,

        duplicate=0,

        short=removed,

        long_reviews=quality_removed,

        final=len(df)

    )

    logger.info(f"{source}: {len(df)} reviews ready.")

    return df.reset_index(drop=True)

# ----------------------------------------------------------
# Amazon Loader
# ----------------------------------------------------------

def load_amazon():

    logger.info("=" * 60)
    logger.info("Loading Amazon Dataset")
    logger.info("=" * 60)

    if not AMAZON_FILE.exists():

        raise FileNotFoundError(

            f"Amazon dataset not found:\n{AMAZON_FILE}"

        )

    extension = AMAZON_FILE.suffix.lower()

    if extension == ".csv":

        df = pd.read_csv(

            AMAZON_FILE,

            low_memory=False

        )

    elif extension == ".json":

        df = pd.read_json(

            AMAZON_FILE,

            lines=True

        )

    else:

        raise ValueError(

            "Unsupported Amazon file format."

        )

    return process_dataframe(

        df,

        "Amazon"

    )

# ----------------------------------------------------------
# TripAdvisor Loader
# ----------------------------------------------------------

def load_tripadvisor():

    logger.info("=" * 60)
    logger.info("Loading TripAdvisor Dataset")
    logger.info("=" * 60)

    if not TRIP_FILE.exists():

        raise FileNotFoundError(

            f"TripAdvisor dataset not found:\n{TRIP_FILE}"

        )

    extension = TRIP_FILE.suffix.lower()

    if extension == ".csv":

        df = pd.read_csv(

            TRIP_FILE,

            low_memory=False

        )

    elif extension == ".json":

        df = pd.read_json(

            TRIP_FILE,

            lines=True

        )

    else:

        raise ValueError(

            "Unsupported TripAdvisor file format."

        )

    return process_dataframe(

        df,

        "TripAdvisor"

    )

# ==========================================================
# END OF BLOCK 4
#
# Continue with BLOCK 5
# ==========================================================

# ==========================================================
# BLOCK 5
# Google Play + Yelp Loaders
# ==========================================================

# ----------------------------------------------------------
# Google Play Loader
# ----------------------------------------------------------

def load_googleplay():

    logger.info("=" * 60)
    logger.info("Loading Google Play Dataset")
    logger.info("=" * 60)

    if not GOOGLEPLAY_FILE.exists():

        raise FileNotFoundError(
            f"Google Play dataset not found:\n{GOOGLEPLAY_FILE}"
        )

    extension = GOOGLEPLAY_FILE.suffix.lower()

    if extension == ".csv":

        df = pd.read_csv(
            GOOGLEPLAY_FILE,
            low_memory=False
        )

    elif extension == ".json":

        df = pd.read_json(
            GOOGLEPLAY_FILE,
            lines=True
        )

    else:

        raise ValueError(
            "Unsupported Google Play dataset format."
        )

    return process_dataframe(
        df,
        "GooglePlay"
    )


# ----------------------------------------------------------
# Yelp Loader
# ----------------------------------------------------------
def load_yelp():

    logger.info("=" * 60)
    logger.info("Loading Yelp Dataset")
    logger.info("=" * 60)

    if not YELP_FILE.exists():

        raise FileNotFoundError(
            f"Yelp dataset not found:\n{YELP_FILE}"
        )

    reviews = []

    SAMPLE_LIMIT = 50000

    with open(YELP_FILE, "r", encoding="utf-8") as f:

        for i, line in enumerate(f):

            try:

                review = json.loads(line)

            except Exception:

                continue

            # Reservoir Sampling
            if len(reviews) < SAMPLE_LIMIT:

                reviews.append(review)

            else:

                j = random.randint(0, i)

                if j < SAMPLE_LIMIT:

                    reviews[j] = review

    df = pd.DataFrame(reviews)

    logger.info(
        f"Randomly sampled {len(df):,} Yelp reviews."
    )

    return process_dataframe(
        df,
        "Yelp"
    )


# ----------------------------------------------------------
# Dataset Summary
# ----------------------------------------------------------

def print_dataset_summary(df, source):

    logger.info("-" * 60)

    logger.info(f"Dataset : {source}")

    logger.info(f"Reviews : {len(df):,}")

    if "rating" in df.columns:

        logger.info(
            f"Average Rating : {round(df['rating'].mean(),2)}"
        )

    logger.info(
        f"Average Words : {round(df['word_count'].mean(),2)}"
    )

    logger.info(
        f"Average Quality : {round(df['quality_score'].mean(),2)}"
    )

    logger.info("-" * 60)


# ----------------------------------------------------------
# Safe Dataset Loader
# ----------------------------------------------------------

def safe_load(loader_function, dataset_name):

    try:

        df = loader_function()

        print_dataset_summary(df, dataset_name)

        return df

    except Exception as e:

        logger.error(
            f"{dataset_name} failed to load."
        )

        logger.error(str(e))

        return pd.DataFrame()


# ==========================================================
# END OF BLOCK 5
#
# Continue with BLOCK 6
# ==========================================================

# ==========================================================
# BLOCK 6
# Merge + Sampling + Reports
# ==========================================================

# ----------------------------------------------------------
# Merge All Datasets
# ----------------------------------------------------------

def merge_datasets(datasets):

    logger.info("=" * 60)
    logger.info("Merging Datasets")
    logger.info("=" * 60)

    datasets = [df for df in datasets if len(df)]

    if len(datasets) == 0:
        raise ValueError("No datasets available.")

    merged = pd.concat(
        datasets,
        ignore_index=True
    )

    logger.info(f"Total Reviews Before Merge : {len(merged):,}")

    # Cross-dataset duplicate removal

    before = len(merged)

    merged["duplicate_key"] = merged["review"].apply(
        normalize_duplicate
    )

    merged = merged.drop_duplicates(
        subset="duplicate_key"
    )

    merged = merged.drop(
        columns=["duplicate_key"]
    )

    logger.info(
        f"Cross Dataset Duplicates Removed : {before-len(merged):,}"
    )

    logger.info(
        f"Remaining Reviews : {len(merged):,}"
    )

    return merged.reset_index(drop=True)


# ----------------------------------------------------------
# Balanced Sampling By Source
# ----------------------------------------------------------

def sample_by_source(df):

    logger.info("=" * 60)
    logger.info("Balanced Sampling")
    logger.info("=" * 60)

    final_parts = []

    for source, target in SAMPLES.items():

        subset = df[df["source"] == source]

        logger.info(
            f"{source:15} Available : {len(subset):,}"
        )

        if len(subset) == 0:
            continue
        
        target = min(target, len(subset))
        
        sampled = balanced_sample(
            subset,
            target
        )
        logger.info(
            f"{source:15} Selected : {len(sampled):,}"
        )

        final_parts.append(sampled)

    final_df = pd.concat(
        final_parts,
        ignore_index=True
    )

    final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    logger.info(
        f"\nFinal Dataset Size : {len(final_df):,}"
    )

    return final_df


# ----------------------------------------------------------
# Save Dataset Statistics
# ----------------------------------------------------------

def save_statistics(df):

    stats = dataset_statistics(df)

    stats.to_csv(
        STATISTICS_FILE,
        index=False
    )

    logger.info("Saved dataset statistics.")


# ----------------------------------------------------------
# Rating Distribution
# ----------------------------------------------------------

def save_rating_distribution(df):

    if "rating" not in df.columns:
        return

    rating = (
        df["rating"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating.columns = [
        "Rating",
        "Count"
    ]

    rating.to_csv(
        RATING_FILE,
        index=False
    )


# ----------------------------------------------------------
# Source Distribution
# ----------------------------------------------------------

def save_source_distribution(df):

    source = (
        df["source"]
        .value_counts()
        .reset_index()
    )

    source.columns = [
        "Source",
        "Count"
    ]

    source.to_csv(
        SOURCE_FILE,
        index=False
    )


# ----------------------------------------------------------
# Category Distribution
# ----------------------------------------------------------

def save_category_distribution(df):

    if "category" not in df.columns:
        return

    category = (
        df["category"]
        .value_counts()
        .reset_index()
    )

    category.columns = [
        "Category",
        "Count"
    ]

    category.to_csv(
        CATEGORY_FILE,
        index=False
    )


# ----------------------------------------------------------
# Quality Report
# ----------------------------------------------------------

def save_quality_report(df):

    report = df[
        [
            "quality_score",
            "word_count",
            "char_count",
            "lexical_diversity"
        ]
    ]

    report.describe().to_csv(
        QUALITY_FILE
    )


# ----------------------------------------------------------
# Cleaning Report
# ----------------------------------------------------------

def save_cleaning_report():

    report = pd.DataFrame(
        cleaning_report
    )

    report.to_csv(
        CLEANING_FILE,
        index=False
    )


# ----------------------------------------------------------
# Save Final Dataset
# ----------------------------------------------------------

def save_dataset(df):

    columns = [

    "review",

    "rating",

    "source",

    "category",

    "word_count",

    "char_count",

    "quality_score",

    "lexical_diversity"

]

    existing = [
        c for c in columns
        if c in df.columns
    ]

    df = df[existing]

    df.to_csv(
        OUTPUT_DATASET,
        index=False,
        encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info(f"Dataset Saved : {OUTPUT_DATASET}")
    logger.info("=" * 60)


# ==========================================================
# END OF BLOCK 6
#
# Continue with BLOCK 7
# ==========================================================

# ==========================================================
# BLOCK 7
# Main Pipeline
# ==========================================================

def print_final_summary(df):

    logger.info("\n" + "=" * 70)
    logger.info("FINAL DATASET SUMMARY")
    logger.info("=" * 70)

    logger.info(f"Total Reviews          : {len(df):,}")

    if "source" in df.columns:

        logger.info("\nReviews by Source")

        logger.info("-" * 70)

        for source, count in df["source"].value_counts().items():

            logger.info(f"{source:20} {count:,}")

    if "rating" in df.columns:

        logger.info("\nAverage Rating")

        logger.info("-" * 70)

        logger.info(round(df["rating"].mean(), 2))

    logger.info("\nAverage Quality Score")

    logger.info("-" * 70)

    logger.info(round(df["quality_score"].mean(), 2))

    logger.info("\nAverage Review Length")

    logger.info("-" * 70)

    logger.info(round(df["word_count"].mean(), 2))

    logger.info("=" * 70)


# ----------------------------------------------------------
# Main Function
# ----------------------------------------------------------

def main():

    logger.info("=" * 70)
    logger.info("VeriSight Dataset Builder")
    logger.info("=" * 70)

    datasets = []

    # Amazon

    amazon = safe_load(load_amazon, "Amazon")

    if len(amazon):

        datasets.append(amazon)

    # Yelp

    yelp = safe_load(load_yelp, "Yelp")

    if len(yelp):

        datasets.append(yelp)

    # TripAdvisor

    trip = safe_load(load_tripadvisor, "TripAdvisor")

    if len(trip):

        datasets.append(trip)

    # Google Play

    gp = safe_load(load_googleplay, "GooglePlay")

    if len(gp):

        datasets.append(gp)

    if len(datasets) == 0:

        logger.error("No datasets were loaded.")

        return

    # Merge

    merged = merge_datasets(datasets)

    # Balanced Sampling

    final_dataset = sample_by_source(merged)

    # Save Dataset

    save_dataset(final_dataset)

    # Reports

    save_statistics(final_dataset)

    save_rating_distribution(final_dataset)

    save_source_distribution(final_dataset)

    save_category_distribution(final_dataset)

    save_quality_report(final_dataset)

    save_cleaning_report()

    # Summary

    print_final_summary(final_dataset)

    logger.info("\nDataset creation completed successfully.")

# ----------------------------------------------------------
# Entry Point
# ----------------------------------------------------------

if __name__ == "__main__":

    main()