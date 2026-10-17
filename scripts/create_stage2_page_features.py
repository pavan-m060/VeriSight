import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter


INPUT = "data/phase2/yelpzip.csv"
OUTPUT_DIR = Path("data/phase2/page_features")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# SENTIMENT WORDS
# ---------------------------------------------------------

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing",
    "awesome", "love", "loved", "nice",
    "perfect", "best", "comfortable",
    "happy", "worth", "recommend",
    "beautiful", "fantastic", "wonderful"
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "horrible",
    "worst", "hate", "hated", "awful",
    "disappointed", "disappointing",
    "cheap", "broken", "slow",
    "uncomfortable", "waste", "return",
    "refund", "problem", "issue"
}


# ---------------------------------------------------------
# TEXT FEATURES
# ---------------------------------------------------------

def word_count(text):
    return len(
        re.findall(
            r"\b\w+\b",
            str(text)
        )
    )


def char_count(text):
    return len(str(text))


def avg_word_length(text):

    words = re.findall(
        r"\b\w+\b",
        str(text)
    )

    if not words:
        return 0.0

    return np.mean(
        [len(w) for w in words]
    )


def punctuation_count(text):

    return len(
        re.findall(
            r"[!?.,;:]+",
            str(text)
        )
    )


def exclamation_count(text):

    return str(text).count("!")


def question_count(text):

    return str(text).count("?")


def uppercase_ratio(text):

    text = str(text)

    letters = [
        c for c in text
        if c.isalpha()
    ]

    if not letters:
        return 0.0

    return (
        sum(
            c.isupper()
            for c in letters
        )
        / len(letters)
    )


def digit_ratio(text):

    text = str(text)

    if not text:
        return 0.0

    return (
        sum(
            c.isdigit()
            for c in text
        )
        / len(text)
    )


def sentiment_score(text):

    words = re.findall(
        r"\b\w+\b",
        str(text).lower()
    )

    if not words:
        return 0.0

    positive = sum(
        w in POSITIVE_WORDS
        for w in words
    )

    negative = sum(
        w in NEGATIVE_WORDS
        for w in words
    )

    return (
        positive - negative
    ) / len(words)


# ---------------------------------------------------------
# DUPLICATE FEATURE
# ---------------------------------------------------------

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9 ]",
        "",
        text
    )

    return text.strip()


# ---------------------------------------------------------
# RATING / SENTIMENT
# ---------------------------------------------------------

def rating_sentiment_difference(
    rating,
    sentiment
):

    expected = (
        (rating - 3.0)
        / 2.0
    )

    return abs(
        expected - sentiment
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

print("=" * 70)
print("VERISIGHT STAGE 2 PAGE-AVAILABLE FEATURE GENERATOR")
print("=" * 70)

print("\nLoading YelpZIP...")

df = pd.read_csv(
    INPUT,
    low_memory=False
)

print(
    f"Rows loaded: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ---------------------------------------------------------
# NORMALIZE LABEL
# ---------------------------------------------------------

# YelpZIP:
# -1 = spam
#  1 = genuine

df["spam"] = (
    df["label"] == -1
).astype(int)


# ---------------------------------------------------------
# BASIC CLEANING
# ---------------------------------------------------------

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
)

df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
).fillna(3.0)


# ---------------------------------------------------------
# DATE
# ---------------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ---------------------------------------------------------
# TEXT FEATURES
# ---------------------------------------------------------

print("\nCreating text features...")

df["word_count"] = (
    df["text"].apply(
        word_count
    )
)

df["char_count"] = (
    df["text"].apply(
        char_count
    )
)

df["avg_word_length"] = (
    df["text"].apply(
        avg_word_length
    )
)

df["punctuation_count"] = (
    df["text"].apply(
        punctuation_count
    )
)

df["exclamation_count"] = (
    df["text"].apply(
        exclamation_count
    )
)

df["question_count"] = (
    df["text"].apply(
        question_count
    )
)

df["uppercase_ratio"] = (
    df["text"].apply(
        uppercase_ratio
    )
)

df["digit_ratio"] = (
    df["text"].apply(
        digit_ratio
    )
)

df["sentiment_score"] = (
    df["text"].apply(
        sentiment_score
    )
)


# ---------------------------------------------------------
# VERIFIED / HELPFUL
# ---------------------------------------------------------

# YelpZIP does not contain Amazon's
# "Verified Purchase" field.
#
# Therefore we use 0 as unavailable.
# The Amazon pipeline will provide the real
# value when available.

df["verified_purchase"] = 0


# Helpful votes are also unavailable
# in YelpZIP.

df["helpful_count"] = 0


# ---------------------------------------------------------
# RATING + SENTIMENT
# ---------------------------------------------------------

df["rating_sentiment_difference"] = (
    df.apply(
        lambda row:
        rating_sentiment_difference(
            row["rating"],
            row["sentiment_score"]
        ),
        axis=1
    )
)


# ---------------------------------------------------------
# REVIEW DATE FEATURES
# ---------------------------------------------------------

print(
    "Creating temporal features..."
)

df = df.sort_values(
    "date"
).reset_index(
    drop=True
)

df["review_date_ordinal"] = (
    df["date"]
    .map(
        lambda x:
        x.toordinal()
        if pd.notna(x)
        else 0
    )
)


# Reviews on the same date

date_counts = (
    df.groupby("date")
    .size()
)

df["reviews_same_date"] = (
    df["date"]
    .map(date_counts)
    .fillna(1)
)


# ---------------------------------------------------------
# RATING DISTRIBUTION FEATURES
# ---------------------------------------------------------

print(
    "Creating rating features..."
)

rating_counts = (
    df.groupby(
        "prod_id"
    )["rating"]
    .transform("count")
)

five_star_counts = (
    df.assign(
        five_star=(
            df["rating"] == 5
        ).astype(int)
    )
    .groupby("prod_id")
    ["five_star"]
    .transform("sum")
)

one_star_counts = (
    df.assign(
        one_star=(
            df["rating"] == 1
        ).astype(int)
    )
    .groupby("prod_id")
    ["one_star"]
    .transform("sum")
)

df["product_review_count"] = (
    rating_counts
)

df["product_five_star_ratio"] = (
    five_star_counts
    / rating_counts.replace(
        0,
        1
    )
)

df["product_one_star_ratio"] = (
    one_star_counts
    / rating_counts.replace(
        0,
        1
    )
)


# ---------------------------------------------------------
# DUPLICATE REVIEW FEATURE
# ---------------------------------------------------------

print(
    "Creating duplicate features..."
)

df["normalized_text"] = (
    df["text"].apply(
        normalize_text
    )
)

text_counts = (
    df.groupby(
        "normalized_text"
    )["normalized_text"]
    .transform("count")
)

df["duplicate_text_count"] = (
    text_counts
)


df["is_duplicate_text"] = (
    df["duplicate_text_count"] > 1
).astype(int)


# ---------------------------------------------------------
# REMOVE UNNECESSARY COLUMNS
# ---------------------------------------------------------

feature_columns = [

    "rating",

    "word_count",
    "char_count",
    "avg_word_length",

    "uppercase_ratio",

    "punctuation_count",
    "exclamation_count",
    "question_count",

    "digit_ratio",

    "sentiment_score",

    "verified_purchase",
    "helpful_count",

    "rating_sentiment_difference",

    "review_date_ordinal",
    "reviews_same_date",

    "product_review_count",
    "product_five_star_ratio",
    "product_one_star_ratio",

    "duplicate_text_count",
    "is_duplicate_text"
]


# ---------------------------------------------------------
# FINAL DATASET
# ---------------------------------------------------------

final_columns = (
    feature_columns
    + ["spam"]
)

features = df[
    final_columns
].copy()


# Replace invalid values

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

features = features.fillna(0)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output_file = (
    OUTPUT_DIR
    / "stage2_page_features.csv"
)

print(
    "\nSaving feature dataset..."
)

features.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)

print(
    f"Rows: {len(features):,}"
)

print(
    f"Features: {len(feature_columns)}"
)

print(
    "\nSpam distribution:"
)

print(
    features["spam"]
    .value_counts()
)

print(
    "\nOutput:"
)

print(
    output_file
)

print()