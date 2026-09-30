import os
import re
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/phase2/yelpzip.csv"

OUTPUT_DIR = "data/phase2/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 50000
RANDOM_STATE = 42

# Output files
FEATURE_FILE = os.path.join(
    OUTPUT_DIR,
    "yelpzip_stage2_features.csv"
)

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "stage2_train.csv"
)

VAL_FILE = os.path.join(
    OUTPUT_DIR,
    "stage2_val.csv"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "stage2_test.csv"
)


# ============================================================
# BASIC TEXT FEATURES
# ============================================================

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome",
    "love", "loved", "perfect", "wonderful", "best",
    "fantastic", "nice", "enjoy", "enjoyed", "friendly",
    "delicious", "happy", "beautiful"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "worst", "horrible",
    "hate", "hated", "poor", "disappointing",
    "disappointed", "rude", "dirty", "slow", "expensive",
    "boring", "disgusting", "unhappy"
}


def text_features(text):
    """
    Extract lightweight linguistic features from a review.
    """

    if pd.isna(text):
        text = ""

    text = str(text)

    words = re.findall(r"\b\w+\b", text.lower())

    word_count = len(words)
    char_count = len(text)

    if word_count > 0:
        avg_word_length = np.mean([len(w) for w in words])
    else:
        avg_word_length = 0.0

    positive_count = sum(
        1 for w in words if w in POSITIVE_WORDS
    )

    negative_count = sum(
        1 for w in words if w in NEGATIVE_WORDS
    )

    uppercase_chars = sum(
        1 for c in text if c.isupper()
    )

    alphabetic_chars = sum(
        1 for c in text if c.isalpha()
    )

    if alphabetic_chars > 0:
        uppercase_ratio = (
            uppercase_chars / alphabetic_chars
        )
    else:
        uppercase_ratio = 0.0

    punctuation_count = sum(
        1 for c in text if c in ".,!?;:"
    )

    exclamation_count = text.count("!")
    question_count = text.count("?")

    digit_count = sum(
        1 for c in text if c.isdigit()
    )

    if char_count > 0:
        punctuation_ratio = (
            punctuation_count / char_count
        )

        digit_ratio = (
            digit_count / char_count
        )
    else:
        punctuation_ratio = 0.0
        digit_ratio = 0.0

    positive_ratio = (
        positive_count / word_count
        if word_count > 0 else 0.0
    )

    negative_ratio = (
        negative_count / word_count
        if word_count > 0 else 0.0
    )

    sentiment_score = (
        positive_count - negative_count
    ) / max(word_count, 1)

    return [
        word_count,
        char_count,
        avg_word_length,
        uppercase_ratio,
        punctuation_count,
        punctuation_ratio,
        exclamation_count,
        question_count,
        digit_count,
        digit_ratio,
        positive_count,
        negative_count,
        positive_ratio,
        negative_ratio,
        sentiment_score
    ]


# ============================================================
# PASS 1
# COLLECT USER / PRODUCT / TEXT STATISTICS
# ============================================================

print("=" * 70)
print("VERISIGHT - STAGE 2 PREPROCESSING")
print("=" * 70)

print("\nInput:", INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n\n"
        "Place yelpzip.csv inside data/phase2/"
    )


user_count = defaultdict(int)
user_rating_sum = defaultdict(float)
user_rating_values = defaultdict(list)
user_products = defaultdict(set)

product_count = defaultdict(int)
product_rating_sum = defaultdict(float)
product_rating_values = defaultdict(list)

text_count = defaultdict(int)

all_users = []
all_labels = []


print("\nPASS 1: Collecting behavioral statistics...")

chunk_number = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    chunk = chunk.rename(
        columns={
            "Unnamed: 0": "original_index"
        }
    )

    # --------------------------------------------------------
    # Normalize columns
    # --------------------------------------------------------

    chunk["user_id"] = chunk["user_id"].astype(str)
    chunk["prod_id"] = chunk["prod_id"].astype(str)

    chunk["rating"] = pd.to_numeric(
        chunk["rating"],
        errors="coerce"
    ).fillna(0)

    chunk["label"] = pd.to_numeric(
        chunk["label"],
        errors="coerce"
    )

    # Spam = 1
    # Genuine = 0

    chunk["spam"] = (
        chunk["label"] == -1
    ).astype(np.int8)

    # --------------------------------------------------------
    # Store user IDs and labels for splitting
    # --------------------------------------------------------

    all_users.extend(
        chunk["user_id"].tolist()
    )

    all_labels.extend(
        chunk["spam"].tolist()
    )

    # --------------------------------------------------------
    # User statistics
    # --------------------------------------------------------

    for uid, rating in zip(
        chunk["user_id"],
        chunk["rating"]
    ):

        user_count[uid] += 1
        user_rating_sum[uid] += float(rating)

    for uid, pid in zip(
        chunk["user_id"],
        chunk["prod_id"]
    ):

        # Keep unique products reviewed
        if len(user_products[uid]) < 10000:
            user_products[uid].add(pid)

    # --------------------------------------------------------
    # Product statistics
    # --------------------------------------------------------

    for pid, rating in zip(
        chunk["prod_id"],
        chunk["rating"]
    ):

        product_count[pid] += 1
        product_rating_sum[pid] += float(rating)

    # --------------------------------------------------------
    # Exact text frequency
    # --------------------------------------------------------

    for text in chunk["text"].fillna("").astype(str):

        text_key = text.strip().lower()

        text_count[text_key] += 1

    if chunk_number % 5 == 0:

        print(
            f"  Processed approximately "
            f"{chunk_number * CHUNK_SIZE:,} rows..."
        )


print("\nPASS 1 complete.")

print(
    f"Unique users:    {len(user_count):,}"
)

print(
    f"Unique products: {len(product_count):,}"
)

print(
    f"Unique texts:    {len(text_count):,}"
)


# ============================================================
# USER / PRODUCT MEAN FEATURES
# ============================================================

user_avg_rating = {
    uid: user_rating_sum[uid] / user_count[uid]
    for uid in user_count
}

user_unique_products = {
    uid: len(user_products[uid])
    for uid in user_count
}

product_avg_rating = {
    pid: product_rating_sum[pid] / product_count[pid]
    for pid in product_count
}


# ============================================================
# CREATE USER-BASED SPLIT
# ============================================================

print("\nCreating user-grouped train/validation/test split...")

split_df = pd.DataFrame({
    "user_id": all_users,
    "spam": all_labels
})

# Remove unnecessary duplicate rows for splitting
user_table = (
    split_df
    .groupby("user_id")["spam"]
    .agg(["count", "mean"])
    .reset_index()
)

# ------------------------------------------------------------
# First split:
# 80% train
# 20% temporary
# ------------------------------------------------------------

gss1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, temp_idx = next(
    gss1.split(
        split_df,
        split_df["spam"],
        groups=split_df["user_id"]
    )
)

train_users = set(
    split_df.iloc[train_idx]["user_id"]
)

temp_users = set(
    split_df.iloc[temp_idx]["user_id"]
)

# ------------------------------------------------------------
# Split temporary into validation/test
# ------------------------------------------------------------

temp_df = split_df.iloc[temp_idx].copy()

gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=RANDOM_STATE
)

val_idx, test_idx = next(
    gss2.split(
        temp_df,
        temp_df["spam"],
        groups=temp_df["user_id"]
    )
)

val_users = set(
    temp_df.iloc[val_idx]["user_id"]
)

test_users = set(
    temp_df.iloc[test_idx]["user_id"]
)


print("\nUser split:")
print(
    f"Train users:      {len(train_users):,}"
)

print(
    f"Validation users:  {len(val_users):,}"
)

print(
    f"Test users:        {len(test_users):,}"
)


# ============================================================
# PASS 2
# GENERATE FEATURES
# ============================================================

print("\nPASS 2: Generating review + behavioral features...")

first_write = True

processed_rows = 0


feature_columns = [
    "user_id",
    "prod_id",
    "rating",
    "date",

    # Review-level
    "word_count",
    "char_count",
    "avg_word_length",
    "uppercase_ratio",
    "punctuation_count",
    "punctuation_ratio",
    "exclamation_count",
    "question_count",
    "digit_count",
    "digit_ratio",
    "positive_count",
    "negative_count",
    "positive_ratio",
    "negative_ratio",
    "sentiment_score",

    # User-level
    "user_review_count",
    "user_avg_rating",
    "user_unique_products",

    # Product-level
    "product_review_count",
    "product_avg_rating",

    # Text repetition
    "exact_text_count",

    # Rating-text relationship
    "rating_sentiment_difference",

    # Target
    "spam"
]


for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):

    chunk = chunk.rename(
        columns={
            "Unnamed: 0": "original_index"
        }
    )

    chunk["user_id"] = chunk["user_id"].astype(str)
    chunk["prod_id"] = chunk["prod_id"].astype(str)

    chunk["rating"] = pd.to_numeric(
        chunk["rating"],
        errors="coerce"
    ).fillna(0)

    chunk["label"] = pd.to_numeric(
        chunk["label"],
        errors="coerce"
    )

    chunk["spam"] = (
        chunk["label"] == -1
    ).astype(np.int8)

    # --------------------------------------------------------
    # Text features
    # --------------------------------------------------------

    text_feature_values = []

    for text in chunk["text"].fillna("").astype(str):

        text_feature_values.append(
            text_features(text)
        )

    text_feature_df = pd.DataFrame(
        text_feature_values,
        columns=[
            "word_count",
            "char_count",
            "avg_word_length",
            "uppercase_ratio",
            "punctuation_count",
            "punctuation_ratio",
            "exclamation_count",
            "question_count",
            "digit_count",
            "digit_ratio",
            "positive_count",
            "negative_count",
            "positive_ratio",
            "negative_ratio",
            "sentiment_score"
        ]
    )

    # --------------------------------------------------------
    # User features
    # --------------------------------------------------------

    chunk["user_review_count"] = (
        chunk["user_id"]
        .map(user_count)
        .fillna(0)
    )

    chunk["user_avg_rating"] = (
        chunk["user_id"]
        .map(user_avg_rating)
        .fillna(0)
    )

    chunk["user_unique_products"] = (
        chunk["user_id"]
        .map(user_unique_products)
        .fillna(0)
    )

    # --------------------------------------------------------
    # Product features
    # --------------------------------------------------------

    chunk["product_review_count"] = (
        chunk["prod_id"]
        .map(product_count)
        .fillna(0)
    )

    chunk["product_avg_rating"] = (
        chunk["prod_id"]
        .map(product_avg_rating)
        .fillna(0)
    )

    # --------------------------------------------------------
    # Exact duplicate text count
    # --------------------------------------------------------

    normalized_text = (
        chunk["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    chunk["exact_text_count"] = (
        normalized_text
        .map(text_count)
        .fillna(1)
    )

    # --------------------------------------------------------
    # Combine text features
    # --------------------------------------------------------

    chunk = pd.concat(
        [
            chunk.reset_index(drop=True),
            text_feature_df
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Rating-text consistency
    #
    # Positive sentiment should generally correspond to
    # higher ratings and negative sentiment to lower ratings.
    # This is a signal, NOT a hard rule.
    # --------------------------------------------------------

    expected_sentiment = (
        (chunk["rating"] - 3.0) / 2.0
    )

    chunk["rating_sentiment_difference"] = (
        chunk["sentiment_score"]
        - expected_sentiment
    ).abs()

    # --------------------------------------------------------
    # Select final features
    # --------------------------------------------------------

    final_chunk = chunk[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    final_chunk.to_csv(
        FEATURE_FILE,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )

    first_write = False

    processed_rows += len(final_chunk)

    if processed_rows % 250000 < CHUNK_SIZE:

        print(
            f"  Generated features for "
            f"{processed_rows:,} rows..."
        )


print("\nFeature generation complete.")


# ============================================================
# CREATE FINAL TRAIN / VAL / TEST FILES
# ============================================================

print("\nCreating final split files...")

# Read generated numerical dataset.
# Text is intentionally not included in this first
# structured-feature pipeline.

features = pd.read_csv(
    FEATURE_FILE
)

features["user_id"] = (
    features["user_id"].astype(str)
)

train_mask = features["user_id"].isin(
    train_users
)

val_mask = features["user_id"].isin(
    val_users
)

test_mask = features["user_id"].isin(
    test_users
)

train_df = features[train_mask].copy()
val_df = features[val_mask].copy()
test_df = features[test_mask].copy()


train_df.to_csv(
    TRAIN_FILE,
    index=False
)

val_df.to_csv(
    VAL_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 PREPROCESSING COMPLETE")
print("=" * 70)

print("\nDataset:")
print(
    f"Total rows: {len(features):,}"
)

print("\nTrain:")
print(
    f"Rows: {len(train_df):,}"
)

print(
    train_df["spam"]
    .value_counts()
    .sort_index()
)

print("\nValidation:")
print(
    f"Rows: {len(val_df):,}"
)

print(
    val_df["spam"]
    .value_counts()
    .sort_index()
)

print("\nTest:")
print(
    f"Rows: {len(test_df):,}"
)

print(
    test_df["spam"]
    .value_counts()
    .sort_index()
)

print("\nFeature count:")

print(
    len(feature_columns) - 5
)

print("\nOutput files:")

print(FEATURE_FILE)
print(TRAIN_FILE)
print(VAL_FILE)
print(TEST_FILE)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)