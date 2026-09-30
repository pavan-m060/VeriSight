import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# VERISIGHT - STAGE 2 HYBRID DATASET PREPARATION
#
# Creates:
#   Raw YelpZIP text
#   +
#   Existing V3 behavioral features
#
# Output:
#   data/phase2/processed_hybrid/
#       stage2_train_hybrid.csv
#       stage2_val_hybrid.csv
#       stage2_test_hybrid.csv
#
# The SAME user-grouped split used by the original
# Stage 2 V3 pipeline is reproduced.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/phase2/yelpzip.csv"

V3_DIR = "data/phase2/processed_behavior_v3"

OUTPUT_DIR = "data/phase2/processed_hybrid"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
CHUNK_SIZE = 50000


# Existing V3 datasets
V3_TRAIN = os.path.join(
    V3_DIR,
    "train_behavior.csv"
)

V3_VAL = os.path.join(
    V3_DIR,
    "val_behavior.csv"
)

V3_TEST = os.path.join(
    V3_DIR,
    "test_behavior.csv"
)


# New hybrid datasets
TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "stage2_train_hybrid.csv"
)

VAL_FILE = os.path.join(
    OUTPUT_DIR,
    "stage2_val_hybrid.csv"
)

TEST_FILE = os.path.join(
    OUTPUT_DIR,
    "stage2_test_hybrid.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("VERISIGHT - STAGE 2 HYBRID DATASET PREPARATION")
print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

required_files = [
    INPUT_FILE,
    V3_TRAIN,
    V3_VAL,
    V3_TEST
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print("FOUND:", path)


# ============================================================
# LOAD EXISTING V3 DATASETS
# ============================================================

print("\n" + "=" * 70)
print("LOADING V3 BEHAVIOR DATA")
print("=" * 70)

v3_train = pd.read_csv(V3_TRAIN)

v3_val = pd.read_csv(V3_VAL)

v3_test = pd.read_csv(V3_TEST)


print(
    f"\nV3 train: {len(v3_train):,}"
)

print(
    f"V3 validation: {len(v3_val):,}"
)

print(
    f"V3 test: {len(v3_test):,}"
)


# ============================================================
# RECREATE ORIGINAL USER-GROUPED SPLIT
#
# SAME LOGIC AS prepare_stage2_yelpzip.py
#
# 80% TRAIN
# 20% TEMP
# TEMP -> 50% VALIDATION / 50% TEST
# RANDOM_STATE = 42
# ============================================================

print("\n" + "=" * 70)
print("RECREATING ORIGINAL USER-GROUPED SPLIT")
print("=" * 70)


users = []

labels = []


print("\nReading user IDs from YelpZIP...")

chunk_number = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    chunk["user_id"] = (
        chunk["user_id"]
        .astype(str)
    )

    chunk["label"] = pd.to_numeric(
        chunk["label"],
        errors="coerce"
    )

    # Original YelpZIP:
    # -1 = fake
    #  1 = real
    #
    # Our project:
    #  1 = spam
    #  0 = genuine

    chunk["spam"] = (
        chunk["label"] == -1
    ).astype(np.int8)

    users.extend(
        chunk["user_id"].tolist()
    )

    labels.extend(
        chunk["spam"].tolist()
    )

    if chunk_number % 5 == 0:

        print(
            f"  Processed approximately "
            f"{chunk_number * CHUNK_SIZE:,} rows..."
        )


split_df = pd.DataFrame(
    {
        "user_id": users,
        "spam": labels
    }
)


print(
    f"\nRows used for split: "
    f"{len(split_df):,}"
)


# ============================================================
# FIRST SPLIT
#
# 80% TRAIN
# 20% TEMP
# ============================================================

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


# ============================================================
# SECOND SPLIT
#
# TEMP -> 50% VALIDATION
# TEMP -> 50% TEST
# ============================================================

temp_df = split_df.iloc[
    temp_idx
].copy()


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


# ============================================================
# SPLIT REPORT
# ============================================================

print("\nUser split:")

print(
    f"Train users:      {len(train_users):,}"
)

print(
    f"Validation users: {len(val_users):,}"
)

print(
    f"Test users:       {len(test_users):,}"
)


# ============================================================
# USER OVERLAP CHECK
# ============================================================

print("\nChecking user overlap...")


train_val_overlap = (
    train_users & val_users
)

train_test_overlap = (
    train_users & test_users
)

val_test_overlap = (
    val_users & test_users
)


if train_val_overlap:

    raise ValueError(
        "Train/validation user overlap detected!"
    )


if train_test_overlap:

    raise ValueError(
        "Train/test user overlap detected!"
    )


if val_test_overlap:

    raise ValueError(
        "Validation/test user overlap detected!"
    )


print("User overlap check: PASSED")


# ============================================================
# LOAD ORIGINAL YELPZIP TEXT
# ============================================================

print("\n" + "=" * 70)
print("LOADING ORIGINAL YELPZIP TEXT")
print("=" * 70)


columns_needed = [
    "user_id",
    "prod_id",
    "rating",
    "date",
    "text",
    "label"
]


parts = []

chunk_number = 0


for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=columns_needed,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    # Normalize identifiers
    chunk["user_id"] = (
        chunk["user_id"]
        .astype(str)
    )

    chunk["prod_id"] = (
        chunk["prod_id"]
        .astype(str)
    )

    # Normalize rating
    chunk["rating"] = pd.to_numeric(
        chunk["rating"],
        errors="coerce"
    ).fillna(0)

    # Normalize label
    chunk["label"] = pd.to_numeric(
        chunk["label"],
        errors="coerce"
    )

    # Convert YelpZIP label
    chunk["spam"] = (
        chunk["label"] == -1
    ).astype(np.int8)

    # Preserve raw review text
    chunk["text"] = (
        chunk["text"]
        .fillna("")
        .astype(str)
    )

    parts.append(
        chunk[
            [
                "user_id",
                "prod_id",
                "rating",
                "date",
                "text",
                "spam"
            ]
        ]
    )

    if chunk_number % 5 == 0:

        print(
            f"  Loaded approximately "
            f"{chunk_number * CHUNK_SIZE:,} rows..."
        )


raw = pd.concat(
    parts,
    ignore_index=True
)


del parts


print(
    f"\nOriginal rows loaded: "
    f"{len(raw):,}"
)


# ============================================================
# APPLY SAME USER SPLIT
# ============================================================

print("\nApplying user split...")


train_raw = raw[
    raw["user_id"].isin(train_users)
].copy()


val_raw = raw[
    raw["user_id"].isin(val_users)
].copy()


test_raw = raw[
    raw["user_id"].isin(test_users)
].copy()


print(
    f"Raw train rows: "
    f"{len(train_raw):,}"
)

print(
    f"Raw validation rows: "
    f"{len(val_raw):,}"
)

print(
    f"Raw test rows: "
    f"{len(test_raw):,}"
)


# ============================================================
# ALIGNMENT CHECK
#
# IMPORTANT:
#
# The original V3 preparation script reads YelpZIP in its
# original order and writes the feature rows in that order.
#
# Therefore we DO NOT sort here.
#
# We preserve the original YelpZIP order.
# ============================================================


def check_alignment(
    raw_df,
    v3_df,
    split_name
):

    print(
        f"\nChecking {split_name} alignment..."
    )


    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    if len(raw_df) != len(v3_df):

        raise ValueError(
            f"\n{split_name} row count mismatch!\n"
            f"Raw: {len(raw_df):,}\n"
            f"V3:  {len(v3_df):,}"
        )


    # Reset index only.
    #
    # DO NOT SORT.
    #
    # Original row ordering is preserved.

    raw_check = (
        raw_df
        .reset_index(drop=True)
    )

    v3_check = (
        v3_df
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Columns that must match
    # --------------------------------------------------------

    columns = [
        "user_id",
        "prod_id",
        "rating",
        "date",
        "spam"
    ]


    for col in columns:

        a = (
            raw_check[col]
            .astype(str)
            .reset_index(drop=True)
        )

        b = (
            v3_check[col]
            .astype(str)
            .reset_index(drop=True)
        )


        if not a.equals(b):

            mismatch = np.where(
                a.values != b.values
            )[0]


            if len(mismatch) > 0:

                first = int(
                    mismatch[0]
                )

            else:

                first = -1


            print(
                f"\nFirst mismatch "
                f"in {split_name}:"
            )


            if first >= 0:

                print(
                    "\nRAW:"
                )

                print(
                    raw_check.iloc[first][
                        [
                            "user_id",
                            "prod_id",
                            "rating",
                            "date",
                            "spam"
                        ]
                    ].to_dict()
                )


                print(
                    "\nV3:"
                )

                print(
                    v3_check.iloc[first][
                        [
                            "user_id",
                            "prod_id",
                            "rating",
                            "date",
                            "spam"
                        ]
                    ].to_dict()
                )


            raise ValueError(
                f"\n{split_name} alignment failed "
                f"for column '{col}'.\n"
                f"First mismatch: {first}"
            )


    print(
        f"  {split_name} alignment: OK"
    )


    return raw_check, v3_check


# ============================================================
# CHECK ALL THREE SPLITS
# ============================================================

train_raw, train_v3 = check_alignment(
    train_raw,
    v3_train,
    "TRAIN"
)


val_raw, val_v3 = check_alignment(
    val_raw,
    v3_val,
    "VALIDATION"
)


test_raw, test_v3 = check_alignment(
    test_raw,
    v3_test,
    "TEST"
)


# ============================================================
# CREATE HYBRID DATASET
# ============================================================

print("\n" + "=" * 70)
print("CREATING HYBRID DATASETS")
print("=" * 70)


def create_hybrid(
    raw_df,
    v3_df
):

    # --------------------------------------------------------
    # Raw information
    # --------------------------------------------------------

    raw_small = raw_df[
        [
            "user_id",
            "prod_id",
            "rating",
            "date",
            "text",
            "spam"
        ]
    ].copy()


    # --------------------------------------------------------
    # Existing V3 behavioral features
    #
    # Remove identifiers and target because they are already
    # present in raw_small.
    # --------------------------------------------------------

    behavior = v3_df.drop(
        columns=[
            "user_id",
            "prod_id",
            "rating",
            "date",
            "spam"
        ],
        errors="ignore"
    ).copy()


    # --------------------------------------------------------
    # Concatenate by row position
    # --------------------------------------------------------

    result = pd.concat(
        [
            raw_small.reset_index(drop=True),
            behavior.reset_index(drop=True)
        ],
        axis=1
    )


    return result


train_hybrid = create_hybrid(
    train_raw,
    train_v3
)


val_hybrid = create_hybrid(
    val_raw,
    val_v3
)


test_hybrid = create_hybrid(
    test_raw,
    test_v3
)


# ============================================================
# FINAL DATASET CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET CHECKS")
print("=" * 70)


datasets = [
    ("TRAIN", train_hybrid),
    ("VALIDATION", val_hybrid),
    ("TEST", test_hybrid)
]


for name, df in datasets:

    print(
        f"\n{name}"
    )

    print(
        f"Rows:    {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    missing_text = (
        df["text"]
        .isna()
        .sum()
    )


    if missing_text > 0:

        raise ValueError(
            f"{name} contains "
            f"{missing_text:,} missing text values!"
        )


    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    missing_labels = (
        df["spam"]
        .isna()
        .sum()
    )


    if missing_labels > 0:

        raise ValueError(
            f"{name} contains "
            f"{missing_labels:,} missing labels!"
        )


    # --------------------------------------------------------
    # Print distribution
    # --------------------------------------------------------

    print(
        "\nLabel distribution:"
    )

    print(
        df["spam"]
        .value_counts()
        .sort_index()
    )


    # --------------------------------------------------------
    # Print text statistics
    # --------------------------------------------------------

    print(
        "\nText statistics:"
    )

    text_lengths = (
        df["text"]
        .str.split()
        .str.len()
    )


    print(
        f"Mean words: "
        f"{text_lengths.mean():.2f}"
    )

    print(
        f"Median words: "
        f"{text_lengths.median():.2f}"
    )

    print(
        f"Maximum words: "
        f"{text_lengths.max():,}"
    )


# ============================================================
# CHECK USER OVERLAP AGAIN
# ============================================================

print(
    "\nChecking final dataset user separation..."
)


train_user_set = set(
    train_hybrid["user_id"]
)

val_user_set = set(
    val_hybrid["user_id"]
)

test_user_set = set(
    test_hybrid["user_id"]
)


if train_user_set & val_user_set:

    raise ValueError(
        "Train/validation user overlap!"
    )


if train_user_set & test_user_set:

    raise ValueError(
        "Train/test user overlap!"
    )


if val_user_set & test_user_set:

    raise ValueError(
        "Validation/test user overlap!"
    )


print(
    "User separation: PASSED"
)


# ============================================================
# SAVE FILES
# ============================================================

print("\n" + "=" * 70)
print("SAVING HYBRID DATASETS")
print("=" * 70)


print(
    "\nWriting train..."
)

train_hybrid.to_csv(
    TRAIN_FILE,
    index=False
)


print(
    "Writing validation..."
)

val_hybrid.to_csv(
    VAL_FILE,
    index=False
)


print(
    "Writing test..."
)

test_hybrid.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 HYBRID DATASET READY")
print("=" * 70)


print("\nTRAIN")
print(
    f"Rows: {len(train_hybrid):,}"
)
print(
    f"Genuine: "
    f"{(train_hybrid['spam'] == 0).sum():,}"
)
print(
    f"Spam: "
    f"{(train_hybrid['spam'] == 1).sum():,}"
)


print("\nVALIDATION")
print(
    f"Rows: {len(val_hybrid):,}"
)
print(
    f"Genuine: "
    f"{(val_hybrid['spam'] == 0).sum():,}"
)
print(
    f"Spam: "
    f"{(val_hybrid['spam'] == 1).sum():,}"
)


print("\nTEST")
print(
    f"Rows: {len(test_hybrid):,}"
)
print(
    f"Genuine: "
    f"{(test_hybrid['spam'] == 0).sum():,}"
)
print(
    f"Spam: "
    f"{(test_hybrid['spam'] == 1).sum():,}"
)


print("\nColumns:")
for i, col in enumerate(
    train_hybrid.columns,
    start=1
):
    print(
        f"{i:02d}. {col}"
    )


print("\nOutput files:")

print(
    TRAIN_FILE
)

print(
    VAL_FILE
)

print(
    TEST_FILE
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)