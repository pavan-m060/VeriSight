import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# VERISIGHT - STEP 1
# STAGE 1 DATASET AUDIT + LENGTH BALANCING
# ============================================================

INPUT_FILE = "data/stage1/stage1_final_v3.csv"

OUTPUT_DIR = "data/stage1/audit"

AUDIT_FILE = os.path.join(
    OUTPUT_DIR,
    "stage1_v3_audit.csv"
)

BALANCED_FILE = os.path.join(
    OUTPUT_DIR,
    "stage1_v3_length_balanced_candidate.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "stage1_v3_audit_summary.txt"
)

RANDOM_STATE = 42


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# WORD COUNT
# ============================================================

def count_words(text):

    if not isinstance(text, str):
        return 0

    return len(
        re.findall(
            r"\b[\w]+(?:['-][\w]+)*\b",
            text
        )
    )


# ============================================================
# SENTENCE COUNT
# ============================================================

def count_sentences(text):

    if not isinstance(text, str):
        return 0

    sentences = re.split(
        r"[.!?]+",
        text
    )

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    return len(sentences)


# ============================================================
# CHARACTER COUNT
# ============================================================

def count_characters(text):

    if not isinstance(text, str):
        return 0

    return len(text)


# ============================================================
# NORMALIZE REVIEW
# ============================================================

def normalize_review(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.strip()


# ============================================================
# LENGTH BIN
# ============================================================

def get_length_bin(words):

    if 20 <= words <= 40:
        return "20-40"

    elif 41 <= words <= 60:
        return "41-60"

    elif 61 <= words <= 80:
        return "61-80"

    elif 81 <= words <= 100:
        return "81-100"

    elif 101 <= words <= 120:
        return "101-120"

    elif 121 <= words <= 150:
        return "121-150"

    elif 151 <= words <= 200:
        return "151-200"

    elif 201 <= words <= 300:
        return "201-300"

    elif 301 <= words <= 450:
        return "301-450"

    elif words < 20:
        return "<20"

    else:
        return "451+"


LENGTH_BIN_ORDER = [
    "<20",
    "20-40",
    "41-60",
    "61-80",
    "81-100",
    "101-120",
    "121-150",
    "151-200",
    "201-300",
    "301-450",
    "451+"
]


# ============================================================
# LOAD DATASET
# ============================================================

print("\n")
print("=" * 80)
print("VERISIGHT - STAGE 1 DATASET AUDIT")
print("=" * 80)

print(
    f"\nLoading:\n{INPUT_FILE}"
)

if not os.path.exists(INPUT_FILE):

    print(
        "\nERROR: Dataset not found."
    )

    print(
        "Make sure you are running this from:"
    )

    print(
        r"C:\Users\ganes\Downloads\VeriSight"
    )

    raise SystemExit


df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nDataset shape: {df.shape}"
)


# ============================================================
# BASIC COLUMN CHECK
# ============================================================

required_columns = [
    "review",
    "source",
    "label"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print(
        "\nERROR: Missing columns:"
    )

    print(
        missing_columns
    )

    raise SystemExit


# ============================================================
# REMOVE EMPTY REVIEWS
# ============================================================

before = len(df)

df["review"] = (
    df["review"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["review"] != ""
].copy()

after = len(df)

print(
    f"\nEmpty reviews removed: {before - after}"
)


# ============================================================
# CREATE TEXT STATISTICS
# ============================================================

print(
    "\nCalculating text statistics..."
)

df["word_count"] = (
    df["review"]
    .apply(count_words)
)

df["sentence_count"] = (
    df["review"]
    .apply(count_sentences)
)

df["character_count"] = (
    df["review"]
    .apply(count_characters)
)

df["normalized_review"] = (
    df["review"]
    .apply(normalize_review)
)

df["length_bin"] = (
    df["word_count"]
    .apply(get_length_bin)
)


# ============================================================
# 1. LABEL DISTRIBUTION
# ============================================================

print("\n")
print("=" * 80)
print("1. LABEL DISTRIBUTION")
print("=" * 80)

label_counts = (
    df["label"]
    .value_counts()
    .sort_index()
)

print(
    label_counts
)

print(
    "\nPercentage:"
)

print(
    (
        df["label"]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ============================================================
# 2. SOURCE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 80)
print("2. SOURCE DISTRIBUTION")
print("=" * 80)

print(
    pd.crosstab(
        df["source"],
        df["label"]
    )
)


# ============================================================
# 3. GENERATOR DISTRIBUTION
# ============================================================

print("\n")
print("=" * 80)
print("3. GENERATOR DISTRIBUTION")
print("=" * 80)

if "generator_model" in df.columns:

    print(
        df["generator_model"]
        .fillna("Human")
        .value_counts()
    )

    print(
        "\nGenerator × Label:"
    )

    print(
        pd.crosstab(
            df["generator_model"]
            .fillna("Human"),
            df["label"]
        )
    )

else:

    print(
        "generator_model column not found."
    )


# ============================================================
# 4. WORD COUNT STATISTICS
# ============================================================

print("\n")
print("=" * 80)
print("4. WORD COUNT STATISTICS")
print("=" * 80)

word_stats = (
    df.groupby("label")["word_count"]
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
    word_stats.round(2)
)


# ============================================================
# HUMAN / AI WORD COUNT SEPARATION
# ============================================================

human_words = df[
    df["label"] == 0
]["word_count"]

ai_words = df[
    df["label"] == 1
]["word_count"]

print("\n")
print(
    "Human mean word count:",
    round(human_words.mean(), 2)
)

print(
    "AI mean word count:",
    round(ai_words.mean(), 2)
)

print(
    "Human median word count:",
    round(human_words.median(), 2)
)

print(
    "AI median word count:",
    round(ai_words.median(), 2)
)


# ============================================================
# 5. LENGTH BIN DISTRIBUTION
# ============================================================

print("\n")
print("=" * 80)
print("5. LENGTH BIN DISTRIBUTION")
print("=" * 80)

length_table = pd.crosstab(
    df["length_bin"],
    df["label"]
)

length_table = (
    length_table
    .reindex(
        LENGTH_BIN_ORDER,
        fill_value=0
    )
)

length_table.columns = [
    "Human"
    if col == 0
    else "AI"
    for col in length_table.columns
]

if "Human" not in length_table.columns:
    length_table["Human"] = 0

if "AI" not in length_table.columns:
    length_table["AI"] = 0

length_table["Total"] = (
    length_table["Human"]
    + length_table["AI"]
)

length_table["AI_%"] = (
    length_table["AI"]
    /
    length_table["Total"].replace(
        0,
        np.nan
    )
    * 100
)

print(
    length_table.round(2)
)


# ============================================================
# 6. SOURCE × LENGTH × LABEL
# ============================================================

print("\n")
print("=" * 80)
print("6. SOURCE × LENGTH DISTRIBUTION")
print("=" * 80)

source_length = pd.crosstab(
    [
        df["source"],
        df["length_bin"]
    ],
    df["label"]
)

source_length = (
    source_length
    .reindex(
        columns=[0, 1],
        fill_value=0
    )
)

source_length.columns = [
    "Human",
    "AI"
]

print(
    source_length
)


# ============================================================
# 7. DUPLICATE CHECK
# ============================================================

print("\n")
print("=" * 80)
print("7. DUPLICATE CHECK")
print("=" * 80)

duplicate_mask = (
    df["normalized_review"]
    .duplicated(
        keep=False
    )
)

duplicate_count = (
    df[
        duplicate_mask
    ].shape[0]
)

unique_duplicate_reviews = (
    df[
        duplicate_mask
    ]["normalized_review"]
    .nunique()
)

print(
    f"Rows involved in duplicates: "
    f"{duplicate_count}"
)

print(
    f"Unique duplicated texts: "
    f"{unique_duplicate_reviews}"
)


# ============================================================
# 8. DUPLICATES BETWEEN HUMAN AND AI
# ============================================================

human_texts = set(
    df[
        df["label"] == 0
    ]["normalized_review"]
)

ai_texts = set(
    df[
        df["label"] == 1
    ]["normalized_review"]
)

cross_label_duplicates = (
    human_texts.intersection(
        ai_texts
    )
)

print(
    "\nExact normalized Human/AI duplicates:",
    len(cross_label_duplicates)
)


# ============================================================
# 9. WORD COUNT ONLY BASELINE
# ============================================================

print("\n")
print("=" * 80)
print("9. WORD-COUNT-ONLY BASELINE")
print("=" * 80)

X = df[
    ["word_count"]
].values

y = df[
    "label"
].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

word_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000
)

word_model.fit(
    X_train_scaled,
    y_train
)

word_predictions = (
    word_model.predict(
        X_test_scaled
    )
)

word_accuracy = accuracy_score(
    y_test,
    word_predictions
)

print(
    f"\nWord-count-only accuracy: "
    f"{word_accuracy * 100:.2f}%"
)

print(
    "\nConfusion matrix:"
)

print(
    confusion_matrix(
        y_test,
        word_predictions
    )
)

print(
    "\nClassification report:"
)

print(
    classification_report(
        y_test,
        word_predictions,
        target_names=[
            "Human",
            "AI"
        ],
        digits=4
    )
)


# ============================================================
# 10. LENGTH OVERLAP
# ============================================================

print("\n")
print("=" * 80)
print("10. LENGTH OVERLAP")
print("=" * 80)

overlap_low = max(
    human_words.min(),
    ai_words.min()
)

overlap_high = min(
    human_words.max(),
    ai_words.max()
)

print(
    f"Overall overlapping word-count range: "
    f"{overlap_low} - {overlap_high}"
)

human_overlap = human_words[
    (
        human_words >= overlap_low
    )
    &
    (
        human_words <= overlap_high
    )
]

ai_overlap = ai_words[
    (
        ai_words >= overlap_low
    )
    &
    (
        ai_words <= overlap_high
    )
]

print(
    f"Human reviews in overlap: "
    f"{len(human_overlap)}"
)

print(
    f"AI reviews in overlap: "
    f"{len(ai_overlap)}"
)


# ============================================================
# 11. CREATE LENGTH-BALANCED CANDIDATE
# ============================================================

print("\n")
print("=" * 80)
print("11. CREATING LENGTH-BALANCED CANDIDATE")
print("=" * 80)

print(
    "\nFor every length bin:"
)

print(
    "Human count = AI count"
)

balanced_parts = []

for length_bin in LENGTH_BIN_ORDER:

    human_part = df[
        (
            df["length_bin"]
            == length_bin
        )
        &
        (
            df["label"]
            == 0
        )
    ].copy()

    ai_part = df[
        (
            df["length_bin"]
            == length_bin
        )
        &
        (
            df["label"]
            == 1
        )
    ].copy()

    if (
        len(human_part) == 0
        or len(ai_part) == 0
    ):

        print(
            f"{length_bin:>10}: "
            f"Human={len(human_part):5d}, "
            f"AI={len(ai_part):5d} "
            f"→ SKIPPED"
        )

        continue

    n = min(
        len(human_part),
        len(ai_part)
    )

    human_sample = (
        human_part
        .sample(
            n=n,
            random_state=RANDOM_STATE
        )
    )

    ai_sample = (
        ai_part
        .sample(
            n=n,
            random_state=RANDOM_STATE
        )
    )

    balanced_parts.append(
        human_sample
    )

    balanced_parts.append(
        ai_sample
    )

    print(
        f"{length_bin:>10}: "
        f"Human={len(human_part):5d}, "
        f"AI={len(ai_part):5d} "
        f"→ Selected={n} each"
    )


if not balanced_parts:

    print(
        "\nERROR: Could not create balanced dataset."
    )

    raise SystemExit


balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

balanced_df = (
    balanced_df
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 12. BALANCED DATASET SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("12. BALANCED DATASET SUMMARY")
print("=" * 80)

print(
    f"\nOriginal dataset: "
    f"{len(df)} rows"
)

print(
    f"Balanced candidate: "
    f"{len(balanced_df)} rows"
)

print(
    "\nLabel distribution:"
)

print(
    balanced_df[
        "label"
    ].value_counts()
)


print(
    "\nLength distribution:"
)

balanced_length = pd.crosstab(
    balanced_df["length_bin"],
    balanced_df["label"]
)

balanced_length = (
    balanced_length
    .reindex(
        LENGTH_BIN_ORDER,
        fill_value=0
    )
)

balanced_length.columns = [
    "Human"
    if c == 0
    else "AI"
    for c in balanced_length.columns
]

print(
    balanced_length
)


# ============================================================
# 13. CHECK BALANCE
# ============================================================

print("\n")
print(
    "Length-bin balance check:"
)

for length_bin in LENGTH_BIN_ORDER:

    human_n = len(
        balanced_df[
            (
                balanced_df[
                    "length_bin"
                ]
                == length_bin
            )
            &
            (
                balanced_df[
                    "label"
                ]
                == 0
            )
        ]
    )

    ai_n = len(
        balanced_df[
            (
                balanced_df[
                    "length_bin"
                ]
                == length_bin
            )
            &
            (
                balanced_df[
                    "label"
                ]
                == 1
            )
        ]
    )

    if human_n or ai_n:

        status = (
            "OK"
            if human_n == ai_n
            else "IMBALANCED"
        )

        print(
            f"{length_bin:>10}: "
            f"Human={human_n:5d}, "
            f"AI={ai_n:5d} "
            f"→ {status}"
        )


# ============================================================
# 14. SAVE AUDIT DATA
# ============================================================

audit_columns = [
    "review",
    "source",
    "label",
    "word_count",
    "sentence_count",
    "character_count",
    "length_bin"
]

if "generator_model" in df.columns:

    audit_columns.append(
        "generator_model"
    )

audit_df = df[
    audit_columns
].copy()

audit_df.to_csv(
    AUDIT_FILE,
    index=False
)


# ============================================================
# 15. SAVE BALANCED DATASET
# ============================================================

# Keep only useful original columns + audit features

balanced_columns = [
    col
    for col in [
        "review_id",
        "review",
        "source",
        "rating",
        "generator_model",
        "label",
        "word_count",
        "sentence_count",
        "character_count",
        "length_bin"
    ]
    if col in balanced_df.columns
]

balanced_df[
    balanced_columns
].to_csv(
    BALANCED_FILE,
    index=False
)


# ============================================================
# 16. SAVE SUMMARY TEXT FILE
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VERISIGHT STAGE 1 DATASET AUDIT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Original rows: {len(df)}\n"
    )

    f.write(
        f"Balanced candidate rows: "
        f"{len(balanced_df)}\n\n"
    )

    f.write(
        "LABEL DISTRIBUTION\n"
    )

    f.write(
        str(label_counts)
    )

    f.write(
        "\n\nWORD COUNT STATISTICS\n"
    )

    f.write(
        str(
            word_stats.round(2)
        )
    )

    f.write(
        "\n\nLENGTH DISTRIBUTION\n"
    )

    f.write(
        str(
            length_table.round(2)
        )
    )

    f.write(
        "\n\nWORD COUNT ONLY BASELINE\n"
    )

    f.write(
        f"Accuracy: "
        f"{word_accuracy * 100:.2f}%\n"
    )

    f.write(
        "\n\nDUPLICATES\n"
    )

    f.write(
        f"Duplicate rows: "
        f"{duplicate_count}\n"
    )

    f.write(
        f"Cross-label duplicates: "
        f"{len(cross_label_duplicates)}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 80)
print("STEP 1 COMPLETE")
print("=" * 80)

print(
    "\nFiles created:"
)

print(
    f"\n1. Audit dataset:"
)

print(
    AUDIT_FILE
)

print(
    f"\n2. Balanced candidate:"
)

print(
    BALANCED_FILE
)

print(
    f"\n3. Audit summary:"
)

print(
    SUMMARY_FILE
)

print(
    "\nIMPORTANT:"
)

print(
    "Do NOT train the new model yet."
)

print(
    "First send me the terminal output from this script."
)

print(
    "We will inspect the actual distributions and "
    "decide whether the candidate balancing strategy "
    "is appropriate before Step 2."
)

print(
    "\n" + "=" * 80
)