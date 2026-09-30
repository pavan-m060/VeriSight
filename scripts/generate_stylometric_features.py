# ==========================================================
# VeriSight
# Stage 1 - Generate Stylometric Features
# ==========================================================

import os
import re
import numpy as np
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

DATA_FILE = "data/stage1/stage1_final_v2.csv"

OUTPUT_DIR = "features"
X_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stylometry_stage1_v2.npy"
)

Y_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "y_stylometry_stage1_v2.npy"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 70)
print("VeriSight - Stylometric Feature Generation")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset Shape :", df.shape)

print("\nLabel Distribution:")
print(df["label"].value_counts())

# ==========================================================
# Stopwords
# ==========================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but",
    "if", "then", "else", "for", "of",
    "in", "on", "at", "to", "from",
    "by", "with", "about", "as",
    "is", "am", "are", "was", "were",
    "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "this", "that", "these", "those",
    "it", "its",
    "i", "me", "my", "mine",
    "we", "us", "our", "ours",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "they", "them", "their", "theirs"
}

# ==========================================================
# Stylometric Feature Extraction
# ==========================================================

def extract_features(text):

    text = str(text).strip()

    # ------------------------------------------------------
    # Basic text statistics
    # ------------------------------------------------------

    characters = len(text)

    words = re.findall(r"\b[\w']+\b", text)
    word_count = len(words)

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Avoid division by zero
    if word_count == 0:
        return np.zeros(24, dtype=np.float32)

    if sentence_count == 0:
        sentence_count = 1

    # ------------------------------------------------------
    # Word statistics
    # ------------------------------------------------------

    word_lengths = [
        len(word.strip("'"))
        for word in words
        if word.strip("'")
    ]

    avg_word_length = np.mean(word_lengths)

    long_words = sum(
        1 for length in word_lengths
        if length >= 7
    )

    short_words = sum(
        1 for length in word_lengths
        if length <= 3
    )

    long_word_ratio = long_words / word_count
    short_word_ratio = short_words / word_count

    # ------------------------------------------------------
    # Sentence statistics
    # ------------------------------------------------------

    words_per_sentence = [
        len(re.findall(r"\b[\w']+\b", sentence))
        for sentence in sentences
    ]

    avg_sentence_length = np.mean(words_per_sentence)

    if len(words_per_sentence) > 1:
        sentence_length_std = np.std(words_per_sentence)
    else:
        sentence_length_std = 0.0

    # ------------------------------------------------------
    # Vocabulary richness
    # ------------------------------------------------------

    lower_words = [
        word.lower()
        for word in words
    ]

    unique_words = len(set(lower_words))

    vocabulary_richness = unique_words / word_count

    # ------------------------------------------------------
    # Repetition
    # ------------------------------------------------------

    repeated_words = word_count - unique_words

    repeated_word_ratio = repeated_words / word_count

    # ------------------------------------------------------
    # Punctuation
    # ------------------------------------------------------

    punctuation_count = len(
        re.findall(r"[^\w\s]", text)
    )

    punctuation_ratio = punctuation_count / max(
        characters, 1
    )

    comma_count = text.count(",")

    period_count = text.count(".")

    question_count = text.count("?")

    exclamation_count = text.count("!")

    # ------------------------------------------------------
    # Digits
    # ------------------------------------------------------

    digit_count = len(
        re.findall(r"\d", text)
    )

    digit_ratio = digit_count / max(
        characters, 1
    )

    # ------------------------------------------------------
    # Uppercase letters
    # ------------------------------------------------------

    letters = re.findall(r"[A-Za-z]", text)

    uppercase_letters = sum(
        1 for char in letters
        if char.isupper()
    )

    uppercase_ratio = uppercase_letters / max(
        len(letters), 1
    )

    # ------------------------------------------------------
    # Stopwords
    # ------------------------------------------------------

    stopword_count = sum(
        1 for word in lower_words
        if word in STOPWORDS
    )

    stopword_ratio = stopword_count / word_count

    # ------------------------------------------------------
    # Personal pronouns
    # ------------------------------------------------------

    first_person = {
        "i", "me", "my", "mine", "we", "us", "our", "ours"
    }

    second_person = {
        "you", "your", "yours"
    }

    third_person = {
        "he", "him", "his",
        "she", "her", "hers",
        "they", "them", "their", "theirs"
    }

    first_person_ratio = sum(
        1 for word in lower_words
        if word in first_person
    ) / word_count

    second_person_ratio = sum(
        1 for word in lower_words
        if word in second_person
    ) / word_count

    third_person_ratio = sum(
        1 for word in lower_words
        if word in third_person
    ) / word_count

    # ------------------------------------------------------
    # Character / word ratio
    # ------------------------------------------------------

    avg_chars_per_word = characters / word_count

    # ------------------------------------------------------
    # Return features
    # ------------------------------------------------------

    features = [
        word_count,
        sentence_count,
        characters,
        avg_word_length,
        avg_sentence_length,
        sentence_length_std,
        vocabulary_richness,
        repeated_word_ratio,
        punctuation_count,
        punctuation_ratio,
        comma_count,
        period_count,
        question_count,
        exclamation_count,
        digit_count,
        digit_ratio,
        uppercase_ratio,
        stopword_ratio,
        long_word_ratio,
        short_word_ratio,
        first_person_ratio,
        second_person_ratio,
        third_person_ratio,
        avg_chars_per_word
    ]

    return np.array(
        features,
        dtype=np.float32
    )


# ==========================================================
# Feature Names
# ==========================================================

FEATURE_NAMES = [
    "word_count",
    "sentence_count",
    "character_count",
    "avg_word_length",
    "avg_sentence_length",
    "sentence_length_std",
    "vocabulary_richness",
    "repeated_word_ratio",
    "punctuation_count",
    "punctuation_ratio",
    "comma_count",
    "period_count",
    "question_count",
    "exclamation_count",
    "digit_count",
    "digit_ratio",
    "uppercase_ratio",
    "stopword_ratio",
    "long_word_ratio",
    "short_word_ratio",
    "first_person_ratio",
    "second_person_ratio",
    "third_person_ratio",
    "avg_chars_per_word"
]

# ==========================================================
# Generate Features
# ==========================================================

print("\nGenerating stylometric features...")

features = []

for i, review in enumerate(df["review"]):

    features.append(
        extract_features(review)
    )

    if (i + 1) % 5000 == 0:
        print(
            f"Processed {i + 1:,} / {len(df):,}"
        )

X = np.array(features, dtype=np.float32)

y = df["label"].astype(int).values

# ==========================================================
# Save
# ==========================================================

np.save(X_OUTPUT, X)
np.save(Y_OUTPUT, y)

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 70)
print("Stylometric Feature Generation Completed")
print("=" * 70)

print("\nFeature Matrix Shape :", X.shape)
print("Label Shape          :", y.shape)

print("\nNumber of Features   :", X.shape[1])

print("\nFeature Names:")

for i, name in enumerate(FEATURE_NAMES):
    print(f"{i + 1:2d}. {name}")

print("\nSaved Files:")

print(f"  {X_OUTPUT}")
print(f"  {Y_OUTPUT}")

print("\nFirst Review:")
print(df["review"].iloc[0])

print("\nFirst Feature Vector:")
print(X[0])

print("\n" + "=" * 70)