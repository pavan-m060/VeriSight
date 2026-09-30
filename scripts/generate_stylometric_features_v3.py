import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "stage1" / "stage1_final_v3.csv"

OUTPUT_X = PROJECT_ROOT / "features" / "stylometry_stage1_v3.npy"
OUTPUT_Y = PROJECT_ROOT / "features" / "y_stylometry_stage1_v3.npy"

OUTPUT_X.parent.mkdir(parents=True, exist_ok=True)


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(text):

    text = str(text)

    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)

    words_clean = [
        w for w in words
        if w.strip()
    ]

    sentences_clean = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    word_count = len(words_clean)
    sentence_count = len(sentences_clean)
    character_count = len(text)

    # ------------------------------------------------------
    # Word lengths
    # ------------------------------------------------------

    word_lengths = [
        len(w)
        for w in words_clean
    ]

    avg_word_length = (
        np.mean(word_lengths)
        if word_lengths else 0
    )

    # ------------------------------------------------------
    # Sentence lengths
    # ------------------------------------------------------

    sentence_lengths = [
        len(re.findall(r"\b\w+\b", s))
        for s in sentences_clean
    ]

    avg_sentence_length = (
        np.mean(sentence_lengths)
        if sentence_lengths else 0
    )

    sentence_length_std = (
        np.std(sentence_lengths)
        if sentence_lengths else 0
    )

    # ------------------------------------------------------
    # Vocabulary richness
    # ------------------------------------------------------

    unique_words = set(
        w.lower()
        for w in words_clean
    )

    vocabulary_richness = (
        len(unique_words) / word_count
        if word_count > 0 else 0
    )

    # ------------------------------------------------------
    # Repeated word ratio
    # ------------------------------------------------------

    word_lower = [
        w.lower()
        for w in words_clean
    ]

    repeated_count = (
        len(word_lower) -
        len(set(word_lower))
    )

    repeated_word_ratio = (
        repeated_count / word_count
        if word_count > 0 else 0
    )

    # ------------------------------------------------------
    # Punctuation
    # ------------------------------------------------------

    punctuation_chars = re.findall(
        r"[^\w\s]",
        text
    )

    punctuation_count = len(
        punctuation_chars
    )

    punctuation_ratio = (
        punctuation_count / character_count
        if character_count > 0 else 0
    )

    comma_count = text.count(",")

    period_count = text.count(".")

    question_count = text.count("?")

    exclamation_count = text.count("!")

    # ------------------------------------------------------
    # Digits
    # ------------------------------------------------------

    digit_count = sum(
        c.isdigit()
        for c in text
    )

    digit_ratio = (
        digit_count / character_count
        if character_count > 0 else 0
    )

    # ------------------------------------------------------
    # Uppercase
    # ------------------------------------------------------

    uppercase_letters = sum(
        c.isupper()
        for c in text
    )

    alphabetic_letters = sum(
        c.isalpha()
        for c in text
    )

    uppercase_ratio = (
        uppercase_letters / alphabetic_letters
        if alphabetic_letters > 0 else 0
    )

    # ------------------------------------------------------
    # Stopwords
    # ------------------------------------------------------

    stopword_count = sum(
        w.lower() in ENGLISH_STOP_WORDS
        for w in words_clean
    )

    stopword_ratio = (
        stopword_count / word_count
        if word_count > 0 else 0
    )

    # ------------------------------------------------------
    # Long / short words
    # ------------------------------------------------------

    long_words = sum(
        len(w) >= 8
        for w in words_clean
    )

    short_words = sum(
        len(w) <= 3
        for w in words_clean
    )

    long_word_ratio = (
        long_words / word_count
        if word_count > 0 else 0
    )

    short_word_ratio = (
        short_words / word_count
        if word_count > 0 else 0
    )

    # ------------------------------------------------------
    # Pronouns
    # ------------------------------------------------------

    first_person = sum(
        w.lower() in {
            "i", "me", "my",
            "mine", "we",
            "us", "our", "ours"
        }
        for w in words_clean
    )

    second_person = sum(
        w.lower() in {
            "you", "your",
            "yours", "yourself",
            "yourselves"
        }
        for w in words_clean
    )

    third_person = sum(
        w.lower() in {
            "he", "him", "his",
            "she", "her", "hers",
            "they", "them",
            "their", "theirs",
            "it", "its"
        }
        for w in words_clean
    )

    first_person_ratio = (
        first_person / word_count
        if word_count > 0 else 0
    )

    second_person_ratio = (
        second_person / word_count
        if word_count > 0 else 0
    )

    third_person_ratio = (
        third_person / word_count
        if word_count > 0 else 0
    )

    # ------------------------------------------------------
    # Average characters per word
    # ------------------------------------------------------

    avg_chars_per_word = (
        character_count / word_count
        if word_count > 0 else 0
    )

    # ------------------------------------------------------
    # 24 FEATURES
    # ------------------------------------------------------

    return [
        word_count,
        sentence_count,
        character_count,
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


# ==========================================================
# MAIN
# ==========================================================

print("=" * 70)
print("VERISIGHT - STAGE 1 V3 STYLOMETRIC FEATURES")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)

reviews = df["review"].astype(str).tolist()

labels = (
    df["label"]
    .astype(int)
    .to_numpy()
)

print("\nExtracting 24 stylometric features...")

features = [
    extract_features(review)
    for review in reviews
]

X = np.asarray(
    features,
    dtype=np.float32
)

print("\nFeature matrix shape:", X.shape)
print("Label shape:", labels.shape)

np.save(
    OUTPUT_X,
    X
)

np.save(
    OUTPUT_Y,
    labels
)

print("\nSaved:")
print(OUTPUT_X)
print(OUTPUT_Y)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)