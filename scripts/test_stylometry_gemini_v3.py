# ==========================================================
# VeriSight
# test_stylometry_gemini_v3.py
#
# Test corrected v3 Stylometry MLP on the frozen
# 320-review Gemini external dataset.
# ==========================================================

import numpy as np
import pandas as pd
import re
import joblib

from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import tensorflow as tf


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==========================================================
# FILE PATHS
# ==========================================================

GEMINI_FILE = (
    PROJECT_ROOT /
    "data" /
    "external" /
    "gemini_external_balanced.csv"
)

MODEL_FILE = (
    PROJECT_ROOT /
    "models" /
    "stylometry_mlp_stage1_v3.keras"
)

SCALER_FILE = (
    PROJECT_ROOT /
    "features" /
    "stylometry_scaler_stage1_v3.pkl"
)

GEMINI_FEATURE_FILE = (
    PROJECT_ROOT /
    "features" /
    "stylometry_gemini_v3.npy"
)


# ==========================================================
# STYLOMETRIC FEATURE EXTRACTION
# ==========================================================

def extract_features(text):

    text = str(text)

    # ------------------------------------------------------
    # Words
    # ------------------------------------------------------

    words = re.findall(
        r"\b\w+\b",
        text
    )

    words_clean = [
        w for w in words
        if w.strip()
    ]

    # ------------------------------------------------------
    # Sentences
    # ------------------------------------------------------

    sentences = re.split(
        r"[.!?]+",
        text
    )

    sentences_clean = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    # ------------------------------------------------------
    # Basic counts
    # ------------------------------------------------------

    word_count = len(words_clean)

    sentence_count = len(
        sentences_clean
    )

    character_count = len(text)

    # ------------------------------------------------------
    # Word length
    # ------------------------------------------------------

    word_lengths = [
        len(w)
        for w in words_clean
    ]

    avg_word_length = (
        np.mean(word_lengths)
        if word_lengths
        else 0
    )

    # ------------------------------------------------------
    # Sentence length
    # ------------------------------------------------------

    sentence_lengths = [
        len(
            re.findall(
                r"\b\w+\b",
                sentence
            )
        )
        for sentence in sentences_clean
    ]

    avg_sentence_length = (
        np.mean(sentence_lengths)
        if sentence_lengths
        else 0
    )

    sentence_length_std = (
        np.std(sentence_lengths)
        if sentence_lengths
        else 0
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
        if word_count > 0
        else 0
    )

    # ------------------------------------------------------
    # Repeated word ratio
    # ------------------------------------------------------

    lowercase_words = [
        w.lower()
        for w in words_clean
    ]

    repeated_count = (
        len(lowercase_words)
        -
        len(set(lowercase_words))
    )

    repeated_word_ratio = (
        repeated_count / word_count
        if word_count > 0
        else 0
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
        if character_count > 0
        else 0
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
        if character_count > 0
        else 0
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
        if alphabetic_letters > 0
        else 0
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
        if word_count > 0
        else 0
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
        if word_count > 0
        else 0
    )

    short_word_ratio = (
        short_words / word_count
        if word_count > 0
        else 0
    )

    # ------------------------------------------------------
    # First person
    # ------------------------------------------------------

    first_person = sum(
        w.lower() in {
            "i",
            "me",
            "my",
            "mine",
            "we",
            "us",
            "our",
            "ours"
        }
        for w in words_clean
    )

    # ------------------------------------------------------
    # Second person
    # ------------------------------------------------------

    second_person = sum(
        w.lower() in {
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves"
        }
        for w in words_clean
    )

    # ------------------------------------------------------
    # Third person
    # ------------------------------------------------------

    third_person = sum(
        w.lower() in {
            "he",
            "him",
            "his",
            "she",
            "her",
            "hers",
            "they",
            "them",
            "their",
            "theirs",
            "it",
            "its"
        }
        for w in words_clean
    )

    first_person_ratio = (
        first_person / word_count
        if word_count > 0
        else 0
    )

    second_person_ratio = (
        second_person / word_count
        if word_count > 0
        else 0
    )

    third_person_ratio = (
        third_person / word_count
        if word_count > 0
        else 0
    )

    # ------------------------------------------------------
    # Average characters per word
    # ------------------------------------------------------

    avg_chars_per_word = (
        character_count / word_count
        if word_count > 0
        else 0
    )

    # ======================================================
    # EXACT 24 FEATURES
    # ======================================================

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
print("VeriSight - Stylometry MLP - Gemini External Test V3")
print("=" * 70)


# ==========================================================
# LOAD GEMINI DATASET
# ==========================================================

print("\nLoading Gemini dataset...")

df = pd.read_csv(
    GEMINI_FILE
)

print(
    "Dataset shape:",
    df.shape
)

print("\nGenerator distribution:")

print(
    df["generator_model"]
    .value_counts()
)

print("\nSource distribution:")

print(
    df["source"]
    .value_counts()
)


# ==========================================================
# GENERATE STYLOMETRIC FEATURES
# ==========================================================

print("\nGenerating stylometric features...")

X_gemini = np.array(
    [
        extract_features(review)
        for review in df["review"]
    ],
    dtype=np.float32
)

# All Gemini reviews are AI
y_gemini = np.ones(
    len(df),
    dtype=np.int32
)

print(
    "Feature shape:",
    X_gemini.shape
)


# ==========================================================
# SAVE GEMINI FEATURES
# ==========================================================

np.save(
    GEMINI_FEATURE_FILE,
    X_gemini
)

print("\nGemini features saved to:")

print(
    GEMINI_FEATURE_FILE
)


# ==========================================================
# LOAD TRAINING SCALER
# ==========================================================

print("\nLoading training scaler...")

scaler = joblib.load(
    SCALER_FILE
)

X_gemini_scaled = scaler.transform(
    X_gemini
)


# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading stylometry MLP...")

model = tf.keras.models.load_model(
    MODEL_FILE
)


# ==========================================================
# PREDICTION
# ==========================================================

print("\nRunning Gemini predictions...")

probabilities = (
    model.predict(
        X_gemini_scaled,
        verbose=0
    )
    .ravel()
)

# Same binary threshold used during training
predictions = (
    probabilities >= 0.5
).astype(int)


# ==========================================================
# RESULTS
# ==========================================================

detected = np.sum(
    predictions == 1
)

missed = np.sum(
    predictions == 0
)

total = len(
    predictions
)

detection_rate = (
    detected / total
    if total > 0
    else 0
)


print("\n" + "=" * 70)
print("GEMINI EXTERNAL TEST RESULTS V3")
print("=" * 70)

print(
    f"\nTotal Gemini Reviews : {total}"
)

print(
    f"AI Detected          : {detected} / {total}"
)

print(
    f"AI Missed            : {missed} / {total}"
)

print(
    f"\nGemini Detection Rate: "
    f"{detection_rate:.2%}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_gemini,
        predictions,
        labels=[0, 1],
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("Confusion Matrix:")

cm = confusion_matrix(
    y_gemini,
    predictions,
    labels=[0, 1]
)

print(cm)


# ==========================================================
# PREDICTION DISTRIBUTION
# ==========================================================

print("\nPrediction Distribution:")

print(
    "Predicted Human:",
    np.sum(predictions == 0)
)

print(
    "Predicted AI   :",
    np.sum(predictions == 1)
)


# ==========================================================
# PROBABILITY STATISTICS
# ==========================================================

print("\nAI Probability Statistics:")

print(
    f"Minimum : {probabilities.min():.4f}"
)

print(
    f"Maximum : {probabilities.max():.4f}"
)

print(
    f"Mean    : {probabilities.mean():.4f}"
)

print(
    f"Median  : {np.median(probabilities):.4f}"
)


# ==========================================================
# SOURCE-WISE RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("SOURCE-WISE GEMINI RESULTS")
print("=" * 70)

for source in sorted(
    df["source"].unique()
):

    mask = (
        df["source"] == source
    )

    source_predictions = predictions[
        mask
    ]

    source_detected = np.sum(
        source_predictions == 1
    )

    source_total = len(
        source_predictions
    )

    source_rate = (
        source_detected /
        source_total
    )

    print(
        f"{source:15s} : "
        f"{source_detected:3d}/{source_total} "
        f"({source_rate:.2%})"
    )


# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

results = df.copy()

results["ai_probability"] = (
    probabilities
)

results["predicted_label"] = (
    predictions
)

results["predicted_class"] = (
    np.where(
        predictions == 1,
        "AI",
        "Human"
    )
)

RESULT_FILE = (
    PROJECT_ROOT /
    "features" /
    "stylometry_gemini_v3_predictions.csv"
)

results.to_csv(
    RESULT_FILE,
    index=False
)

print("\nPrediction results saved to:")

print(
    RESULT_FILE
)


# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 70)
print("Gemini external testing completed")
print("=" * 70)