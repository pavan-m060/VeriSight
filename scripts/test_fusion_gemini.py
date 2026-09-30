# ==========================================================
# VeriSight
# Stage 1 - MiniLM + Stylometry Fusion
# Unseen Gemini External Test
# ==========================================================

import os
import re
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# Paths
# ==========================================================

GEMINI_FILE = "data/external/gemini_external_balanced.csv"

MODEL_FILE = "models/fusion_mlp_stage1_v2.keras"

SCALER_FILE = (
    "features/stylometry_scaler_fusion_stage1_v2.pkl"
)

# ==========================================================
# MiniLM Model
# ==========================================================

MINILM_MODEL = "all-MiniLM-L6-v2"

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
# MUST MATCH TRAINING
# ==========================================================

def extract_features(text):

    text = str(text).strip()

    characters = len(text)

    words = re.findall(r"\b[\w']+\b", text)

    word_count = len(words)

    sentences = re.split(r"[.!?]+", text)

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    sentence_count = len(sentences)

    if word_count == 0:
        return np.zeros(24, dtype=np.float32)

    if sentence_count == 0:
        sentence_count = 1

    word_lengths = [
        len(word.strip("'"))
        for word in words
        if word.strip("'")
    ]

    avg_word_length = np.mean(
        word_lengths
    )

    long_words = sum(
        1
        for length in word_lengths
        if length >= 7
    )

    short_words = sum(
        1
        for length in word_lengths
        if length <= 3
    )

    long_word_ratio = (
        long_words / word_count
    )

    short_word_ratio = (
        short_words / word_count
    )

    words_per_sentence = [
        len(
            re.findall(
                r"\b[\w']+\b",
                sentence
            )
        )
        for sentence in sentences
    ]

    avg_sentence_length = np.mean(
        words_per_sentence
    )

    if len(words_per_sentence) > 1:
        sentence_length_std = np.std(
            words_per_sentence
        )
    else:
        sentence_length_std = 0.0

    lower_words = [
        word.lower()
        for word in words
    ]

    unique_words = len(
        set(lower_words)
    )

    vocabulary_richness = (
        unique_words / word_count
    )

    repeated_words = (
        word_count - unique_words
    )

    repeated_word_ratio = (
        repeated_words / word_count
    )

    punctuation_count = len(
        re.findall(
            r"[^\w\s]",
            text
        )
    )

    punctuation_ratio = (
        punctuation_count /
        max(characters, 1)
    )

    comma_count = text.count(",")

    period_count = text.count(".")

    question_count = text.count("?")

    exclamation_count = text.count("!")

    digit_count = len(
        re.findall(
            r"\d",
            text
        )
    )

    digit_ratio = (
        digit_count /
        max(characters, 1)
    )

    letters = re.findall(
        r"[A-Za-z]",
        text
    )

    uppercase_letters = sum(
        1
        for char in letters
        if char.isupper()
    )

    uppercase_ratio = (
        uppercase_letters /
        max(len(letters), 1)
    )

    stopword_count = sum(
        1
        for word in lower_words
        if word in STOPWORDS
    )

    stopword_ratio = (
        stopword_count / word_count
    )

    first_person = {
        "i", "me", "my", "mine",
        "we", "us", "our", "ours"
    }

    second_person = {
        "you", "your", "yours"
    }

    third_person = {
        "he", "him", "his",
        "she", "her", "hers",
        "they", "them",
        "their", "theirs"
    }

    first_person_ratio = sum(
        1
        for word in lower_words
        if word in first_person
    ) / word_count

    second_person_ratio = sum(
        1
        for word in lower_words
        if word in second_person
    ) / word_count

    third_person_ratio = sum(
        1
        for word in lower_words
        if word in third_person
    ) / word_count

    avg_chars_per_word = (
        characters / word_count
    )

    return np.array([
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
    ], dtype=np.float32)


# ==========================================================
# Start
# ==========================================================

print("=" * 70)
print("VeriSight - MiniLM + Stylometry Fusion")
print("Gemini External Test")
print("=" * 70)

# ==========================================================
# Load Gemini Dataset
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
    df["generator_model"].value_counts()
)

print("\nSource distribution:")

print(
    df["source"].value_counts()
)

# ==========================================================
# Extract Reviews
# ==========================================================

reviews = (
    df["review"]
    .astype(str)
    .tolist()
)

# All samples are AI
y_gemini = np.ones(
    len(reviews),
    dtype=np.int32
)

# ==========================================================
# MiniLM Embeddings
# ==========================================================

print("\nLoading MiniLM...")

embedder = SentenceTransformer(
    MINILM_MODEL
)

print("\nGenerating MiniLM embeddings...")

X_semantic = embedder.encode(
    reviews,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

X_semantic = np.asarray(
    X_semantic,
    dtype=np.float32
)

print(
    "\nMiniLM shape:",
    X_semantic.shape
)

# ==========================================================
# Stylometric Features
# ==========================================================

print("\nGenerating stylometric features...")

X_stylo = np.array(
    [
        extract_features(review)
        for review in reviews
    ],
    dtype=np.float32
)

print(
    "Stylometry shape:",
    X_stylo.shape
)

# ==========================================================
# Load Fusion Scaler
# ==========================================================

print("\nLoading stylometric scaler...")

scaler = joblib.load(
    SCALER_FILE
)

X_stylo_scaled = scaler.transform(
    X_stylo
)

# ==========================================================
# Fuse Features
# ==========================================================

print("\nCreating fused features...")

X_fusion = np.concatenate(
    [
        X_semantic,
        X_stylo_scaled
    ],
    axis=1
)

print(
    "Fusion shape:",
    X_fusion.shape
)

# ==========================================================
# Load Fusion Model
# ==========================================================

print("\nLoading fusion MLP...")

model = tf.keras.models.load_model(
    MODEL_FILE
)

# ==========================================================
# Prediction
# ==========================================================

print("\nRunning Gemini predictions...")

probabilities = model.predict(
    X_fusion,
    verbose=0
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)

# ==========================================================
# Results
# ==========================================================

accuracy = accuracy_score(
    y_gemini,
    predictions
)

detected_ai = np.sum(
    predictions == 1
)

missed_ai = np.sum(
    predictions == 0
)

print("\n" + "=" * 70)
print("GEMINI FUSION TEST RESULTS")
print("=" * 70)

print(
    f"\nTotal Gemini Reviews : "
    f"{len(reviews)}"
)

print(
    f"AI Detected          : "
    f"{detected_ai} / {len(reviews)}"
)

print(
    f"AI Missed            : "
    f"{missed_ai} / {len(reviews)}"
)

print(
    f"\nGemini Detection Rate: "
    f"{accuracy * 100:.2f}%"
)

# ==========================================================
# Classification Report
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
        zero_division=0,
        digits=4
    )
)

# ==========================================================
# Confusion Matrix
# ==========================================================

print("Confusion Matrix:")

cm = confusion_matrix(
    y_gemini,
    predictions,
    labels=[0, 1]
)

print(cm)

# ==========================================================
# Prediction Distribution
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
# Probability Statistics
# ==========================================================

print("\nAI Probability Statistics:")

print(
    f"Minimum : "
    f"{probabilities.min():.4f}"
)

print(
    f"Maximum : "
    f"{probabilities.max():.4f}"
)

print(
    f"Mean    : "
    f"{probabilities.mean():.4f}"
)

print(
    f"Median  : "
    f"{np.median(probabilities):.4f}"
)

# ==========================================================
# Per-Source Results
# ==========================================================

print("\n" + "=" * 70)
print("PER-SOURCE GEMINI RESULTS")
print("=" * 70)

for source in sorted(
    df["source"].unique()
):

    mask = (
        df["source"].values ==
        source
    )

    source_total = np.sum(mask)

    source_detected = np.sum(
        predictions[mask] == 1
    )

    source_rate = (
        source_detected /
        source_total
    ) * 100

    print(
        f"{source:12s} : "
        f"{source_detected}/{source_total} "
        f"({source_rate:.2f}%)"
    )

# ==========================================================
# Final
# ==========================================================

print("\n" + "=" * 70)
print("Gemini fusion testing completed")
print("=" * 70)