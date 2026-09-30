import pandas as pd
import numpy as np
import joblib
import re

from pathlib import Path
from tensorflow.keras.models import load_model


# ============================================================
# VERISIGHT - LENGTH CONTROLLED GEMINI TEST
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

GEMINI_PATH = BASE_DIR / "data" / "external" / "gemini_external_balanced.csv"

MODEL_PATH = BASE_DIR / "models" / "stylometry_length_controlled_v3.keras"

SCALER_PATH = BASE_DIR / "features" / "stylometry_length_controlled_scaler_v3.pkl"

OUTPUT_FEATURES = BASE_DIR / "features" / "stylometry_length_controlled_gemini_v3.npy"

OUTPUT_PREDICTIONS = BASE_DIR / "features" / "stylometry_length_controlled_gemini_predictions_v3.csv"


# ============================================================
# STYLOMETRIC FEATURE EXTRACTION
# ============================================================

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except ImportError:
    ENGLISH_STOP_WORDS = set()


def extract_features(text):

    text = str(text)

    words = re.findall(r"\b\w+\b", text)
    sentences = re.split(r"[.!?]+", text)

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    word_count = len(words)
    sentence_count = len(sentences)
    character_count = len(text)

    if word_count > 0:
        avg_word_length = np.mean([
            len(w)
            for w in words
        ])
    else:
        avg_word_length = 0

    if sentence_count > 0:
        sentence_lengths = [
            len(re.findall(r"\b\w+\b", s))
            for s in sentences
        ]

        avg_sentence_length = np.mean(sentence_lengths)

        if len(sentence_lengths) > 1:
            sentence_length_std = np.std(sentence_lengths)
        else:
            sentence_length_std = 0

    else:
        avg_sentence_length = 0
        sentence_length_std = 0

    unique_words = set(
        w.lower()
        for w in words
    )

    if word_count > 0:
        vocabulary_richness = len(unique_words) / word_count
    else:
        vocabulary_richness = 0

    word_freq = {}

    for w in words:

        w = w.lower()

        word_freq[w] = word_freq.get(w, 0) + 1

    if word_count > 0:

        repeated_words = sum(
            count - 1
            for count in word_freq.values()
            if count > 1
        )

        repeated_word_ratio = repeated_words / word_count

    else:

        repeated_word_ratio = 0

    punctuation_count = len(
        re.findall(r"[^\w\s]", text)
    )

    if character_count > 0:
        punctuation_ratio = punctuation_count / character_count
    else:
        punctuation_ratio = 0

    comma_count = text.count(",")

    period_count = text.count(".")

    question_count = text.count("?")

    exclamation_count = text.count("!")

    digit_count = len(
        re.findall(r"\d", text)
    )

    if character_count > 0:
        digit_ratio = digit_count / character_count
    else:
        digit_ratio = 0

    uppercase_count = len(
        re.findall(r"[A-Z]", text)
    )

    if character_count > 0:
        uppercase_ratio = uppercase_count / character_count
    else:
        uppercase_ratio = 0

    stopword_count = sum(
        1
        for w in words
        if w.lower() in ENGLISH_STOP_WORDS
    )

    if word_count > 0:
        stopword_ratio = stopword_count / word_count
    else:
        stopword_ratio = 0

    long_word_count = sum(
        1
        for w in words
        if len(w) >= 7
    )

    short_word_count = sum(
        1
        for w in words
        if len(w) <= 3
    )

    if word_count > 0:

        long_word_ratio = long_word_count / word_count

        short_word_ratio = short_word_count / word_count

        first_person_ratio = sum(
            1
            for w in words
            if w.lower() in {
                "i", "me", "my", "mine", "we", "us", "our", "ours"
            }
        ) / word_count

        second_person_ratio = sum(
            1
            for w in words
            if w.lower() in {
                "you", "your", "yours"
            }
        ) / word_count

        third_person_ratio = sum(
            1
            for w in words
            if w.lower() in {
                "he", "she", "it", "they",
                "them", "his", "her", "their"
            }
        ) / word_count

    else:

        long_word_ratio = 0
        short_word_ratio = 0
        first_person_ratio = 0
        second_person_ratio = 0
        third_person_ratio = 0

    avg_chars_per_word = (
        character_count / word_count
        if word_count > 0
        else 0
    )

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


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("VERISIGHT - LENGTH CONTROLLED GEMINI TEST")
print("=" * 70)


# ------------------------------------------------------------
# Load frozen Gemini dataset
# ------------------------------------------------------------

df = pd.read_csv(GEMINI_PATH)

print("\nOriginal Gemini dataset:", df.shape)


# ------------------------------------------------------------
# Calculate word count
# ------------------------------------------------------------

df["word_count_temp"] = df["review"].astype(str).apply(
    lambda x: len(re.findall(r"\b\w+\b", x))
)


# ------------------------------------------------------------
# Select only 30-51 word Gemini reviews
# ------------------------------------------------------------

controlled = df[
    (df["word_count_temp"] >= 30) &
    (df["word_count_temp"] <= 51)
].copy()


print("\nLength-controlled Gemini subset:", controlled.shape)

print("\nWord-count distribution:")
print(
    controlled["word_count_temp"].describe()
)


print("\nSource distribution:")
print(
    controlled["source"].value_counts()
)


# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

if len(controlled) != 110:

    print(
        f"\nWARNING: Expected 110 reviews, "
        f"but found {len(controlled)}."
    )


# ------------------------------------------------------------
# Extract 24 stylometric features
# ------------------------------------------------------------

print("\nExtracting 24 stylometric features...")

X = np.array(
    [
        extract_features(review)
        for review in controlled["review"]
    ],
    dtype=np.float32
)


print("Feature shape:", X.shape)


# ------------------------------------------------------------
# Save features
# ------------------------------------------------------------

np.save(
    OUTPUT_FEATURES,
    X
)


# ------------------------------------------------------------
# Load scaler
# ------------------------------------------------------------

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_PATH
)


X_scaled = scaler.transform(X)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("Loading length-controlled model...")

model = load_model(
    MODEL_PATH
)


# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------

probabilities = model.predict(
    X_scaled,
    verbose=0
).ravel()


predictions = (
    probabilities >= 0.5
).astype(int)


# ------------------------------------------------------------
# AI detection rate
# ------------------------------------------------------------

total = len(predictions)

ai_detected = int(
    np.sum(predictions == 1)
)

ai_missed = int(
    np.sum(predictions == 0)
)

detection_rate = (
    ai_detected / total * 100
    if total > 0
    else 0
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LENGTH CONTROLLED GEMINI RESULTS")
print("=" * 70)

print(f"\nTotal Gemini reviews : {total}")

print(
    f"AI detected          : "
    f"{ai_detected}/{total}"
)

print(
    f"AI missed            : "
    f"{ai_missed}/{total}"
)

print(
    f"\nAI Detection Rate    : "
    f"{detection_rate:.2f}%"
)


# ------------------------------------------------------------
# Probability statistics
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Source-wise results
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("SOURCE-WISE RESULTS")
print("-" * 70)

for source, group in controlled.groupby("source"):

    idx = group.index

    source_predictions = predictions[
        controlled.index.get_indexer(idx)
    ]

    detected = int(
        np.sum(source_predictions == 1)
    )

    total_source = len(source_predictions)

    rate = (
        detected / total_source * 100
        if total_source > 0
        else 0
    )

    print(
        f"{source:15s}: "
        f"{detected}/{total_source} "
        f"({rate:.2f}%)"
    )


# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------

controlled["ai_probability"] = probabilities

controlled["prediction"] = predictions

controlled["prediction_label"] = controlled[
    "prediction"
].map({
    0: "HUMAN",
    1: "AI"
})


controlled.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\nPredictions saved to:")

print(
    OUTPUT_PREDICTIONS
)

print("\nFeatures saved to:")

print(
    OUTPUT_FEATURES
)

print("\n" + "=" * 70)
print("LENGTH CONTROLLED GEMINI TEST COMPLETED")
print("=" * 70)