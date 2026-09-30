import numpy as np
import pandas as pd
import re
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT /
    "data" /
    "stage1" /
    "stage1_final_v3.csv"
)

MODEL_FILE = (
    PROJECT_ROOT /
    "models" /
    "stylometry_length_controlled_v3.keras"
)

SCALER_FILE = (
    PROJECT_ROOT /
    "features" /
    "stylometry_length_controlled_scaler_v3.pkl"
)


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def extract_features(text):

    text = str(text)

    words = re.findall(
        r"\b\w+\b",
        text
    )

    words_clean = [
        w for w in words
        if w.strip()
    ]

    sentences = re.split(
        r"[.!?]+",
        text
    )

    sentences_clean = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    word_count = len(words_clean)
    sentence_count = len(sentences_clean)
    character_count = len(text)

    word_lengths = [
        len(w)
        for w in words_clean
    ]

    avg_word_length = (
        np.mean(word_lengths)
        if word_lengths else 0
    )

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

    unique_words = set(
        w.lower()
        for w in words_clean
    )

    vocabulary_richness = (
        len(unique_words) / word_count
        if word_count > 0 else 0
    )

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
        if word_count > 0 else 0
    )

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

    digit_count = sum(
        c.isdigit()
        for c in text
    )

    digit_ratio = (
        digit_count / character_count
        if character_count > 0 else 0
    )

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

    stopword_count = sum(
        w.lower() in ENGLISH_STOP_WORDS
        for w in words_clean
    )

    stopword_ratio = (
        stopword_count / word_count
        if word_count > 0 else 0
    )

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

    first_person = sum(
        w.lower() in {
            "i", "me", "my",
            "mine", "we", "us",
            "our", "ours"
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

    avg_chars_per_word = (
        character_count / word_count
        if word_count > 0 else 0
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


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("VERISIGHT - LENGTH CONTROLLED STYLOMETRY")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nFull dataset:", df.shape)


# ==========================================================
# WORD COUNT
# ==========================================================

df["word_count_temp"] = (
    df["review"]
    .astype(str)
    .str.split()
    .str.len()
)


# ==========================================================
# SELECT 30-51 WORD REVIEWS
# ==========================================================

controlled = df[
    (df["word_count_temp"] >= 30) &
    (df["word_count_temp"] <= 51)
].copy()

print(
    "\n30-51 word subset:",
    controlled.shape
)

print("\nAvailable samples:")

print(
    controlled["label"]
    .value_counts()
    .sort_index()
)


# ==========================================================
# BALANCE HUMAN AND AI
# ==========================================================

human = controlled[
    controlled["label"] == 0
]

ai = controlled[
    controlled["label"] == 1
]

n = min(
    len(human),
    len(ai)
)

print(
    f"\nUsing {n} Human and {n} AI samples."
)

human = human.sample(
    n=n,
    random_state=42
)

ai = ai.sample(
    n=n,
    random_state=42
)

balanced = pd.concat(
    [human, ai],
    ignore_index=True
)

balanced = balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print(
    "\nBalanced controlled dataset:",
    balanced.shape
)

print(
    balanced["label"]
    .value_counts()
    .sort_index()
)


# ==========================================================
# EXTRACT FEATURES
# ==========================================================

print(
    "\nExtracting 24 stylometric features..."
)

X = np.array(
    [
        extract_features(review)
        for review in balanced["review"]
    ],
    dtype=np.float32
)

y = balanced["label"].astype(
    np.int32
).to_numpy()

print(
    "Feature shape:",
    X.shape
)


# ==========================================================
# SPLIT
# ==========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nSplit sizes:")

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ==========================================================
# SCALE
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_val = scaler.transform(
    X_val
)

X_test = scaler.transform(
    X_test
)


# ==========================================================
# MODEL
# ==========================================================

model = Sequential([
    Dense(
        128,
        activation="relu",
        input_shape=(24,)
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ==========================================================
# TRAIN
# ==========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)


model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=30,
    batch_size=64,
    callbacks=[
        early_stopping
    ],
    verbose=1
)


# ==========================================================
# TEST
# ==========================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print("\n" + "=" * 70)
print("CONTROLLED TEST RESULTS")
print("=" * 70)

print(
    f"\nTest Loss     : {loss:.4f}"
)

print(
    f"Test Accuracy : {accuracy:.2%}"
)


pred_prob = model.predict(
    X_test,
    verbose=0
).ravel()

pred = (
    pred_prob >= 0.5
).astype(int)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred,
        target_names=[
            "Human",
            "AI"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        pred
    )
)


# ==========================================================
# SAVE
# ==========================================================

model.save(
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)


print("\nModel saved to:")
print(MODEL_FILE)

print("\nScaler saved to:")
print(SCALER_FILE)


# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 70)
print("LENGTH CONTROLLED TRAINING COMPLETED")
print("=" * 70)