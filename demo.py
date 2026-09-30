# ==========================================================
# VeriSight - Live AI Review Detection Demo
# ==========================================================

import os
import numpy as np
import joblib

from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

from features.stylometric_features import extract_stylometric_features


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mlp_length_balanced.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "stylometric_scaler.pkl"
)


# ==========================================================
# FEATURE ORDER
# IMPORTANT: Must match training order
# ==========================================================

STYLE_FEATURES = [
    "word_count",
    "character_count",
    "sentence_count",
    "avg_sentence_length",
    "avg_word_length",
    "vocabulary_diversity",
    "type_token_ratio",
    "punctuation_ratio",
    "comma_ratio",
    "exclamation_ratio",
    "question_ratio",
    "capitalization_ratio",
    "digit_ratio",
    "stopword_ratio",
    "repeated_word_ratio",
    "unique_word_ratio",
    "sentence_length_std",
    "paragraph_count"
]


# ==========================================================
# HEADER
# ==========================================================

print("=" * 65)
print("             VeriSight - Live Demo")
print("       AI-Generated Review Detection")
print("=" * 65)


# ==========================================================
# LOAD MINILM
# ==========================================================

print("\nLoading MiniLM...")

encoder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("✓ MiniLM loaded")


# ==========================================================
# LOAD ANN MODEL
# ==========================================================

print("\nLoading ANN model...")

model = load_model(
    MODEL_PATH
)

print("✓ ANN model loaded")


# ==========================================================
# LOAD SCALER
# ==========================================================

print("\nLoading stylometric scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print("✓ Scaler loaded")


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_review(review):

    # ------------------------------------------------------
    # STEP 1 - MiniLM
    # ------------------------------------------------------

    embedding = encoder.encode(
        [review],
        convert_to_numpy=True
    )

    embedding = embedding.astype(
        np.float32
    )


    # ------------------------------------------------------
    # STEP 2 - Stylometric Features
    # ------------------------------------------------------

    style_dict = extract_stylometric_features(
        review
    )

    # Convert dictionary → ordered feature vector
    style = np.array(
        [
            style_dict[name]
            for name in STYLE_FEATURES
        ],
        dtype=np.float32
    ).reshape(1, -1)


    # ------------------------------------------------------
    # Verify 18 features
    # ------------------------------------------------------

    if style.shape[1] != 18:

        raise ValueError(
            f"Expected 18 stylometric features, "
            f"got {style.shape[1]}"
        )


    # ------------------------------------------------------
    # STEP 3 - Apply Training Scaler
    # ------------------------------------------------------

    style_scaled = scaler.transform(
        style
    )


    # ------------------------------------------------------
    # STEP 4 - Feature Fusion
    # ------------------------------------------------------

    fused = np.concatenate(
        [
            embedding,
            style_scaled
        ],
        axis=1
    )


    # ------------------------------------------------------
    # Verify 402 features
    # ------------------------------------------------------

    if fused.shape[1] != 402:

        raise ValueError(
            f"Expected 402 features, "
            f"got {fused.shape[1]}"
        )


    # ------------------------------------------------------
    # STEP 5 - ANN Prediction
    # ------------------------------------------------------

    probability = model.predict(
        fused,
        verbose=0
    )[0][0]


    # ------------------------------------------------------
    # STEP 6 - Classification
    # ------------------------------------------------------

    if probability >= 0.5:

        prediction = "AI GENERATED"

    else:

        prediction = "HUMAN WRITTEN"


    return prediction, probability


# ==========================================================
# READY
# ==========================================================

print("\n" + "=" * 65)
print("Model Ready!")
print("=" * 65)

print(
    "\nEnter a review below."
)

print(
    "Type 'exit' to close the demo."
)


# ==========================================================
# LIVE LOOP
# ==========================================================

while True:

    print("\n" + "-" * 65)

    review = input(
        "Enter Review: "
    ).strip()


    # ------------------------------------------------------
    # Exit
    # ------------------------------------------------------

    if review.lower() == "exit":

        print(
            "\nExiting VeriSight..."
        )

        break


    # ------------------------------------------------------
    # Empty input
    # ------------------------------------------------------

    if len(review) == 0:

        print(
            "Please enter a review."
        )

        continue


    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    try:

        prediction, probability = (
            predict_review(review)
        )

    except Exception as e:

        print(
            "\nPrediction Error:"
        )

        print(e)

        continue


    # ------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------

    ai_percentage = (
        probability * 100
    )

    human_percentage = (
        100 - ai_percentage
    )


    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "                 VERISIGHT RESULT"
    )

    print("=" * 65)

    print(
        f"\nPrediction: {prediction}"
    )

    print(
        f"AI Probability: "
        f"{ai_percentage:.2f}%"
    )

    print(
        f"Human Probability: "
        f"{human_percentage:.2f}%"
    )

    print(
        "\nFeatures Used:"
    )

    print(
        "MiniLM       : 384 features"
    )

    print(
        "Stylometry   : 18 features"
    )

    print(
        "Total        : 402 features"
    )

    print("=" * 65)