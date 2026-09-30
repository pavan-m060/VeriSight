import os
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGE1_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "mlp_stage1_v2.keras"
)

STAGE2_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "stage2",
    "stage2_hybrid_stronger.keras"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "stage2",
    "stage2_hybrid_stronger_scaler.pkl"
)

STAGE2_TEST_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "phase2",
    "processed_hybrid",
    "stage2_test_hybrid.csv"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

STAGE1_THRESHOLD = 0.50
STAGE2_THRESHOLD = 0.62


# Exactly the 40 behavioral features used during Stage 2
BEHAVIOR_FEATURES = [
    "rating",
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
    "user_review_count",
    "user_avg_rating",
    "user_unique_products",
    "product_review_count",
    "product_avg_rating",
    "exact_text_count",
    "rating_sentiment_difference",
    "user_previous_review_count",
    "user_previous_avg_rating",
    "user_previous_rating_std",
    "user_previous_unique_products",
    "time_since_previous_user_review_hours",
    "user_days_since_first_review",
    "user_previous_reviews_per_day",
    "reviews_previous_24h",
    "reviews_previous_7days",
    "product_previous_review_count",
    "product_previous_avg_rating",
    "time_since_previous_product_review_hours",
    "product_reviews_previous_1h",
    "product_reviews_previous_24h",
    "product_reviews_previous_7days",
    "rating_distance_from_user_history",
    "rating_distance_from_product_history",
]

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading VeriSight models...")

stage1_model = load_model(STAGE1_MODEL_PATH)
stage2_model = load_model(STAGE2_MODEL_PATH)

stage2_scaler = joblib.load(SCALER_PATH)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

stage2_test_df = pd.read_csv(STAGE2_TEST_PATH)

print("Stage 1 model loaded.")
print("Stage 2 model loaded.")
print("Stage 2 scaler loaded.")
print("MiniLM loaded.")
print(f"Stage 2 test rows: {len(stage2_test_df)}")


# ============================================================
# TEXT EMBEDDING
# ============================================================

def generate_embedding(text: str):
    """
    Convert review text into a 384-dimensional MiniLM embedding.
    """

    embedding = embedding_model.encode(
        [text],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    return embedding.astype(np.float32)


# ============================================================
# STAGE 1
# ============================================================

def predict_ai_probability(text: str):

    embedding = generate_embedding(text)

    prediction = stage1_model.predict(
        embedding,
        verbose=0
    )

    ai_probability = float(np.asarray(prediction).reshape(-1)[0])

    # Stage 1 label:
    # 0 = Human
    # 1 = AI

    human_probability = 1.0 - ai_probability

    return ai_probability, human_probability


# ============================================================
# STAGE 2
# ============================================================

def get_behavior_features(row_index: int):

    if row_index < 0 or row_index >= len(stage2_test_df):
        raise ValueError(
            f"test_row_index must be between 0 and "
            f"{len(stage2_test_df) - 1}"
        )

    row = stage2_test_df.iloc[row_index]

    values = []

    for feature in BEHAVIOR_FEATURES:

        if feature not in stage2_test_df.columns:
            raise ValueError(
                f"Missing Stage 2 feature: {feature}"
            )

        value = row[feature]

        if pd.isna(value):
            value = 0.0

        values.append(float(value))

    return np.asarray(values, dtype=np.float32)


def predict_spam_probability(
    text: str,
    row_index: int
):

    embedding = generate_embedding(text)

    behavior = get_behavior_features(row_index)

    behavior_scaled = stage2_scaler.transform(
        behavior.reshape(1, -1)
    ).astype(np.float32)

    prediction = stage2_model.predict(
        {
            "minilm_embedding": embedding,
            "behavior_features": behavior_scaled
        },
        verbose=0
    )

    spam_probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    genuine_probability = 1.0 - spam_probability

    return spam_probability, genuine_probability


# ============================================================
# RATING / TEXT CONSISTENCY
# ============================================================

def rating_text_consistency(text: str, rating):

    if rating is None:
        return "Not available"

    text_lower = text.lower()

    positive_words = [
        "excellent",
        "amazing",
        "great",
        "good",
        "wonderful",
        "fantastic",
        "love",
        "perfect",
        "best"
    ]

    negative_words = [
        "bad",
        "terrible",
        "awful",
        "horrible",
        "poor",
        "worst",
        "hate",
        "disappointing"
    ]

    positive_score = sum(
        word in text_lower
        for word in positive_words
    )

    negative_score = sum(
        word in text_lower
        for word in negative_words
    )

    if rating >= 4 and negative_score > positive_score:
        return "Potentially inconsistent"

    if rating <= 2 and positive_score > negative_score:
        return "Potentially inconsistent"

    return "Consistent"


# ============================================================
# FINAL CLASSIFICATION
# ============================================================

def classify_review(
    ai_probability,
    spam_probability
):

    is_ai = ai_probability >= STAGE1_THRESHOLD
    is_spam = spam_probability >= STAGE2_THRESHOLD

    if is_ai and is_spam:
        return "AI_SPAM"

    if is_ai and not is_spam:
        return "AI_GENUINE"

    if not is_ai and is_spam:
        return "HUMAN_SPAM"

    return "HUMAN_GENUINE"


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_risk(
    ai_probability,
    spam_probability,
    classification
):

    # Strongest condition
    if classification == "AI_SPAM":
        return "HIGH"

    if spam_probability >= STAGE2_THRESHOLD:
        return "HIGH"

    if spam_probability >= 0.40:
        return "MEDIUM"

    if ai_probability >= 0.60:
        return "MEDIUM"

    return "LOW"


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_review(
    text: str,
    rating=None,
    row_index=0
):

    ai_probability, human_probability = \
        predict_ai_probability(text)

    spam_probability, genuine_probability = \
        predict_spam_probability(
            text,
            row_index
        )

    classification = classify_review(
        ai_probability,
        spam_probability
    )

    risk_level = calculate_risk(
        ai_probability,
        spam_probability,
        classification
    )

    consistency = rating_text_consistency(
        text,
        rating
    )

    if ai_probability >= STAGE1_THRESHOLD:
        ai_signal = "Likely AI-generated"
    else:
        ai_signal = "Likely human-written"

    if spam_probability >= STAGE2_THRESHOLD:
        spam_signal = "High spam probability"
    elif spam_probability >= 0.40:
        spam_signal = "Moderate spam probability"
    else:
        spam_signal = "Low spam probability"

    return {
        "ai_probability": round(ai_probability, 4),
        "human_probability": round(human_probability, 4),

        "spam_probability": round(spam_probability, 4),
        "genuine_probability": round(genuine_probability, 4),

        "classification": classification,
        "risk_level": risk_level,

        "signals": {
            "ai_detection": ai_signal,
            "spam_detection": spam_signal,
            "rating_text_consistency": consistency
        }
    }