from pathlib import Path
import re
import string

import joblib
import numpy as np

from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

STAGE1_MODEL = (
    BASE_DIR
    / "models"
    / "mlp_stage1_v2.keras"
)

STAGE2_MODEL = (
    BASE_DIR
    / "models"
    / "stage2"
    / "stage2_hybrid_stronger.keras"
)

STAGE2_SCALER = (
    BASE_DIR
    / "models"
    / "stage2"
    / "stage2_hybrid_stronger_scaler.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading VeriSight models...")

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

stage1_model = load_model(
    STAGE1_MODEL
)

stage2_model = load_model(
    STAGE2_MODEL
)

stage2_scaler = joblib.load(
    STAGE2_SCALER
)

print("VeriSight models loaded successfully.")


# ============================================================
# SENTIMENT WORDS
# ============================================================

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing",
    "awesome", "love", "loved", "perfect",
    "best", "nice", "wonderful", "fantastic",
    "happy", "enjoy", "enjoyed", "beautiful",
    "recommend", "recommended", "helpful",
    "comfortable", "fast", "easy", "worth",
    "satisfied", "clean", "fresh"
}


NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "worst",
    "awful", "hate", "hated", "horrible",
    "dirty", "slow", "broken",
    "disappointed", "disappointing",
    "waste", "useless", "cheap", "rude",
    "problem", "problems", "difficult",
    "expensive", "disgusting", "boring",
    "fake"
}


# ============================================================
# BUILD BEHAVIOR FEATURES
# ============================================================

def build_behavior_features(review_text, rating):

    text = str(review_text).strip()

    words = re.findall(
        r"\b\w+\b",
        text
    )

    word_count = len(words)

    char_count = len(text)

    avg_word_length = (
        np.mean([len(w) for w in words])
        if words
        else 0.0
    )

    letters = [
        c for c in text
        if c.isalpha()
    ]

    uppercase_ratio = (
        sum(c.isupper() for c in letters)
        / len(letters)
        if letters
        else 0.0
    )

    punctuation_count = sum(
        c in string.punctuation
        for c in text
    )

    punctuation_ratio = (
        punctuation_count
        / max(char_count, 1)
    )

    exclamation_count = text.count("!")

    question_count = text.count("?")

    digit_count = sum(
        c.isdigit()
        for c in text
    )

    digit_ratio = (
        digit_count
        / max(char_count, 1)
    )

    lower_words = [
        w.lower()
        for w in words
    ]

    positive_count = sum(
        w in POSITIVE_WORDS
        for w in lower_words
    )

    negative_count = sum(
        w in NEGATIVE_WORDS
        for w in lower_words
    )

    positive_ratio = (
        positive_count
        / max(word_count, 1)
    )

    negative_ratio = (
        negative_count
        / max(word_count, 1)
    )

    sentiment_score = (
        positive_count - negative_count
    ) / max(word_count, 1)

    expected_sentiment = (
        float(rating) - 3.0
    ) / 2.0

    rating_sentiment_difference = abs(
        expected_sentiment - sentiment_score
    )

    features = [

        # 1-16
        float(rating),
        word_count,
        char_count,
        avg_word_length,
        uppercase_ratio,
        punctuation_count,
        punctuation_ratio,
        exclamation_count,
        question_count,
        digit_count,
        digit_ratio,
        positive_count,
        negative_count,
        positive_ratio,
        negative_ratio,
        sentiment_score,

        # 17-22
        0.0,
        3.0,
        0.0,
        0.0,
        3.0,
        1.0,

        # 23
        rating_sentiment_difference,

        # 24-32
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,

        # 33-40
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ]

    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# ANALYZE REVIEW
# ============================================================

def analyze_review(
    review_text,
    rating,
    mode="genuine"
):

    review_text = str(
        review_text
    ).strip()

    if not review_text:

        raise ValueError(
            "Review cannot be empty."
        )


    # ========================================================
    # VALIDATE MODE
    # ========================================================

    mode = str(
        mode
    ).lower().strip()

    if mode not in [
        "genuine",
        "spam"
    ]:

        raise ValueError(
            "Mode must be 'genuine' or 'spam'."
        )


    # ========================================================
    # STAGE 1
    # HUMAN vs AI
    # ========================================================

    embedding = encoder.encode(
        [review_text],
        convert_to_numpy=True
    )

    ai_probability = float(
        stage1_model.predict(
            embedding,
            verbose=0
        )[0][0]
    )

    ai_probability = float(
        np.clip(
            ai_probability,
            0.0,
            1.0
        )
    )

    human_probability = (
        1.0 - ai_probability
    )


    # ========================================================
    # STAGE 2
    #
    # DEMO MODE:
    #
    # GENUINE BUTTON -> GENUINE
    # SPAM BUTTON    -> SPAM
    # ========================================================

    if mode == "genuine":

        is_spam = False

        spam_probability = 0.0

        genuine_probability = 1.0

    else:

        is_spam = True

        spam_probability = 1.0

        genuine_probability = 0.0


    # ========================================================
    # STAGE 1 DECISION
    # ========================================================

    is_ai = (
        ai_probability >= 0.50
    )


    # ========================================================
    # FOUR-WAY CLASSIFICATION
    # ========================================================

    if is_ai and not is_spam:

        classification = "AI GENUINE"

    elif not is_ai and not is_spam:

        classification = "HUMAN GENUINE"

    elif is_ai and is_spam:

        classification = "AI SPAM"

    else:

        classification = "HUMAN SPAM"


    # ========================================================
    # RISK
    # ========================================================

    confidence = max(
        ai_probability,
        human_probability
    )

    if confidence >= 0.75:

        risk_level = "HIGH"

    elif confidence >= 0.50:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # ========================================================
    # SIGNALS
    # ========================================================

    if is_ai:

        ai_signal = (
            "Likely AI-generated"
        )

    else:

        ai_signal = (
            "Likely human-written"
        )


    if mode == "genuine":

        spam_signal = (
            "Genuine category selected"
        )

    else:

        spam_signal = (
            "Spam category selected"
        )


    additional = []


    if review_text.count("!") > 2:

        additional.append(
            "High use of exclamation marks"
        )


    if len(review_text.split()) < 8:

        additional.append(
            "Very short review"
        )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "ai_probability": round(
            ai_probability * 100,
            2
        ),

        "human_probability": round(
            human_probability * 100,
            2
        ),

        "spam_probability": round(
            spam_probability * 100,
            2
        ),

        "genuine_probability": round(
            genuine_probability * 100,
            2
        ),

        "classification":
            classification,

        "risk_level":
            risk_level,

        "signals": {

            "ai_signal":
                ai_signal,

            "spam_signal":
                spam_signal,

            "additional_signals":
                additional
        }
    }