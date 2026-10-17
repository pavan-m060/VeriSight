from pathlib import Path
import json
import re
import string

import joblib
import numpy as np
import pandas as pd

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
    / "stage2_page_model.pkl"
)

STAGE2_SCALER = (
    BASE_DIR
    / "models"
    / "stage2"
    / "stage2_page_scaler.pkl"
)

STAGE2_FEATURES = (
    BASE_DIR
    / "models"
    / "stage2"
    / "stage2_page_features.txt"
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

stage2_model = joblib.load(
    STAGE2_MODEL
)

stage2_scaler = joblib.load(
    STAGE2_SCALER
)

with open(
    STAGE2_FEATURES,
    "r",
    encoding="utf-8"
) as f:

    STAGE2_FEATURE_NAMES = [
        line.strip()
        for line in f
        if line.strip()
    ]


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
    "satisfied", "clean", "fresh", "quality",
    "stylish", "authentic", "impressive",
    "light", "lightweight"
}


NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "worst",
    "awful", "hate", "hated", "horrible",
    "dirty", "slow", "broken",
    "disappointed", "disappointing",
    "waste", "useless", "cheap", "rude",
    "problem", "problems", "difficult",
    "expensive", "disgusting", "boring",
    "fake", "hard", "issue", "issues",
    "return", "refund", "wrong",
    "uncomfortable", "never", "not",
    "don't"
}


# ============================================================
# SENTIMENT
# ============================================================

def calculate_sentiment(text):

    words = re.findall(
        r"\b\w+\b",
        str(text)
    )

    if not words:
        return 0.0

    positive_count = sum(
        word.lower() in POSITIVE_WORDS
        for word in words
    )

    negative_count = sum(
        word.lower() in NEGATIVE_WORDS
        for word in words
    )

    return (
        positive_count -
        negative_count
    ) / len(words)


# ============================================================
# BUILD STAGE 2 FEATURES
# ============================================================

def build_page_features(reviews):

    product_review_count = len(
        reviews
    )

    five_star_count = sum(
        1
        for review in reviews
        if float(
            review.get("rating", 0)
        ) == 5
    )

    one_star_count = sum(
        1
        for review in reviews
        if float(
            review.get("rating", 0)
        ) == 1
    )

    five_star_ratio = (
        five_star_count /
        product_review_count
        if product_review_count
        else 0.0
    )

    one_star_ratio = (
        one_star_count /
        product_review_count
        if product_review_count
        else 0.0
    )


    # --------------------------------------------------------
    # DATE COUNTS
    # --------------------------------------------------------

    date_counts = {}

    for review in reviews:

        date = review.get(
            "date",
            ""
        )

        date_counts[date] = (
            date_counts.get(
                date,
                0
            ) + 1
        )


    # --------------------------------------------------------
    # DUPLICATE COUNTS
    # --------------------------------------------------------

    text_counts = {}

    for review in reviews:

        text = review.get(
            "text",
            ""
        ).strip().lower()

        text_counts[text] = (
            text_counts.get(
                text,
                0
            ) + 1
        )


    rows = []


    # --------------------------------------------------------
    # CREATE FEATURES FOR EACH REVIEW
    # --------------------------------------------------------

    for review in reviews:

        text = str(
            review.get(
                "text",
                ""
            )
        ).strip()

        rating = float(
            review.get(
                "rating",
                5.0
            )
        )


        words = text.split()

        word_count = len(words)

        char_count = len(text)


        avg_word_length = (

            sum(
                len(word)
                for word in words
            )
            / word_count

            if word_count

            else 0.0
        )


        uppercase_ratio = (

            sum(
                char.isupper()
                for char in text
            )
            / char_count

            if char_count

            else 0.0
        )


        punctuation_count = sum(

            1

            for char in text

            if char in
            ".,!?;:'\"-()[]{}"
        )


        exclamation_count = (
            text.count("!")
        )

        question_count = (
            text.count("?")
        )


        digit_ratio = (

            sum(
                char.isdigit()
                for char in text
            )
            / char_count

            if char_count

            else 0.0
        )


        sentiment_score = (
            calculate_sentiment(
                text
            )
        )


        rating_sentiment_difference = abs(

            (rating / 5.0)
            - sentiment_score
        )


        date = review.get(
            "date",
            ""
        )


        try:

            review_date_ordinal = (
                pd.Timestamp(
                    date
                ).toordinal()
            )

        except Exception:

            review_date_ordinal = 0


        reviews_same_date = (
            date_counts.get(
                date,
                1
            )
        )


        normalized_text = (
            text.lower().strip()
        )


        duplicate_text_count = (
            text_counts.get(
                normalized_text,
                1
            )
        )


        is_duplicate_text = (

            1

            if duplicate_text_count > 1

            else 0
        )


        rows.append([

            rating,

            word_count,

            char_count,

            avg_word_length,

            uppercase_ratio,

            punctuation_count,

            exclamation_count,

            question_count,

            digit_ratio,

            sentiment_score,

            rating_sentiment_difference,

            review_date_ordinal,

            reviews_same_date,

            product_review_count,

            five_star_ratio,

            one_star_ratio,

            duplicate_text_count,

            is_duplicate_text

        ])


    X = pd.DataFrame(
        rows,
        columns=STAGE2_FEATURE_NAMES
    )


    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)


    return X


# ============================================================
# STAGE 1
# ============================================================

def predict_ai_probability(
    review_text
):

    embedding = encoder.encode(
        [review_text],
        convert_to_numpy=True
    )


    output = stage1_model.predict(
        embedding,
        verbose=0
    )


    ai_probability = float(
        output[0][0]
    )


    ai_probability = float(
        np.clip(
            ai_probability,
            0.0,
            1.0
        )
    )


    return ai_probability


# ============================================================
# ANALYZE MULTIPLE AMAZON REVIEWS
# ============================================================

def analyze_amazon_reviews(
    reviews
):

    if not reviews:

        raise ValueError(
            "No reviews were extracted from the page."
        )


    # ========================================================
    # STAGE 2 FEATURES
    # ========================================================

    X_stage2 = build_page_features(
        reviews
    )


    X_scaled = stage2_scaler.transform(
        X_stage2
    )


    spam_probabilities = (
        stage2_model.predict_proba(
            X_scaled
        )[:, 1]
    )


    STAGE2_THRESHOLD = 0.19


    results = []


    # ========================================================
    # PROCESS EACH REVIEW
    # ========================================================

    for index, review in enumerate(
        reviews
    ):

        text = str(
            review.get(
                "text",
                ""
            )
        ).strip()


        # ----------------------------------------------------
        # STAGE 1
        # ----------------------------------------------------

        ai_probability = (
            predict_ai_probability(
                text
            )
        )


        human_probability = (
            1.0 -
            ai_probability
        )


        if ai_probability >= 0.50:

            author_class = "AI"

        else:

            author_class = "HUMAN"


        # ----------------------------------------------------
        # STAGE 2
        # ----------------------------------------------------

        spam_probability = float(
            spam_probabilities[index]
        )


        genuine_probability = (
            1.0 -
            spam_probability
        )


        if (
            spam_probability
            >= STAGE2_THRESHOLD
        ):

            integrity_class = "SPAM"

        else:

            integrity_class = "GENUINE"


        # ----------------------------------------------------
        # FOUR-WAY CLASS
        # ----------------------------------------------------

        classification = (
            f"{author_class} "
            f"{integrity_class}"
        )


        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        risk_confidence = max(
            ai_probability,
            human_probability,
            spam_probability,
            genuine_probability
        )


        if risk_confidence >= 0.75:

            risk_level = "HIGH"

        elif risk_confidence >= 0.50:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        # ----------------------------------------------------
        # SIGNALS
        # ----------------------------------------------------

        if author_class == "AI":

            ai_signal = (
                "Likely AI-generated"
            )

        else:

            ai_signal = (
                "Likely human-written"
            )


        if integrity_class == "SPAM":

            spam_signal = (
                "Page-based spam indicators detected"
            )

        else:

            spam_signal = (
                "No strong page-based spam indicators"
            )


        additional_signals = []


        if len(text.split()) < 8:

            additional_signals.append(
                "Very short review"
            )


        if text.count("!") > 2:

            additional_signals.append(
                "High use of exclamation marks"
            )


        if review.get(
            "verified_purchase",
            False
        ):

            additional_signals.append(
                "Verified purchase"
            )


        results.append({

            "review_number":
                index + 1,

            "rating":
                review.get(
                    "rating"
                ),

            "review_text":
                text,

            "date":
                review.get(
                    "date"
                ),

            "verified_purchase":
                review.get(
                    "verified_purchase",
                    False
                ),

            "helpful_count":
                review.get(
                    "helpful_count",
                    0
                ),

            "ai_probability":
                round(
                    ai_probability * 100,
                    2
                ),

            "human_probability":
                round(
                    human_probability * 100,
                    2
                ),

            "spam_probability":
                round(
                    spam_probability * 100,
                    2
                ),

            "genuine_probability":
                round(
                    genuine_probability * 100,
                    2
                ),

            "author_class":
                author_class,

            "integrity_class":
                integrity_class,

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
                    additional_signals
            }
        })


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "total_reviews":
            len(results),

        "human_reviews":
            sum(
                r["author_class"] == "HUMAN"
                for r in results
            ),

        "ai_reviews":
            sum(
                r["author_class"] == "AI"
                for r in results
            ),

        "genuine_reviews":
            sum(
                r["integrity_class"] == "GENUINE"
                for r in results
            ),

        "spam_reviews":
            sum(
                r["integrity_class"] == "SPAM"
                for r in results
            )
    }


    return {

        "reviews":
            results,

        "summary":
            summary
    }