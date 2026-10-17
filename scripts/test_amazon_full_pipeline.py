import json
import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

REVIEWS_FILE = "results/amazon_reviews_structured.json"

STAGE1_MODEL = "models/mlp_stage1_v2.keras"

STAGE2_MODEL = "models/stage2/stage2_page_model.pkl"
STAGE2_SCALER = "models/stage2/stage2_page_scaler.pkl"

STAGE2_FEATURES = (
    "models/stage2/stage2_page_features.txt"
)


# ============================================================
# LOAD REVIEWS
# ============================================================

print("=" * 70)
print("VERISIGHT - FULL AMAZON PIPELINE")
print("=" * 70)

with open(
    REVIEWS_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


reviews = data["reviews"]


print(
    f"\nAmazon reviews loaded: {len(reviews)}"
)


# ============================================================
# LOAD STAGE 1
# ============================================================

print("\nLoading Stage 1...")

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

from tensorflow.keras.models import load_model

stage1_model = load_model(
    STAGE1_MODEL
)

print("Stage 1 loaded.")


# ============================================================
# LOAD STAGE 2
# ============================================================

print("\nLoading Stage 2...")

stage2_model = joblib.load(
    STAGE2_MODEL
)

stage2_scaler = joblib.load(
    STAGE2_SCALER
)


with open(
    STAGE2_FEATURES,
    "r"
) as f:

    FEATURES = [
        line.strip()
        for line in f
        if line.strip()
    ]


print(
    f"Stage 2 loaded with "
    f"{len(FEATURES)} features."
)


# ============================================================
# STAGE 1
# ============================================================

print("\nGenerating MiniLM embeddings...")

texts = [
    review.get(
        "text",
        ""
    )
    for review in reviews
]


embeddings = encoder.encode(
    texts,
    batch_size=16,
    show_progress_bar=True
)


embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# STAGE 1 PREDICTIONS
# ============================================================

stage1_output = stage1_model.predict(
    embeddings,
    verbose=0
)


# Handle either sigmoid output or 2-class output

if stage1_output.ndim == 1:

    ai_probabilities = stage1_output

elif stage1_output.shape[1] == 1:

    ai_probabilities = (
        stage1_output[:, 0]
    )

else:

    ai_probabilities = (
        stage1_output[:, 1]
    )


ai_probabilities = np.asarray(
    ai_probabilities
).reshape(-1)


# ============================================================
# AMAZON PAGE INFORMATION
# ============================================================

product_review_count = len(reviews)


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
)


one_star_ratio = (
    one_star_count /
    product_review_count
)


# ============================================================
# DATE COUNTS
# ============================================================

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


# ============================================================
# DUPLICATE COUNTS
# ============================================================

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


# ============================================================
# SENTIMENT
# ============================================================

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "love",
    "loved",
    "nice",
    "best",
    "comfortable",
    "happy",
    "awesome",
    "perfect",
    "beautiful",
    "worth",
    "quality",
    "stylish",
    "easy",
    "authentic",
    "recommend",
    "recommended",
    "satisfied",
    "fantastic",
    "wonderful",
    "impressive",
    "light",
    "lightweight"
}


NEGATIVE_WORDS = {
    "bad",
    "poor",
    "worst",
    "hate",
    "hated",
    "terrible",
    "awful",
    "hard",
    "difficult",
    "problem",
    "problems",
    "issue",
    "issues",
    "disappointed",
    "waste",
    "broken",
    "damaged",
    "slow",
    "heavy",
    "uncomfortable",
    "fake",
    "return",
    "refund",
    "late",
    "wrong",
    "poorly",
    "useless",
    "not",
    "don't",
    "never"
}


def sentiment_score(text):

    words = text.split()

    if not words:

        return 0.0


    positive = 0

    negative = 0


    for word in words:

        word = (
            word.lower()
            .strip(
                ".,!?;:'\"()[]{}"
            )
        )

        if word in POSITIVE_WORDS:

            positive += 1

        if word in NEGATIVE_WORDS:

            negative += 1


    return (
        positive - negative
    ) / len(words)


# ============================================================
# STAGE 2 FEATURE CREATION
# ============================================================

def create_stage2_features(
    review
):

    text = review.get(
        "text",
        ""
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

        else 0
    )


    uppercase_ratio = (

        sum(
            char.isupper()
            for char in text
        )

        / char_count

        if char_count

        else 0
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

        else 0
    )


    sentiment = sentiment_score(
        text
    )


    rating_sentiment_difference = abs(

        (rating / 5.0)
        - sentiment
    )


    date = review.get(
        "date",
        ""
    )


    try:

        date_ordinal = (
            pd.Timestamp(
                date
            ).toordinal()
        )

    except Exception:

        date_ordinal = 0


    reviews_same_date = (
        date_counts.get(
            date,
            1
        )
    )


    normalized_text = (
        text.lower().strip()
    )


    duplicate_count = (
        text_counts.get(
            normalized_text,
            1
        )
    )


    duplicate_flag = (
        1
        if duplicate_count > 1
        else 0
    )


    return [

        rating,

        word_count,

        char_count,

        avg_word_length,

        uppercase_ratio,

        punctuation_count,

        exclamation_count,

        question_count,

        digit_ratio,

        sentiment,

        rating_sentiment_difference,

        date_ordinal,

        reviews_same_date,

        product_review_count,

        five_star_ratio,

        one_star_ratio,

        duplicate_count,

        duplicate_flag

    ]


# ============================================================
# CREATE STAGE 2 MATRIX
# ============================================================

stage2_rows = [

    create_stage2_features(
        review
    )

    for review in reviews
]


X_stage2 = pd.DataFrame(
    stage2_rows,
    columns=FEATURES
)


X_stage2 = X_stage2.replace(
    [np.inf, -np.inf],
    np.nan
)


X_stage2 = X_stage2.fillna(0)


# ============================================================
# STAGE 2 PREDICTION
# ============================================================

X_stage2_scaled = (
    stage2_scaler.transform(
        X_stage2
    )
)


stage2_probabilities = (
    stage2_model.predict_proba(
        X_stage2_scaled
    )[:, 1]
)


STAGE2_THRESHOLD = 0.19


# ============================================================
# FINAL FOUR-WAY CLASSIFICATION
# ============================================================

print()
print("=" * 70)
print("VERISIGHT AMAZON ANALYSIS")
print("=" * 70)


results = []


for i, review in enumerate(
    reviews
):

    ai_probability = float(
        ai_probabilities[i]
    )


    spam_probability = float(
        stage2_probabilities[i]
    )


    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    if ai_probability >= 0.50:

        author_class = "AI"

    else:

        author_class = "HUMAN"


    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    if (
        spam_probability
        >= STAGE2_THRESHOLD
    ):

        integrity_class = "SPAM"

    else:

        integrity_class = "GENUINE"


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    final_class = (
        f"{author_class} "
        f"{integrity_class}"
    )


    print()
    print(
        f"Review {i + 1}"
    )

    print(
        "-" * 70
    )

    print(
        f"AI Probability   : "
        f"{ai_probability * 100:.2f}%"
    )

    print(
        f"Author           : "
        f"{author_class}"
    )

    print(
        f"Spam Probability : "
        f"{spam_probability * 100:.2f}%"
    )

    print(
        f"Integrity        : "
        f"{integrity_class}"
    )

    print(
        f"FINAL            : "
        f"{final_class}"
    )

    print(
        f"Review           : "
        f"{review.get('text', '')[:120]}"
    )


    results.append({

        "review_number":
            i + 1,

        "rating":
            review.get(
                "rating"
            ),

        "ai_probability":
            round(
                ai_probability * 100,
                2
            ),

        "author_class":
            author_class,

        "spam_probability":
            round(
                spam_probability * 100,
                2
            ),

        "integrity_class":
            integrity_class,

        "final_class":
            final_class,

        "text":
            review.get(
                "text",
                ""
            )

    })


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)


from collections import Counter


class_counts = Counter(
    result["final_class"]
    for result in results
)


for classification, count in (
    class_counts.items()
):

    print(
        f"{classification:<20}"
        f": {count}"
    )


# ============================================================
# SAVE
# ============================================================

output_file = (
    "results/"
    "amazon_full_pipeline.json"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )


print()
print(
    f"Results saved to: "
    f"{output_file}"
)

print()
print("=" * 70)
print("FULL AMAZON PIPELINE COMPLETE")
print("=" * 70)