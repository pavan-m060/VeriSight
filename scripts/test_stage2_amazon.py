import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

REVIEWS_FILE = "results/amazon_reviews_structured.json"

MODEL_FILE = "models/stage2/stage2_page_model.pkl"

SCALER_FILE = "models/stage2/stage2_page_scaler.pkl"

FEATURES_FILE = "models/stage2/stage2_page_features.txt"


# ============================================================
# LOAD AMAZON REVIEWS
# ============================================================

print("=" * 70)
print("VERISIGHT - AMAZON STAGE 2 TEST")
print("=" * 70)

with open(
    REVIEWS_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


reviews = data["reviews"]


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)

scaler = joblib.load(
    SCALER_FILE
)


# ============================================================
# LOAD FEATURE ORDER
# ============================================================

with open(
    FEATURES_FILE,
    "r",
    encoding="utf-8"
) as f:

    FEATURES = [
        line.strip()
        for line in f
        if line.strip()
    ]


print(
    f"\nAmazon reviews : {len(reviews)}"
)

print(
    f"Model features : {len(FEATURES)}"
)


# ============================================================
# AMAZON PAGE-LEVEL INFORMATION
# ============================================================

product_review_count = len(reviews)


five_star_count = sum(
    1
    for review in reviews
    if float(review.get("rating", 0)) == 5
)


one_star_count = sum(
    1
    for review in reviews
    if float(review.get("rating", 0)) == 1
)


five_star_ratio = (
    five_star_count /
    product_review_count
    if product_review_count > 0
    else 0
)


one_star_ratio = (
    one_star_count /
    product_review_count
    if product_review_count > 0
    else 0
)


# ============================================================
# REVIEW DATE COUNTS
# ============================================================

date_counts = {}

for review in reviews:

    review_date = review.get(
        "date",
        ""
    )

    date_counts[review_date] = (
        date_counts.get(review_date, 0) + 1
    )


# ============================================================
# DUPLICATE TEXT COUNTS
# ============================================================

text_counts = {}

for review in reviews:

    text = review.get(
        "text",
        ""
    ).strip().lower()

    text_counts[text] = (
        text_counts.get(text, 0) + 1
    )


# ============================================================
# SENTIMENT LEXICON
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


# ============================================================
# SENTIMENT FUNCTION
# ============================================================

def calculate_sentiment(text):

    words = text.split()

    if not words:
        return 0.0

    positive_count = 0

    negative_count = 0

    for word in words:

        clean_word = (
            word.lower()
            .strip(
                ".,!?;:'\"()[]{}"
            )
        )

        if clean_word in POSITIVE_WORDS:

            positive_count += 1

        if clean_word in NEGATIVE_WORDS:

            negative_count += 1


    sentiment = (
        positive_count -
        negative_count
    ) / len(words)


    return sentiment


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(review):

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


    # --------------------------------------------------------
    # TEXT FEATURES
    # --------------------------------------------------------

    words = text.split()

    word_count = len(words)

    char_count = len(text)


    avg_word_length = (

        sum(
            len(word)
            for word in words
        )

        / word_count

        if word_count > 0

        else 0
    )


    uppercase_chars = sum(
        1
        for char in text
        if char.isupper()
    )


    uppercase_ratio = (

        uppercase_chars /
        char_count

        if char_count > 0

        else 0
    )


    punctuation_count = sum(

        1
        for char in text
        if char in
        ".,!?;:'\"-()[]{}"
    )


    exclamation_count = text.count("!")

    question_count = text.count("?")


    digit_ratio = (

        sum(
            char.isdigit()
            for char in text
        )

        / char_count

        if char_count > 0

        else 0
    )


    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    sentiment_score = calculate_sentiment(
        text
    )


    rating_sentiment_difference = abs(

        (rating / 5.0)

        - sentiment_score
    )


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    review_date = review.get(
        "date",
        ""
    )


    try:

        review_date_ordinal = (
            pd.Timestamp(
                review_date
            ).toordinal()
        )

    except Exception:

        review_date_ordinal = 0


    # --------------------------------------------------------
    # SAME-DATE REVIEWS
    # --------------------------------------------------------

    reviews_same_date = date_counts.get(

        review_date,

        1
    )


    # --------------------------------------------------------
    # DUPLICATE TEXT
    # --------------------------------------------------------

    normalized_text = text.lower().strip()


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


    # --------------------------------------------------------
    # FINAL 18 FEATURES
    # --------------------------------------------------------

    feature_values = [

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

    ]


    return feature_values


# ============================================================
# CREATE FEATURE MATRIX
# ============================================================

rows = []


for review in reviews:

    features = create_features(
        review
    )

    rows.append(features)


X = pd.DataFrame(
    rows,
    columns=FEATURES
)


# ============================================================
# CLEAN FEATURES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


X = X.fillna(0)


# ============================================================
# DISPLAY FEATURES
# ============================================================

print()
print("=" * 70)
print("FEATURE MATRIX")
print("=" * 70)

print(
    X.to_string(index=False)
)


# ============================================================
# SCALE
# ============================================================

X_scaled = scaler.transform(
    X
)


# ============================================================
# PREDICTION
# ============================================================

probabilities = model.predict_proba(
    X_scaled
)[:, 1]


# ============================================================
# CLASSIFICATION THRESHOLD
# ============================================================

THRESHOLD = 0.19


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("AMAZON REVIEW RESULTS")
print("=" * 70)


results = []


for index, (
    review,
    probability
) in enumerate(
    zip(
        reviews,
        probabilities
    ),
    start=1
):

    prediction = (

        "SPAM"

        if probability >= THRESHOLD

        else "GENUINE"
    )


    results.append({

        "review_number": index,

        "rating": review.get(
            "rating"
        ),

        "spam_probability":
            round(
                float(probability) * 100,
                2
            ),

        "prediction":
            prediction,

        "text":
            review.get(
                "text",
                ""
            )

    })


    print()
    print(
        f"Review {index}"
    )

    print(
        "-" * 70
    )

    print(
        f"Rating          : "
        f"{review.get('rating')}"
    )

    print(
        f"Spam Probability: "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Prediction      : "
        f"{prediction}"
    )

    print(
        f"Text            : "
        f"{review.get('text', '')[:150]}"
    )


# ============================================================
# SUMMARY
# ============================================================

spam_count = sum(

    1
    for probability in probabilities

    if probability >= THRESHOLD
)


genuine_count = (
    len(reviews)
    - spam_count
)


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Total Reviews : {len(reviews)}"
)

print(
    f"Spam          : {spam_count}"
)

print(
    f"Genuine       : {genuine_count}"
)

print(
    f"Threshold     : {THRESHOLD}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    "results/"
    "amazon_stage2_predictions.json"
)


with open(
    output_path,
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
    f"{output_path}"
)

print()
print("=" * 70)
print("AMAZON STAGE 2 TEST COMPLETE")
print("=" * 70)