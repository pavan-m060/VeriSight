import os
import re
import time
import random
import requests
import pandas as pd


# ============================================================
# VERISIGHT
# External Unseen Generator Dataset
#
# Models:
#   Qwen 2.5:3B
#   Phi-4-mini
#
# 320 total reviews
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "qwen2.5:3b",
    "phi4-mini"
]

TOTAL_REVIEWS = 320

OUTPUT_DIR = "data/external"

PROGRESS_FILE = os.path.join(
    OUTPUT_DIR,
    "external_unseen_length_matched_progress.csv"
)

FINAL_FILE = os.path.join(
    OUTPUT_DIR,
    "external_unseen_length_matched_320.csv"
)

DETAILED_FILE = os.path.join(
    OUTPUT_DIR,
    "external_unseen_length_matched_320_detailed.csv"
)

random.seed(42)


# ============================================================
# SOURCES
# ============================================================

SOURCES = [
    "Amazon",
    "Yelp",
    "TripAdvisor",
    "GooglePlay"
]


# ============================================================
# RATINGS
# ============================================================

RATINGS = [1, 2, 3, 4, 5]


# ============================================================
# TARGET LENGTH DISTRIBUTION
# Based on Stage-1 AI training data
# ============================================================

LENGTH_BINS = [
    (20, 40, 6),
    (41, 60, 13),
    (61, 80, 32),
    (81, 100, 109),
    (101, 120, 70),
    (121, 150, 48),
    (151, 200, 29),
    (201, 300, 11),
    (301, 450, 2)
]


# ============================================================
# WORD COUNT
# ============================================================

def count_words(text):

    return len(
        re.findall(
            r"\b[\w]+(?:['-][\w]+)*\b",
            str(text)
        )
    )


# ============================================================
# NORMALIZE
# ============================================================

def normalize_review(text):

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    return text.strip()


# ============================================================
# CLEAN MODEL OUTPUT
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.strip()

    prefixes = [
        "Review:",
        "review:",
        "Here is the review:",
        "Here’s the review:",
        "Here is a review:",
        "Here’s a review:"
    ]

    for prefix in prefixes:

        if text.startswith(prefix):

            text = text[
                len(prefix):
            ].strip()

    if (
        len(text) >= 2
        and text[0] in ['"', "'"]
        and text[-1] in ['"', "'"]
    ):

        text = text[1:-1].strip()

    return text


# ============================================================
# OLLAMA GENERATION
# ============================================================

def ollama_generate(
    model,
    prompt,
    temperature=0.8
):

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        if response.status_code != 200:

            print(
                f"      Ollama error: "
                f"{response.status_code}"
            )

            return ""

        data = response.json()

        return clean_text(
            data.get("response", "")
        )

    except Exception as e:

        print(
            f"      Generation error: {e}"
        )

        return ""


# ============================================================
# INITIAL REVIEW GENERATION
# ============================================================

def generate_initial_review(
    model,
    source,
    rating,
    min_words,
    max_words
):

    # Ask for slightly more than the minimum.
    # Smaller models tend to undershoot.
    target_words = min(
        max_words + 15,
        int(max_words * 1.15)
    )

    prompt = f"""
Write ONE realistic {source} review.

Rating: {rating}/5

The review should be approximately
{min_words}-{target_words} words long.

IMPORTANT:
- Write only the review.
- No title.
- No explanation.
- No "Review:" prefix.
- Do not mention AI or language models.
- Do not mention this prompt.
- Make it realistic and specific.
- Match the sentiment to the {rating}/5 rating.
- Use natural variation in sentence length.
- Do not copy any existing review.
- Write a complete review.

Try to produce at least {min_words} words.

Review:
"""

    review = ollama_generate(
        model,
        prompt
    )

    return review


# ============================================================
# EXPAND SHORT REVIEW
# ============================================================

def expand_review(
    model,
    review,
    source,
    rating,
    min_words,
    max_words
):

    current_words = count_words(
        review
    )

    attempts = 0

    while (
        current_words < min_words
        and attempts < 8
    ):

        needed = min_words - current_words

        # Ask for a reasonable amount of additional text
        requested_addition = max(
            25,
            min(
                needed + 20,
                80
            )
        )

        prompt = f"""
You are continuing an existing {source} review.

Rating: {rating}/5

Existing review:

{review}

The existing review has approximately
{current_words} words.

Add approximately {requested_addition} more words.

IMPORTANT:
- Continue naturally from the existing review.
- Do not repeat previous information.
- Do not start with "Sure", "Here is", or similar.
- Do not mention AI.
- Do not mention this instruction.
- Write only the additional review text.
- Keep the same tone and rating.
- Add useful details such as experience, quality,
  usability, service, value, delivery, features,
  atmosphere, or specific observations as appropriate.

Additional text:
"""

        addition = ollama_generate(
            model,
            prompt,
            temperature=0.75
        )

        if not addition:

            attempts += 1
            continue

        # Avoid obvious repetition
        if normalize_review(addition) in normalize_review(review):

            attempts += 1
            continue

        review = (
            review.strip()
            + " "
            + addition.strip()
        )

        current_words = count_words(
            review
        )

        attempts += 1

    return review


# ============================================================
# TRIM TO MAXIMUM
# ============================================================

def trim_to_max_words(
    text,
    max_words
):

    words = re.findall(
        r"\b[\w]+(?:['-][\w]+)*\b",
        text
    )

    if len(words) <= max_words:

        return text.strip()

    # First try sentence-level trimming
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    result = ""

    for sentence in sentences:

        candidate = (
            result + " " + sentence
        ).strip()

        if count_words(candidate) <= max_words:

            result = candidate

        else:

            break

    if count_words(result) >= max_words - 10:

        return result.strip()

    # Fallback: hard word trimming
    trimmed = words[:max_words]

    return " ".join(trimmed).strip()


# ============================================================
# GENERATE VALID REVIEW
# ============================================================

def generate_valid_review(
    model,
    source,
    rating,
    min_words,
    max_words
):

    for attempt in range(1, 6):

        print(
            f"      Generation attempt {attempt}"
        )

        review = generate_initial_review(
            model,
            source,
            rating,
            min_words,
            max_words
        )

        if not review:

            continue

        word_count = count_words(
            review
        )

        print(
            f"      Initial length: "
            f"{word_count} words"
        )

        # ----------------------------------------------------
        # If short -> expand
        # ----------------------------------------------------

        if word_count < min_words:

            print(
                f"      Expanding review "
                f"to at least {min_words} words..."
            )

            review = expand_review(
                model,
                review,
                source,
                rating,
                min_words,
                max_words
            )

            word_count = count_words(
                review
            )

            print(
                f"      Expanded length: "
                f"{word_count} words"
            )

        # ----------------------------------------------------
        # If still too short, try again
        # ----------------------------------------------------

        if word_count < min_words:

            print(
                "      Still too short."
            )

            continue

        # ----------------------------------------------------
        # Trim if too long
        # ----------------------------------------------------

        if word_count > max_words:

            review = trim_to_max_words(
                review,
                max_words
            )

            word_count = count_words(
                review
            )

            print(
                f"      Trimmed length: "
                f"{word_count} words"
            )

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        if (
            min_words
            <= word_count
            <= max_words
        ):

            return review

    return None


# ============================================================
# CREATE LENGTH TARGETS
# ============================================================

def create_length_targets():

    targets = []

    for min_words, max_words, count in LENGTH_BINS:

        for _ in range(count):

            targets.append(
                (
                    min_words,
                    max_words
                )
            )

    random.shuffle(
        targets
    )

    return targets


# ============================================================
# CREATE JOBS
# ============================================================

def create_jobs():

    jobs = []

    for source in SOURCES:

        for rating in RATINGS:

            for model in MODELS:

                # 8 reviews per model
                # per source/rating combination
                for _ in range(8):

                    jobs.append(
                        {
                            "source": source,
                            "rating": rating,
                            "model": model
                        }
                    )

    random.shuffle(
        jobs
    )

    return jobs


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():

    print(
        "\nChecking Ollama..."
    )

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )

        data = response.json()

        installed = [
            x.get("name", "")
            for x in data.get(
                "models",
                []
            )
        ]

        print(
            "\nInstalled models:"
        )

        for model in installed:

            print(
                "  -",
                model
            )

        for required in MODELS:

            found = any(
                x == required
                or x.startswith(
                    required + ":"
                )
                for x in installed
            )

            if not found:

                print(
                    f"\nERROR: "
                    f"{required} not installed."
                )

                return False

        print(
            "\nAll required models available."
        )

        return True

    except Exception as e:

        print(
            "\nCannot connect to Ollama:"
        )

        print(e)

        return False


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_progress(records):

    if not records:

        return

    pd.DataFrame(
        records
    ).to_csv(
        PROGRESS_FILE,
        index=False
    )


# ============================================================
# VALIDATE FINAL DATASET
# ============================================================

def validate_dataset(df):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL DATASET VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal reviews: {len(df)}"
    )

    print(
        "\nGenerator distribution:"
    )

    print(
        df[
            "generator_model"
        ].value_counts()
    )

    print(
        "\nSource distribution:"
    )

    print(
        df[
            "source"
        ].value_counts()
    )

    print(
        "\nRating distribution:"
    )

    print(
        df[
            "rating"
        ].value_counts()
        .sort_index()
    )

    print(
        "\nSource × Rating:"
    )

    print(
        pd.crosstab(
            df["source"],
            df["rating"]
        )
    )

    print(
        "\nGenerator × Source:"
    )

    print(
        pd.crosstab(
            df["generator_model"],
            df["source"]
        )
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = (
        df["review"]
        .apply(normalize_review)
        .duplicated()
        .sum()
    )

    print(
        f"\nDuplicate reviews: {duplicates}"
    )

    # --------------------------------------------------------
    # Word counts
    # --------------------------------------------------------

    df["word_count"] = (
        df["review"]
        .apply(count_words)
    )

    print(
        "\nWord count statistics:"
    )

    print(
        df["word_count"].describe()
    )

    # --------------------------------------------------------
    # Length bins
    # --------------------------------------------------------

    def find_bin(words):

        for low, high, _ in LENGTH_BINS:

            if low <= words <= high:

                return f"{low}-{high}"

        return "OUT_OF_RANGE"

    df["length_bin"] = (
        df["word_count"]
        .apply(find_bin)
    )

    print(
        "\nLength distribution:"
    )

    print(
        df["length_bin"].value_counts()
    )

    out_of_range = (
        df["length_bin"]
        == "OUT_OF_RANGE"
    ).sum()

    print(
        f"\nOut-of-range: {out_of_range}"
    )

    print(
        "\nLabel distribution:"
    )

    print(
        df["label"].value_counts()
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VERISIGHT - UNSEEN GENERATOR DATASET"
    )

    print(
        "=" * 70
    )

    print(
        "\nModels:"
    )

    for model in MODELS:

        print(
            "  -",
            model
        )

    print(
        "\nTotal reviews:",
        TOTAL_REVIEWS
    )

    print(
        "Per generator:",
        TOTAL_REVIEWS // 2
    )

    print(
        "=" * 70
    )

    if not check_ollama():

        return

    # --------------------------------------------------------
    # Create jobs
    # --------------------------------------------------------

    jobs = create_jobs()

    targets = create_length_targets()

    records = []

    # --------------------------------------------------------
    # IMPORTANT:
    # Start fresh.
    #
    # Delete old progress file if it exists because the previous
    # script used a different generation strategy.
    # --------------------------------------------------------

    if os.path.exists(
        PROGRESS_FILE
    ):

        print(
            "\nExisting progress file found."
        )

        print(
            "The new generation strategy will "
            "start fresh."
        )

        try:

            os.remove(
                PROGRESS_FILE
            )

        except Exception:

            pass

    existing_reviews = set()

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    for index, job in enumerate(
        jobs
    ):

        job_number = index + 1

        source = job["source"]

        rating = job["rating"]

        model = job["model"]

        min_words, max_words = (
            targets[index]
        )

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"[{job_number}/{TOTAL_REVIEWS}]"
        )

        print(
            f"Model  : {model}"
        )

        print(
            f"Source : {source}"
        )

        print(
            f"Rating : {rating}/5"
        )

        print(
            f"Target : {min_words}-{max_words} words"
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        review = generate_valid_review(
            model,
            source,
            rating,
            min_words,
            max_words
        )

        if review is None:

            print(
                "\n      FAILED."
            )

            print(
                "      Trying another fresh generation..."
            )

            continue

        # ----------------------------------------------------
        # Duplicate check
        # ----------------------------------------------------

        normalized = normalize_review(
            review
        )

        if normalized in existing_reviews:

            print(
                "      Duplicate detected."
            )

            print(
                "      Skipping."
            )

            continue

        existing_reviews.add(
            normalized
        )

        word_count = count_words(
            review
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        record = {
            "review_id":
                f"UNSEEN_{len(records)+1:03d}",

            "review":
                review,

            "source":
                source,

            "rating":
                rating,

            "generator_model":
                model,

            "label":
                1,

            "word_count":
                word_count,

            "target_min":
                min_words,

            "target_max":
                max_words
        }

        records.append(
            record
        )

        save_progress(
            records
        )

        print(
            f"      ✓ SUCCESS: "
            f"{word_count} words"
        )

        time.sleep(0.2)

    # ========================================================
    # CHECK WHETHER 320 WERE CREATED
    # ========================================================

    if not records:

        print(
            "\nNo reviews generated."
        )

        return

    df = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    df = validate_dataset(
        df
    )

    # --------------------------------------------------------
    # Save detailed file
    # --------------------------------------------------------

    df.to_csv(
        DETAILED_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final clean file
    # --------------------------------------------------------

    final_columns = [
        "review_id",
        "review",
        "source",
        "rating",
        "generator_model",
        "label"
    ]

    final_df = df[
        final_columns
    ].copy()

    final_df.to_csv(
        FINAL_FILE,
        index=False
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GENERATION FINISHED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCreated: {len(final_df)} reviews"
    )

    print(
        f"\nFinal CSV:"
    )

    print(
        FINAL_FILE
    )

    print(
        f"\nDetailed CSV:"
    )

    print(
        DETAILED_FILE
    )

    if len(final_df) == 320:

        print(
            "\n✓ COMPLETE 320-REVIEW EXTERNAL DATASET"
        )

    else:

        print(
            f"\n⚠ Only {len(final_df)}/320 "
            "reviews were generated."
        )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()