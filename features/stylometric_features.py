# ==========================================================
# VeriSight - Stylometric Feature Extractor
# ==========================================================
#
# Extracts writing-style features from reviews.
#
# These features are intended to complement the existing
# 384-dimensional MiniLM embeddings.
#
# ==========================================================

import re
import numpy as np
import pandas as pd


# ==========================================================
# Feature Extraction Function
# ==========================================================

def extract_stylometric_features(text):

    # ------------------------------------------------------
    # Basic cleaning
    # ------------------------------------------------------

    text = str(text).strip()

    if not text:
        return {
            "word_count": 0,
            "character_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "avg_word_length": 0,
            "vocabulary_diversity": 0,
            "type_token_ratio": 0,
            "punctuation_ratio": 0,
            "comma_ratio": 0,
            "exclamation_ratio": 0,
            "question_ratio": 0,
            "capitalization_ratio": 0,
            "digit_ratio": 0,
            "stopword_ratio": 0,
            "repeated_word_ratio": 0,
            "unique_word_ratio": 0,
            "sentence_length_std": 0,
            "paragraph_count": 0
        }

    # ------------------------------------------------------
    # Words
    # ------------------------------------------------------

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    word_count = len(words)

    # ------------------------------------------------------
    # Characters
    # ------------------------------------------------------

    character_count = len(text)

    # ------------------------------------------------------
    # Sentences
    # ------------------------------------------------------

    sentences = re.split(r"[.!?]+", text)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    sentence_count = len(sentences)

    # ------------------------------------------------------
    # Average sentence length
    # ------------------------------------------------------

    if sentence_count > 0:

        sentence_lengths = [
            len(re.findall(
                r"\b[a-zA-Z]+\b",
                sentence
            ))
            for sentence in sentences
        ]

        avg_sentence_length = np.mean(
            sentence_lengths
        )

        sentence_length_std = np.std(
            sentence_lengths
        )

    else:

        sentence_lengths = []

        avg_sentence_length = 0

        sentence_length_std = 0

    # ------------------------------------------------------
    # Average word length
    # ------------------------------------------------------

    if word_count > 0:

        avg_word_length = np.mean([
            len(word)
            for word in words
        ])

    else:

        avg_word_length = 0

    # ------------------------------------------------------
    # Vocabulary diversity
    # ------------------------------------------------------

    unique_words = set(words)

    unique_word_count = len(unique_words)

    if word_count > 0:

        vocabulary_diversity = (
            unique_word_count /
            word_count
        )

    else:

        vocabulary_diversity = 0

    type_token_ratio = vocabulary_diversity

    # ------------------------------------------------------
    # Punctuation
    # ------------------------------------------------------

    punctuation_count = len(
        re.findall(r"[^\w\s]", text)
    )

    if character_count > 0:

        punctuation_ratio = (
            punctuation_count /
            character_count
        )

    else:

        punctuation_ratio = 0

    # ------------------------------------------------------
    # Commas
    # ------------------------------------------------------

    comma_count = text.count(",")

    if character_count > 0:

        comma_ratio = (
            comma_count /
            character_count
        )

    else:

        comma_ratio = 0

    # ------------------------------------------------------
    # Exclamation marks
    # ------------------------------------------------------

    exclamation_count = text.count("!")

    if character_count > 0:

        exclamation_ratio = (
            exclamation_count /
            character_count
        )

    else:

        exclamation_ratio = 0

    # ------------------------------------------------------
    # Question marks
    # ------------------------------------------------------

    question_count = text.count("?")

    if character_count > 0:

        question_ratio = (
            question_count /
            character_count
        )

    else:

        question_ratio = 0

    # ------------------------------------------------------
    # Capitalization
    # ------------------------------------------------------

    uppercase_count = sum(
        1 for char in text
        if char.isupper()
    )

    alphabetic_count = sum(
        1 for char in text
        if char.isalpha()
    )

    if alphabetic_count > 0:

        capitalization_ratio = (
            uppercase_count /
            alphabetic_count
        )

    else:

        capitalization_ratio = 0

    # ------------------------------------------------------
    # Digit ratio
    # ------------------------------------------------------

    digit_count = sum(
        1 for char in text
        if char.isdigit()
    )

    if character_count > 0:

        digit_ratio = (
            digit_count /
            character_count
        )

    else:

        digit_ratio = 0

    # ------------------------------------------------------
    # Stopword ratio
    # ------------------------------------------------------
    #
    # A small built-in stopword list is used so that
    # additional packages are not required.
    #
    # ------------------------------------------------------

    stopwords = {
        "a", "an", "the", "and", "or", "but",
        "if", "then", "than", "this", "that",
        "these", "those", "is", "am", "are",
        "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "for", "with",
        "at", "by", "from", "as", "it", "its",
        "i", "me", "my", "we", "our", "you",
        "your", "he", "she", "they", "them",
        "their", "have", "has", "had", "do",
        "does", "did", "will", "would", "can",
        "could", "should"
    }

    stopword_count = sum(
        1 for word in words
        if word in stopwords
    )

    if word_count > 0:

        stopword_ratio = (
            stopword_count /
            word_count
        )

    else:

        stopword_ratio = 0

    # ------------------------------------------------------
    # Repeated words
    # ------------------------------------------------------

    word_counts = {}

    for word in words:

        word_counts[word] = (
            word_counts.get(word, 0) + 1
        )

    repeated_word_count = sum(
        count - 1
        for count in word_counts.values()
        if count > 1
    )

    if word_count > 0:

        repeated_word_ratio = (
            repeated_word_count /
            word_count
        )

    else:

        repeated_word_ratio = 0

    # ------------------------------------------------------
    # Unique word ratio
    # ------------------------------------------------------

    if word_count > 0:

        unique_word_ratio = (
            unique_word_count /
            word_count
        )

    else:

        unique_word_ratio = 0

    # ------------------------------------------------------
    # Paragraph count
    # ------------------------------------------------------

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    paragraph_count = len(paragraphs)

    # ------------------------------------------------------
    # Return features
    # ------------------------------------------------------

    return {

        "word_count": word_count,

        "character_count": character_count,

        "sentence_count": sentence_count,

        "avg_sentence_length": avg_sentence_length,

        "avg_word_length": avg_word_length,

        "vocabulary_diversity": vocabulary_diversity,

        "type_token_ratio": type_token_ratio,

        "punctuation_ratio": punctuation_ratio,

        "comma_ratio": comma_ratio,

        "exclamation_ratio": exclamation_ratio,

        "question_ratio": question_ratio,

        "capitalization_ratio": capitalization_ratio,

        "digit_ratio": digit_ratio,

        "stopword_ratio": stopword_ratio,

        "repeated_word_ratio": repeated_word_ratio,

        "unique_word_ratio": unique_word_ratio,

        "sentence_length_std": sentence_length_std,

        "paragraph_count": paragraph_count
    }


# ==========================================================
# Test Function
# ==========================================================

def main():

    print("=" * 65)
    print("VeriSight - Stylometric Feature Test")
    print("=" * 65)

    # ------------------------------------------------------
    # Test reviews
    # ------------------------------------------------------

    test_reviews = [

        "The food was really good, but the service was a little slow. I would visit again.",

        "I had an absolutely wonderful experience! The staff were friendly, the food was delicious, and the atmosphere was excellent.",

        "Bad service. Food was cold. Not coming back."
    ]

    # ------------------------------------------------------
    # Extract features
    # ------------------------------------------------------

    results = []

    for i, review in enumerate(
        test_reviews,
        start=1
    ):

        features = extract_stylometric_features(
            review
        )

        features["review_id"] = i

        results.append(features)

    # ------------------------------------------------------
    # Create DataFrame
    # ------------------------------------------------------

    df = pd.DataFrame(results)

    # Move review_id to first column

    columns = [
        "review_id"
    ] + [
        column
        for column in df.columns
        if column != "review_id"
    ]

    df = df[columns]

    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    print("\nExtracted Features:\n")

    print(
        df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------
    # Feature names
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("FEATURE COUNT")
    print("=" * 65)

    print(
        f"\nNumber of stylometric features: "
        f"{len(df.columns) - 1}"
    )

    print("\nFeature Names:")

    for column in df.columns:

        if column != "review_id":

            print(
                f"  - {column}"
            )

    print("\n" + "=" * 65)
    print("Test Completed")
    print("=" * 65)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()