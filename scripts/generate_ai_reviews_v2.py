import random
from pathlib import Path

import ollama
import pandas as pd
from tqdm import tqdm

# ===========================
# Paths
# ===========================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "human_reviews.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "ai_reviews_v2.csv"

TARGET_NEW_REVIEWS = 20000
# ===========================
# Models
# ===========================

MODELS = [
    "llama3.2:3b",
    "gemma3:4b",
    "mistral:7b"
]

# ===========================
# Load Data
# ===========================

def load_human_reviews():
    return pd.read_csv(INPUT_FILE)


def load_existing_reviews():
    if OUTPUT_FILE.exists():
        return pd.read_csv(OUTPUT_FILE)

    return pd.DataFrame(columns=[
        "review",
        "rating",
        "source",
        "category",
        "label",
        "generator_model"
    ])


def save_reviews(df):
    df.to_csv(OUTPUT_FILE, index=False)

# ===========================
# Generate AI Review
# ===========================

def generate_ai_review(rating, category):

    prompt = f"""
Write a completely new customer product review.

Category: {category}
Rating: {rating}/5

Requirements:
- Create the review from scratch.
- Do not rewrite, paraphrase, or copy any existing review.
- Write as a realistic customer who actually used the product.
- Keep the review consistent with the given rating.
- Use 30-120 words.
- Use natural and varied language.
- Do not mention AI, language models, or this prompt.
- No headings.
- No bullet points.
- Return ONLY the review.
"""

    model = random.choice(MODELS)

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    ai_review = response["message"]["content"].strip()

    return ai_review, model

# ===========================
# Main
# ===========================

def main():

    print("=" * 60)
    print("VeriSight AI Review Generator V2")
    print("=" * 60)

    human = load_human_reviews()
    ai = load_existing_reviews()

    start = len(ai)

    # Generate only 100 for the speed test
    end = min(start + TARGET_NEW_REVIEWS, len(human))

    print(f"Human Reviews       : {len(human):,}")
    print(f"Already Generated   : {start:,}")
    print(f"This Run            : {end - start:,}")

    if start >= len(human):
        print("\nAll reviews already generated.")
        return

    print(f"\nGenerating reviews {start + 1} to {end}\n")

    for i in tqdm(range(start, end), desc="Generating"):

        row = human.iloc[i]

        try:

            review, model = generate_ai_review(
                row["rating"],
                row["category"]
            )

            ai.loc[len(ai)] = [
                review,
                row["rating"],
                row["source"],
                row["category"],
                1,
                model
            ]

            # Save every 10 reviews during the test
            if len(ai) % 10 == 0:
                save_reviews(ai)

        except Exception as e:

            print(f"\nError at review {i + 1}: {e}")
            continue

    # Final save
    save_reviews(ai)

    print("\n")
    print("=" * 60)
    print("Generation Completed")
    print("=" * 60)
    print(f"Total AI Reviews : {len(ai):,}")
    print(f"Saved to         : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()