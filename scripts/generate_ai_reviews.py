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
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "ai_reviews.csv"

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

def generate_ai_review(review, rating, category):

    prompt = f"""
Rewrite the following HUMAN product review into a completely natural customer review.

Rules:
- Keep the same sentiment.
- Keep the same rating ({rating}/5).
- Keep the same meaning.
- Write naturally like another customer.
- Use 30-120 words.
- No headings.
- No bullet points.
- Return ONLY the review.

Category:
{category}

Original Review:
{review}
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
    print("VeriSight AI Review Generator")
    print("=" * 60)

    human = load_human_reviews()
    ai = load_existing_reviews()

    start = len(ai)
    end = len(human)

    print(f"Human Reviews : {len(human):,}")
    print(f"Already Generated : {start:,}")
    print(f"Remaining : {end-start:,}")

    if start >= end:
        print("\nAll reviews already generated.")
        return

    print(f"\nGenerating reviews {start+1} to {end}\n")

    for i in tqdm(range(start, end), desc="Generating"):

        row = human.iloc[i]

        try:

            review, model = generate_ai_review(
                row["review"],
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

            save_reviews(ai)

        except Exception as e:

            print(f"\nError at review {i+1}: {e}")

            continue

    print("\n")
    print("=" * 60)
    print("Generation Completed")
    print("=" * 60)
    print(f"Total AI Reviews : {len(ai):,}")
    print(f"Saved to : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()