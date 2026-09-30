#!/usr/bin/env python3
"""
Generate a labeled synthetic-review research dataset from human-written reviews.

For each row in the input CSV, calls the Claude API to produce a synthetic
review with matching sentiment/rating but different wording, then writes out
a new CSV with two extra columns: generator_model, is_synthetic.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python generate_synthetic_reviews.py input.csv output.csv

Requirements:
    pip install anthropic --break-system-packages
"""

import csv
import os
import sys
import time
import random
import argparse

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency. Run: pip install anthropic --break-system-packages")

MODEL = "claude-sonnet-4-6"

# A few different instruction phrasings + temperatures, rotated across rows,
# so the synthetic set isn't stylistically uniform (important for training
# a *detector* -- you don't want every synthetic example to share one tell).
PROMPT_VARIANTS = [
    "Write a natural, casual review a different customer might post.",
    "Write the review the way a slightly more formal reviewer would phrase it.",
    "Write a brief, to-the-point version of this kind of review.",
    "Write it the way someone rushing and typing on their phone might.",
    "Write a more detailed, narrative-style version of this kind of review.",
]

SYSTEM_PROMPT = (
    "You generate synthetic product reviews for an academic research dataset "
    "used to train AI-generated-text detectors. Given a reference review and "
    "its star rating, write a NEW review that:\n"
    "- Expresses the same overall sentiment and rating\n"
    "- Is written in different words and sentence structure than the original\n"
    "- Does not copy phrases or sentences from the original\n"
    "- Is roughly the same length as the original\n"
    "- Reads like an authentic, naturally written customer review\n"
    "Return ONLY the review text, nothing else -- no preamble, no quotes, no labels."
)


def generate_one(client, review, rating, category, retries=3):
    variant = random.choice(PROMPT_VARIANTS)
    temperature = round(random.uniform(0.7, 1.0), 2)

    user_prompt = (
        f"Reference review (category: {category}, rating: {rating}/5):\n"
        f"\"{review}\"\n\n"
        f"{variant}"
    )

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=600,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(
                block.text for block in resp.content if block.type == "text"
            ).strip()
            if text:
                return text
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt+1}/{retries} after error: {e} (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError("Failed to generate review after retries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--start", type=int, default=0, help="row index to resume from")
    parser.add_argument("--limit", type=int, default=None, help="max rows to process")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    client = anthropic.Anthropic(api_key=api_key)

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["generator_model", "is_synthetic"]
        rows = list(reader)

    if args.limit:
        rows = rows[args.start:args.start + args.limit]
    else:
        rows = rows[args.start:]

    write_header = not (args.start and os.path.exists(args.output_csv))
    mode = "a" if (args.start and os.path.exists(args.output_csv)) else "w"

    with open(args.output_csv, mode, newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(rows, start=args.start):
            original = row["review"]
            rating = row.get("rating", "")
            category = row.get("category", row.get("source", ""))

            print(f"[{i+1}] generating...")
            synthetic = generate_one(client, original, rating, category)

            new_row = dict(row)
            new_row["review"] = synthetic
            new_row["generator_model"] = "Claude"
            new_row["is_synthetic"] = "Yes"

            writer.writerow(new_row)
            out_f.flush()

            time.sleep(0.3)  # gentle rate limiting

    print(f"Done. Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()