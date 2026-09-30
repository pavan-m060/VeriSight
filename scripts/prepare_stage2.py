import os
import glob
import pandas as pd

# ==========================
# Load Human Dataset 1
# ==========================
human1 = pd.read_csv("data/phase2/deceptive_clean.csv")
human1["type"] = "Human"
human1["source"] = "MyleOtt"

# ==========================
# Load Human Dataset 2
# ==========================
human2 = pd.read_csv("data/phase2/yelp_clean.csv")
human2["type"] = "Human"
human2["source"] = "Yelp"

# ==========================
# Read AI Reviews
# ==========================

base = "data/phase2/op_spam_AI-main"

folders = {
    "generated_GPT3_positive": ("GPT3", 0),
    "generated_GPT3_negative": ("GPT3", 1),
    "generated_LLama_positive": ("Llama", 0),
    "generated_LLama_negative": ("Llama", 1)
}

ai_reviews = []

for folder, (source, label) in folders.items():

    path = os.path.join(base, folder)

    files = glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)

    print(f"{folder}: {len(files)} files found")

    for file in files:

        with open(file, encoding="utf-8") as f:
            review = f.read().strip()

        ai_reviews.append({
            "review": review,
            "label": label,
            "type": "AI",
            "source": source
        })

ai = pd.DataFrame(ai_reviews)

# ==========================
# Merge
# ==========================

final = pd.concat(
    [
        human1,
        human2,
        ai
    ],
    ignore_index=True
)

# ==========================
# Cleaning
# ==========================

final = final.dropna(subset=["review"])

final["review"] = final["review"].astype(str).str.strip()

final = final[final["review"] != ""]

final = final.drop_duplicates(subset=["review"])

final = final[final["review"].str.split().str.len() >= 5]

# ==========================
# Statistics
# ==========================

print("\nFinal Shape")
print(final.shape)

print("\nType Distribution")
print(final["type"].value_counts())

print("\nLabel Distribution")
print(final["label"].value_counts())

print("\nSource Distribution")
print(final["source"].value_counts())

# ==========================
# Save
# ==========================

output = "data/phase2/stage2_final.csv"

final.to_csv(output, index=False)

print(f"\nSaved to {output}")