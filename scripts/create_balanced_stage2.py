import pandas as pd

# ====================================
# Load merged dataset
# ====================================

df = pd.read_csv("data/phase2/stage2_final.csv")

print("Original Shape:", df.shape)

# ====================================
# Split into four categories
# ====================================

human_genuine = df[(df["type"] == "Human") & (df["label"] == 0)]
human_spam    = df[(df["type"] == "Human") & (df["label"] == 1)]
ai_genuine    = df[(df["type"] == "AI") & (df["label"] == 0)]
ai_spam       = df[(df["type"] == "AI") & (df["label"] == 1)]

print("\nBefore Balancing")
print("---------------------------")
print("Human Genuine :", len(human_genuine))
print("Human Spam    :", len(human_spam))
print("AI Genuine    :", len(ai_genuine))
print("AI Spam       :", len(ai_spam))

# ====================================
# Find smallest category
# ====================================

sample_size = min(
    len(human_genuine),
    len(human_spam),
    len(ai_genuine),
    len(ai_spam)
)

print(f"\nUsing {sample_size} samples from each category.")

# ====================================
# Random Sampling
# ====================================

human_genuine = human_genuine.sample(
    n=sample_size,
    random_state=42
)

human_spam = human_spam.sample(
    n=sample_size,
    random_state=42
)

ai_genuine = ai_genuine.sample(
    n=sample_size,
    random_state=42
)

ai_spam = ai_spam.sample(
    n=sample_size,
    random_state=42
)

# ====================================
# Merge
# ====================================

balanced = pd.concat([
    human_genuine,
    human_spam,
    ai_genuine,
    ai_spam
])

# ====================================
# Shuffle
# ====================================

balanced = balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ====================================
# Statistics
# ====================================

print("\nAfter Balancing")
print("---------------------------")

print("\nType Distribution")
print(balanced["type"].value_counts())

print("\nLabel Distribution")
print(balanced["label"].value_counts())

print("\nType + Label Distribution")
print(
    balanced.groupby(["type", "label"]).size()
)

# ====================================
# Save
# ====================================

output = "data/phase2/stage2_balanced.csv"

balanced.to_csv(output, index=False)

print("\nFinal Shape:", balanced.shape)
print(f"\nSaved to: {output}")