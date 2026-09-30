import pandas as pd

# -----------------------------
# Load Yelp Dataset
# -----------------------------
file_path = "data/phase2/Labelled Yelp Dataset.csv"

df = pd.read_csv(file_path)

print("Original Shape:", df.shape)
print(df.columns.tolist())

# -----------------------------
# Rename columns
# -----------------------------
df = df.rename(columns={
    "Review": "review",
    "Label": "label"
})

# -----------------------------
# Convert labels
# Dataset:
#  1  -> Genuine
# -1 -> Spam
# Our labels:
#  0 -> Genuine
#  1 -> Spam
# -----------------------------
df["label"] = df["label"].replace({
    1: 0,
    -1: 1
})

# -----------------------------
# Keep only required columns
# -----------------------------
df = df[["review", "label"]]

# -----------------------------
# Remove empty reviews
# -----------------------------
df = df.dropna(subset=["review"])
df["review"] = df["review"].astype(str).str.strip()
df = df[df["review"] != ""]

# -----------------------------
# Remove duplicates
# -----------------------------
df = df.drop_duplicates(subset=["review"])

# -----------------------------
# Remove very short reviews
# -----------------------------
df = df[df["review"].str.split().str.len() >= 5]

# -----------------------------
# Show class distribution
# -----------------------------
print("\nClass Distribution:")
print(df["label"].value_counts())

# -----------------------------
# Save
# -----------------------------
output_file = "data/phase2/yelp_clean.csv"
df.to_csv(output_file, index=False)

print("\nCleaning Completed!")
print("Final Shape:", df.shape)
print(f"Saved to: {output_file}")