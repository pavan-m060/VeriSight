import pandas as pd
import os

# Input file
input_file = "data/processed/human_reviews.csv"

# Output folder
output_folder = "data/processed"
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_csv(input_file)

# Take the last 1500 reviews
last_300 = df.tail(300).reset_index(drop=True)

# Split into 3 files of 500 reviews each
for i in range(3):
    start = i * 100
    end = start + 100

    chunk = last_300.iloc[start:end]

    output_file = os.path.join(
        output_folder,
        f"human_reviews_part_{i+1}.csv"
    )

    chunk.to_csv(output_file, index=False)

    print(f"Saved {len(chunk)} reviews -> {output_file}")

print("\nDone!")