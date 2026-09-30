# ==========================================================
# VeriSight - Generate Fused Features
# MiniLM Embeddings + Stylometric Features
# ==========================================================

import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from stylometric_features import extract_stylometric_features


# ==========================================================
# Paths
# ==========================================================

from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Paths
# ==========================================================

DATA_FILE = PROJECT_ROOT / "data" / "stage1" / "stage1_length_balanced.csv"

OUTPUT_DIR = PROJECT_ROOT / "embeddings"

X_OUTPUT = OUTPUT_DIR / "X_fused_length_balanced.npy"
Y_OUTPUT = OUTPUT_DIR / "y_fused_length_balanced.npy"
STYLE_OUTPUT = OUTPUT_DIR / "stylometric_length_balanced.csv"

X_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "X_fused_stage1.npy"
)

Y_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "y_fused_stage1.npy"
)

STYLE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stylometric_stage1.csv"
)


# ==========================================================
# Create Output Directory
# ==========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("VeriSight - Fused Feature Generation")
    print("=" * 70)

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_FILE)

    print(
        f"Dataset Shape: {df.shape}"
    )

    print("\nClass Distribution:")

    print(
        df["label"].value_counts()
    )

    # ------------------------------------------------------
    # Prepare Reviews
    # ------------------------------------------------------

    reviews = (
        df["review"]
        .astype(str)
        .tolist()
    )

    labels = (
        df["label"]
        .astype(int)
        .values
    )

    # ======================================================
    # STEP 1
    # Generate MiniLM Embeddings
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 1 - MiniLM Embeddings")
    print("=" * 70)

    print("\nLoading SentenceTransformer...")

    encoder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("\nGenerating embeddings...")

    X_semantic = encoder.encode(
        reviews,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(
        "\nMiniLM Shape:",
        X_semantic.shape
    )

    # ======================================================
    # STEP 2
    # Generate Stylometric Features
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 2 - Stylometric Features")
    print("=" * 70)

    style_features = []

    for i, review in enumerate(reviews):

        features = extract_stylometric_features(
            review
        )

        style_features.append(features)

        if (i + 1) % 1000 == 0:

            print(
                f"Processed "
                f"{i + 1:,} / {len(reviews):,}"
            )

    # ------------------------------------------------------
    # Convert to DataFrame
    # ------------------------------------------------------

    style_df = pd.DataFrame(
        style_features
    )

    print(
        "\nStylometric Shape:",
        style_df.shape
    )

    # ------------------------------------------------------
    # Convert to NumPy
    # ------------------------------------------------------

    X_style = style_df.values.astype(
        np.float32
    )

    print(
        "Stylometric NumPy Shape:",
        X_style.shape
    )

    # ======================================================
    # STEP 3
    # Combine Features
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 3 - Feature Fusion")
    print("=" * 70)

    X_semantic = X_semantic.astype(
        np.float32
    )

    # ------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------

    X_fused = np.concatenate(
        [
            X_semantic,
            X_style
        ],
        axis=1
    )

    print(
        "\nMiniLM Features     :",
        X_semantic.shape[1]
    )

    print(
        "Stylometric Features:",
        X_style.shape[1]
    )

    print(
        "Total Features      :",
        X_fused.shape[1]
    )

    # ======================================================
    # STEP 4
    # Validation
    # ======================================================

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        "\nReviews:",
        len(reviews)
    )

    print(
        "Labels:",
        len(labels)
    )

    print(
        "Fused Samples:",
        X_fused.shape[0]
    )

    print(
        "Fused Features:",
        X_fused.shape[1]
    )

    # ------------------------------------------------------
    # Check NaN
    # ------------------------------------------------------

    print(
        "\nNaN values:",
        np.isnan(X_fused).sum()
    )

    # ------------------------------------------------------
    # Check Infinite
    # ------------------------------------------------------

    print(
        "Infinite values:",
        np.isinf(X_fused).sum()
    )

    # ======================================================
    # STEP 5
    # Save
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 5 - Saving")
    print("=" * 70)

    np.save(
        X_OUTPUT,
        X_fused
    )

    np.save(
        Y_OUTPUT,
        labels
    )

    style_df.to_csv(
        STYLE_OUTPUT,
        index=False
    )

    print(
        "\nSaved fused features:"
    )

    print(
        X_OUTPUT
    )

    print(
        "\nSaved labels:"
    )

    print(
        Y_OUTPUT
    )

    print(
        "\nSaved stylometric features:"
    )

    print(
        STYLE_OUTPUT
    )

    # ======================================================
    # Final Summary
    # ======================================================

    print("\n" + "=" * 70)
    print("FUSED FEATURE GENERATION COMPLETED")
    print("=" * 70)

    print(
        f"\nFinal X Shape: {X_fused.shape}"
    )

    print(
        f"Final Y Shape: {labels.shape}"
    )

    print(
        "\nExpected:"
    )

    print(
        "384 MiniLM + 18 Stylometric = 402 features"
    )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()