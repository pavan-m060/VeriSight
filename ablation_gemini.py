# ==========================================================
# VeriSight - Gemini Ablation Study
#
# Compare:
# 1. MiniLM only
# 2. Stylometry only
# 3. Fusion (MiniLM + Stylometry)
#
# Same 640 Human + Gemini external dataset
# ==========================================================

import os
import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from features.stylometric_features import (
    extract_stylometric_features
)


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "external",
    "gemini_unseen_balanced.csv"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "external",
    "ablation"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("VeriSight - Gemini Ablation Study")
    print("=" * 70)


    # ======================================================
    # STEP 1 - Load Dataset
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 1 - Loading External Dataset")
    print("=" * 70)

    df = pd.read_csv(
        DATA_FILE
    )

    texts = (
        df["review"]
        .astype(str)
        .tolist()
    )

    y = (
        df["label"]
        .astype(int)
        .values
    )

    print(
        "\nDataset Shape:",
        df.shape
    )

    print(
        "\nLabels:"
    )

    print(
        np.bincount(y)
    )


    # ======================================================
    # STEP 2 - MiniLM
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 2 - MiniLM Embeddings")
    print("=" * 70)

    encoder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    X_minilm = encoder.encode(

        texts,

        batch_size=32,

        show_progress_bar=True,

        convert_to_numpy=True
    )

    X_minilm = X_minilm.astype(
        np.float32
    )

    print(
        "\nMiniLM Shape:",
        X_minilm.shape
    )


    # ======================================================
    # STEP 3 - Stylometry
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 3 - Stylometric Features")
    print("=" * 70)

    style_features = []

    for i, review in enumerate(texts):

        features = (
            extract_stylometric_features(
                review
            )
        )

        style_features.append(
            features
        )

        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1} / "
                f"{len(texts)}"
            )


    style_df = pd.DataFrame(
        style_features
    )

    X_style = style_df.values.astype(
        np.float32
    )

    print(
        "\nStylometry Shape:",
        X_style.shape
    )


    # ======================================================
    # STEP 4 - Scale Stylometry
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 4 - Scaling Stylometry")
    print("=" * 70)

    scaler = StandardScaler()

    X_style_scaled = (
        scaler.fit_transform(
            X_style
        )
    )

    print(
        "\n✓ Stylometric features scaled."
    )


    # ======================================================
    # Helper Function
    # ======================================================

    def run_experiment(
        name,
        X,
        description
    ):

        print("\n" + "=" * 70)

        print(
            f"EXPERIMENT: {name}"
        )

        print("=" * 70)

        print(
            f"\nFeatures: "
            f"{X.shape[1]}"
        )

        print(
            f"Description: "
            f"{description}"
        )


        # --------------------------------------------------
        # IMPORTANT:
        #
        # This is NOT training on Gemini.
        #
        # This experiment trains a classifier using the
        # external dataset only to measure whether the
        # representation itself contains separable signal.
        #
        # Therefore this is NOT a generalization result.
        # --------------------------------------------------

        classifier = MLPClassifier(

            hidden_layer_sizes=(
                128,
                64
            ),

            activation="relu",

            solver="adam",

            learning_rate_init=0.001,

            batch_size=32,

            max_iter=100,

            random_state=42,

            early_stopping=True,

            validation_fraction=0.2,

            n_iter_no_change=8
        )


        classifier.fit(
            X,
            y
        )


        predictions = (
            classifier.predict(X)
        )


        probabilities = (
            classifier.predict_proba(X)[:, 1]
        )


        accuracy = accuracy_score(
            y,
            predictions
        )


        cm = confusion_matrix(
            y,
            predictions,
            labels=[
                0,
                1
            ]
        )


        print(
            f"\nAccuracy: "
            f"{accuracy * 100:.2f}%"
        )

        print(
            "\nClassification Report:"
        )

        print(
            classification_report(

                y,

                predictions,

                labels=[
                    0,
                    1
                ],

                target_names=[
                    "Human",
                    "Gemini"
                ],

                digits=4,

                zero_division=0
            )
        )

        print(
            "Confusion Matrix:"
        )

        print(
            cm
        )


        return {
            "name": name,
            "features": X.shape[1],
            "accuracy": accuracy,
            "mean_human_probability":
                probabilities[y == 0].mean(),
            "mean_gemini_probability":
                probabilities[y == 1].mean()
        }


    # ======================================================
    # EXPERIMENT 1 - MiniLM ONLY
    # ======================================================

    result_minilm = run_experiment(

        "MiniLM Only",

        X_minilm,

        "384 semantic embedding features"
    )


    # ======================================================
    # EXPERIMENT 2 - Stylometry ONLY
    # ======================================================

    result_style = run_experiment(

        "Stylometry Only",

        X_style_scaled,

        "18 scaled stylometric features"
    )


    # ======================================================
    # EXPERIMENT 3 - FUSION
    # ======================================================

    X_fusion = np.concatenate(

        [
            X_minilm,
            X_style_scaled
        ],

        axis=1
    )

    result_fusion = run_experiment(

        "MiniLM + Stylometry",

        X_fusion,

        "384 MiniLM + 18 stylometric features"
    )


    # ======================================================
    # Summary
    # ======================================================

    results = pd.DataFrame([

        result_minilm,

        result_style,

        result_fusion

    ])


    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )


    # ======================================================
    # Save
    # ======================================================

    output_file = os.path.join(

        RESULT_DIR,

        "ablation_results.csv"
    )

    results.to_csv(

        output_file,

        index=False
    )


    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETED")
    print("=" * 70)

    print(
        "\nResults saved:"
    )

    print(
        output_file
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These three experiments measure "
        "representation separability."
    )

    print(
        "They are NOT the final unseen-generator "
        "generalization results."
    )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()