# ==========================================================
# VeriSight - Length Balanced Fusion MLP
# 384 MiniLM + 18 Scaled Stylometric = 402 Features
# ==========================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

X_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "X_fused_stage1.npy"
)

Y_FILE = os.path.join(
    BASE_DIR,
    "embeddings",
    "y_fused_stage1.npy"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "length_balanced"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


MODEL_FILE = os.path.join(
    MODEL_DIR,
    "mlp_length_balanced.keras"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "stylometric_scaler.pkl"
)

REPORT_FILE = os.path.join(
    RESULT_DIR,
    "training_report.txt"
)

HISTORY_FILE = os.path.join(
    RESULT_DIR,
    "training_history.csv"
)


# ==========================================================
# Reproducibility
# ==========================================================

RANDOM_STATE = 42

np.random.seed(
    RANDOM_STATE
)

tf.random.set_seed(
    RANDOM_STATE
)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("VeriSight - Length Balanced Fusion MLP")
    print("=" * 70)


    # ======================================================
    # STEP 1 - Load Features
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 1 - Loading Features")
    print("=" * 70)

    X = np.load(
        X_FILE
    )

    y = np.load(
        Y_FILE
    )

    print(
        "\nX Shape:",
        X.shape
    )

    print(
        "Y Shape:",
        y.shape
    )

    print(
        "\nClass Distribution:"
    )

    print(
        np.bincount(y)
    )


    # ======================================================
    # Verify 402 Features
    # ======================================================

    if X.shape[1] != 402:

        raise ValueError(
            f"Expected 402 features, "
            f"got {X.shape[1]}"
        )

    print(
        "\n✓ 402 features verified"
    )


    # ======================================================
    # STEP 2 - Split Dataset
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 2 - Train / Validation / Test Split")
    print("=" * 70)

    X_train, X_temp, y_train, y_temp = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y
    )


    X_val, X_test, y_val, y_test = train_test_split(

        X_temp,
        y_temp,

        test_size=0.50,

        random_state=RANDOM_STATE,

        stratify=y_temp
    )


    print(
        "\nTraining:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Testing:",
        X_test.shape
    )


    # ======================================================
    # STEP 3 - Scale ONLY Stylometric Features
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 3 - Scaling Stylometric Features")
    print("=" * 70)

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # Columns 0-383  = MiniLM
    # Columns 384-401 = Stylometric
    #
    # We scale ONLY the stylometric features.
    #
    # ------------------------------------------------------

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()


    scaler = StandardScaler()


    # Fit ONLY on training data
    X_train_style = scaler.fit_transform(
        X_train[:, 384:]
    )


    # Transform validation/test using same scaler
    X_val_style = scaler.transform(
        X_val[:, 384:]
    )

    X_test_style = scaler.transform(
        X_test[:, 384:]
    )


    # Put scaled features back
    X_train[:, 384:] = X_train_style

    X_val[:, 384:] = X_val_style

    X_test[:, 384:] = X_test_style


    # ------------------------------------------------------
    # Save scaler
    # ------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_FILE
    )


    print(
        "\n✓ Stylometric scaler fitted "
        "on training data only."
    )

    print(
        "Scaler saved:",
        SCALER_FILE
    )


    # ======================================================
    # STEP 4 - Verify
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 4 - Feature Verification")
    print("=" * 70)

    print(
        "\nMiniLM feature count:",
        X_train[:, :384].shape[1]
    )

    print(
        "Stylometric feature count:",
        X_train[:, 384:].shape[1]
    )

    print(
        "Total feature count:",
        X_train.shape[1]
    )


    print(
        "\nScaled stylometric mean:"
    )

    print(
        np.mean(
            X_train[:, 384:],
            axis=0
        ).round(4)
    )


    print(
        "\nScaled stylometric std:"
    )

    print(
        np.std(
            X_train[:, 384:],
            axis=0
        ).round(4)
    )


    # ======================================================
    # STEP 5 - Build MLP
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 5 - Building MLP")
    print("=" * 70)


    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(402,)
        ),

        tf.keras.layers.Dense(
            256,
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.20
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])


    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    model.summary()


    # ======================================================
    # STEP 6 - Callbacks
    # ======================================================

    callbacks = [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(

            MODEL_FILE,

            monitor="val_loss",

            save_best_only=True
        )
    ]


    # ======================================================
    # STEP 7 - Train
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 6 - Training")
    print("=" * 70)


    history = model.fit(

        X_train,

        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=30,

        batch_size=64,

        callbacks=callbacks,

        verbose=1
    )


    # ======================================================
    # STEP 8 - Evaluate
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 7 - Internal Test Evaluation")
    print("=" * 70)


    test_loss, test_accuracy = model.evaluate(

        X_test,

        y_test,

        verbose=0
    )


    print(
        f"\nTest Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


    # ======================================================
    # STEP 9 - Predictions
    # ======================================================

    probabilities = model.predict(

        X_test,

        batch_size=64,

        verbose=1
    ).ravel()


    predictions = (
        probabilities >= 0.5
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    # ======================================================
    # STEP 10 - Classification Report
    # ======================================================

    report = classification_report(

        y_test,

        predictions,

        labels=[
            0,
            1
        ],

        target_names=[
            "Human",
            "AI"
        ],

        digits=4,

        zero_division=0
    )


    print(
        "\nClassification Report:"
    )

    print(
        report
    )


    # ======================================================
    # STEP 11 - Confusion Matrix
    # ======================================================

    cm = confusion_matrix(

        y_test,

        predictions,

        labels=[
            0,
            1
        ]
    )


    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )


    # ======================================================
    # STEP 12 - Save History
    # ======================================================

    history_df = pd.DataFrame(
        history.history
    )

    history_df.to_csv(

        HISTORY_FILE,

        index=False
    )


    # ======================================================
    # STEP 13 - Save Report
    # ======================================================

    with open(

        REPORT_FILE,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "VeriSight - Length Balanced Fusion MLP\n"
        )

        f.write(
            "=" * 65 + "\n\n"
        )

        f.write(
            "Features:\n"
        )

        f.write(
            "384 MiniLM + 18 Stylometric = 402\n\n"
        )

        f.write(
            "Dataset:\n"
        )

        f.write(
            "Length-balanced Human/AI dataset\n\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{accuracy:.4f}\n"
        )

        f.write(
            f"Test Accuracy (%): "
            f"{accuracy * 100:.2f}%\n\n"
        )

        f.write(
            "Classification Report:\n"
        )

        f.write(
            report
        )

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            str(cm)
        )


    # ======================================================
    # Final
    # ======================================================

    print("\n" + "=" * 70)
    print("LENGTH BALANCED MLP TRAINING COMPLETED")
    print("=" * 70)

    print(
        "\nModel:"
    )

    print(
        MODEL_FILE
    )

    print(
        "\nScaler:"
    )

    print(
        SCALER_FILE
    )

    print(
        "\nReport:"
    )

    print(
        REPORT_FILE
    )

    print(
        "\nHistory:"
    )

    print(
        HISTORY_FILE
    )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()