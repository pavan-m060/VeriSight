import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

X_FILE = "embeddings/X_stage1_v2.npy"
Y_FILE = "embeddings/y_stage1_v2.npy"

OUTPUT_DIR = Path("embeddings/stage1_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load
# ==========================================================

print("=" * 60)
print("Creating Stage 1 Train/Validation/Test Split")
print("=" * 60)

X = np.load(X_FILE)
y = np.load(Y_FILE)

print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# ==========================================================
# First split: 80% train, 20% temporary
# ==========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Second split: 10% validation, 10% test
# ==========================================================

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# ==========================================================
# Save
# ==========================================================

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "y_train.npy", y_train)

np.save(OUTPUT_DIR / "X_val.npy", X_val)
np.save(OUTPUT_DIR / "y_val.npy", y_val)

np.save(OUTPUT_DIR / "X_test.npy", X_test)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

# ==========================================================
# Results
# ==========================================================

print("\n" + "=" * 60)
print("Split Completed")
print("=" * 60)

print("\nTraining:")
print("Shape:", X_train.shape)
print("Labels:", np.bincount(y_train))

print("\nValidation:")
print("Shape:", X_val.shape)
print("Labels:", np.bincount(y_val))

print("\nTesting:")
print("Shape:", X_test.shape)
print("Labels:", np.bincount(y_test))

print("\nSaved to:")
print(OUTPUT_DIR)