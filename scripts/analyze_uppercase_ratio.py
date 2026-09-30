import numpy as np
import pandas as pd

X = np.load(
    "features/stylometry_stage1_v2.npy"
)

y = np.load(
    "features/y_stylometry_stage1_v2.npy"
)

# Feature index 16 = uppercase_ratio
uppercase_ratio = X[:, 16]

human = uppercase_ratio[y == 0]
ai = uppercase_ratio[y == 1]

print("=" * 70)
print("UPPERCASE RATIO ANALYSIS")
print("=" * 70)

print("\nHuman reviews:")
print("Count :", len(human))
print("Mean  :", human.mean())
print("Median:", np.median(human))
print("Min   :", human.min())
print("Max   :", human.max())

print("\nAI reviews:")
print("Count :", len(ai))
print("Mean  :", ai.mean())
print("Median:", np.median(ai))
print("Min   :", ai.min())
print("Max   :", ai.max())

print("\nDifference:")
print(
    "Mean difference:",
    abs(human.mean() - ai.mean())
)

print("=" * 70)