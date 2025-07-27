import numpy as np
import os
import glob

features_dir = "./features"
labels_dir = "./labels"

X_all = []
y_all = []

# Match features/*.npy and find corresponding labels/*.npy
for feature_path in glob.glob(os.path.join(features_dir, "*.npy")):
    base_name = os.path.basename(feature_path).replace("_features.npy", "").replace(".npy", "")
    label_path = os.path.join(labels_dir, base_name + "_labels.npy")
    
    if not os.path.exists(label_path):
        print(f"❌ Missing label file for: {base_name}")
        continue

    X = np.load(feature_path)
    y = np.load(label_path)

    X_all.append(X)
    y_all.append(y)

# Check if anything was loaded
if len(X_all) == 0:
    raise ValueError("No features loaded. Check file paths and naming.")

# Combine all subjects
X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print(f"✅ Combined X shape: {X_all.shape}, y shape: {y_all.shape}")
