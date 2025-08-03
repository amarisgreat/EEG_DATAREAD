import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


def visualize_features(features_path, labels_path, method='tsne'):
    # Load the feature and label data
    X = np.load(features_path)
    y = np.load(labels_path)

    print(f"Loaded: {os.path.basename(features_path)}")
    print(f"Feature shape: {X.shape}, Labels: {np.unique(y)}")

    # Reduce dimensionality to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        perplexity = min(10, len(X) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    X_reduced = reducer.fit_transform(X)

    # Plot
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], label=f"Class {label}", alpha=0.7)

    plt.title(f"{method.upper()} Visualization of EEG Features")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === Example usage ===
if __name__ == "__main__":
    base_filename = "S001R04"  # Change this to any filename from your RECORDS list
    features_dir = "preprocessed_data/features"
    labels_dir = "preprocessed_data_1005/labels"

    features_path = os.path.join(features_dir, f"{base_filename}_features.npy")
    labels_path = os.path.join(labels_dir, f"{base_filename}_labels.npy")

    if os.path.exists(features_path) and os.path.exists(labels_path):
        visualize_features(features_path, labels_path, method='tsne')  # or method='tsne'
    else:
        print(f"Missing files: {features_path} or {labels_path}")
