import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def visualize_npy_features(features_path, labels_path, method='pca'):
    # Load the data
    X = np.load(features_path)
    y = np.load(labels_path)

    print(f"Loaded {features_path} with shape {X.shape}")
    print(f"Labels: {np.unique(y)}")

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    X_reduced = reducer.fit_transform(X)

    # Plotting
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], label=f"Class {label}", alpha=0.7)
    plt.title(f"Feature Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    base_filename = "."  # Change based on file you're testing
    feature_file = os.path.join("preprocessed_data_fbcsp", "features", f"{base_filename}_features.npy")
    label_file = os.path.join("preprocessed_data_fbcsp", "labels", f"{base_filename}_labels.npy")

    visualize_npy_features(feature_file, label_file, method='pca')  # or method='tsne'
