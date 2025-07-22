import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- 1. Load Combined Features and Labels ---
features = pd.read_csv(r"E:\AMAR\git\EEG_DATAREAD\edf_analyser\steps\second\csp_features.csv", header=None).values
labels = pd.read_csv(r"E:\AMAR\git\EEG_DATAREAD\edf_analyser\steps\second\csp_labels.csv", header=None).values.ravel()

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# --- 2. Normalize Features ---
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --- 3. Choose Perplexity Based on Sample Size ---
n_samples = features.shape[0]
perplexity = min(30, (n_samples - 1) // 3)  # Rule of thumb: perplexity < n_samples / 3

# --- 4. t-SNE Projection ---
tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
features_tsne = tsne.fit_transform(features_scaled)

# --- 5. Plotting ---
plt.figure(figsize=(10, 7))
unique_labels = np.unique(labels)
colors = ['blue', 'orange', 'green', 'red', 'purple']

for i, label in enumerate(unique_labels):
    idx = labels == label
    plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1],
                label=f'Class {label}', alpha=0.7, s=60, color=colors[i % len(colors)])

plt.title("t-SNE of Combined CSP + ERD/ERS Features")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_combined_features.png", dpi=300)
plt.show()
