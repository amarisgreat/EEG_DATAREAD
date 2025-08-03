import os
import numpy as np
import scipy.io
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score

# --------- PSO Optimization ---------
def pso_optimize(X, y, n_particles=10, n_iterations=15):
    def fitness(params):
        C, gamma = params
        clf = SVC(kernel='rbf', C=C, gamma=gamma)
        scores = cross_val_score(clf, X, y, cv=5)
        return np.mean(scores)

    dim = 2
    bounds = [(0.1, 100), (0.001, 10)]
    particles = np.random.rand(n_particles, dim)
    for i in range(dim):
        particles[:, i] = particles[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    velocities = np.zeros_like(particles)
    personal_best = particles.copy()
    personal_best_scores = np.array([fitness(p) for p in particles])
    global_best = personal_best[np.argmax(personal_best_scores)]
    w, c1, c2 = 0.5, 1.5, 1.5

    for _ in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best[i] - particles[i]) + c2 * r2 * (global_best - particles[i])
            particles[i] += velocities[i]
            for d in range(dim):
                particles[i, d] = np.clip(particles[i, d], bounds[d][0], bounds[d][1])
            score = fitness(particles[i])
            if score > personal_best_scores[i]:
                personal_best[i] = particles[i]
                personal_best_scores[i] = score
        global_best = personal_best[np.argmax(personal_best_scores)]
    return global_best

# --------- EEG Processing ---------
data_dir = r"E:\AMAR\ROBOARM\DATASET\BCI Competition  IV\BCICIV_4_mat\DATASET_IVa\100Hz"
subjects = ["aa", "al", "av", "aw", "ay"]
sfreq = 100
tmin, tmax = 0, 3.5
use_channels = 22

results = []
os.makedirs("figures", exist_ok=True)


def load_data(subject):
    mat = scipy.io.loadmat(os.path.join(data_dir, f"data_set_IVa_{subject}.mat"))
    X = mat['cnt'] * 0.1e-6
    y = mat['mrk'][0][0]['y'][0]
    pos = mat['mrk'][0][0]['pos'][0]
    channel_names = [str(ch[0]) for ch in mat['nfo']['clab'][0][0][0]]
    ch_types = ['eeg'] * len(channel_names)
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(X.T, info, verbose=False)
    return raw, y, pos


def create_epochs(raw, y, pos):
    X, labels, test_indices = [], [], []
    samples = int((tmax - tmin) * sfreq)
    for i, (label, p) in enumerate(zip(y, pos)):
        start = p + int(tmin * sfreq)
        stop = start + samples
        if stop < raw.n_times:
            data, _ = raw[:, start:stop]
            X.append(data[:use_channels])
            labels.append(label)
            test_indices.append(i)
    return np.array(X), np.array(labels), np.array(test_indices)


def visualize_signal(raw, subject):
    raw.plot(n_channels=10, duration=10.0, title=f"Raw EEG Signal - Subject {subject}", show=False)
    plt.savefig(f"figures/raw_signal_{subject}.png")
    plt.close()


def run_pipeline(subject):
    raw, y, pos = load_data(subject)
    visualize_signal(raw, subject)
    X, labels, indices = create_epochs(raw, y, pos)

    is_train = ~np.isnan(labels)
    y_train = labels[is_train].astype(int)
    X_train = X[is_train]

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train.reshape(len(X_train), -1))
    X_train = X_train_flat.reshape(len(X_train), X.shape[1], X.shape[2])

    csp = CSP(n_components=4, log=True)
    X_train_csp = csp.fit_transform(X_train, y_train)

    best_params = pso_optimize(X_train_csp, y_train)
    clf = SVC(kernel='rbf', C=best_params[0], gamma=best_params[1])

    # Validation-based accuracy (since test labels are unavailable)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_csp, y_train, test_size=0.3, random_state=42)
    clf.fit(X_train_split, y_train_split)
    y_pred_val = clf.predict(X_val_split)
    acc = np.mean(y_pred_val == y_val_split)

    results.append({
        "Subject": subject,
        "Accuracy (%)": acc * 100,
        "C": best_params[0],
        "Gamma": best_params[1]
    })

    # Save predictions for validation set
    df_pred = pd.DataFrame({
        "Trial_Index": np.arange(len(y_val_split)),
        "Prediction": y_pred_val,
        "True_Label": y_val_split
    })
    df_pred.to_csv(f"result_IVa_{subject}.csv", index=False)

    cm = confusion_matrix(y_val_split, y_pred_val)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Right Hand", "Foot"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {subject}")
    plt.savefig(f"figures/confusion_matrix_{subject}.png")
    plt.close()

    print(f"[{subject}] Accuracy: {acc*100:.2f}% | C={best_params[0]:.3f}, Gamma={best_params[1]:.3f}")


# --------- Run All Subjects ---------
if __name__ == "__main__":
    for subj in subjects:
        run_pipeline(subj)

    df = pd.DataFrame(results)
    df.to_csv("bci_iv_summary.csv", index=False)

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Subject", y="Accuracy (%)", data=df, palette="viridis")
    plt.ylim(0, 100)
    plt.title("Classification Accuracy per Subject")
    plt.ylabel("Accuracy (%)")
    plt.savefig("figures/subject_accuracy_plot.png")
    plt.show()
