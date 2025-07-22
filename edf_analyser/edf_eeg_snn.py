import mne
from mne.decoding import CSP
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Load EEG Data ---
file_path = r".\steps\S001R04.edf"
if not os.path.exists(file_path):
    raise FileNotFoundError("EDF file not found. Place it in the current directory.")

raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
raw.rename_channels(lambda x: x.strip('.').upper())
raw.set_montage('standard_1020', on_missing='ignore')
sfreq = raw.info['sfreq']

# --- 2. Bandpass Filtering ---
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
raw.plot_psd(fmin=2, fmax=40, average=True)
plt.savefig("filtered_psd.png")
plt.close()

# --- 3. ICA Artifact Removal ---
ica = ICA(n_components=15, random_state=42, method='fastica')
ica.fit(raw)

try:
    eog_epochs = create_eog_epochs(raw)
    eog_inds, _ = ica.find_bads_eog(eog_epochs)
    ica.exclude.extend(eog_inds)
except Exception:
    print("No EOG artifacts detected")

try:
    ecg_epochs = create_ecg_epochs(raw)
    ecg_inds, _ = ica.find_bads_ecg(ecg_epochs)
    ica.exclude.extend(ecg_inds)
except Exception:
    print("No ECG artifacts detected")

ica.apply(raw)
fig = ica.plot_overlay(raw, exclude=ica.exclude, show=True)
if not os.path.exists("figures"):
    os.makedirs("figures")
fig.savefig("figures/ica_overlay.png", dpi=300)

# --- 4. Event Extraction ---
events, event_id = mne.events_from_annotations(raw)
class_ids = {'left_fist': event_id['T1'], 'right_fist': event_id['T2']}

epochs = mne.Epochs(raw, events, event_id=class_ids, tmin=-1., tmax=4.,
                    picks='eeg', baseline=(None, 0), preload=True)

labels = epochs.events[:, -1]

# --- 5. CSP Feature Extraction ---
X = epochs.get_data()
csp = CSP(n_components=4, log=True)
X_csp = csp.fit_transform(X, labels)

# --- 6. ERD/ERS Feature Extraction ---
# Use log ratio of band power between baseline and task period
bands = {'alpha': (8, 12), 'beta': (13, 30)}
X_erd = []

for band_name, (fmin, fmax) in bands.items():
    band_epochs = epochs.copy().filter(fmin, fmax, fir_design='firwin')
    data = band_epochs.get_data()

    baseline_power = np.mean(data[:, :, :int(1*sfreq)] ** 2, axis=2)  # −1 to 0 sec
    task_power = np.mean(data[:, :, int(1*sfreq):] ** 2, axis=2)      # 0 to 4 sec

    erd = 10 * np.log10(task_power / baseline_power)
    X_erd.append(erd)

X_erd_combined = np.concatenate(X_erd, axis=1)  # Combine alpha and beta across channels

# --- 7. Combine CSP + ERD/ERS ---
X_combined = np.concatenate([X_csp, X_erd_combined], axis=1)

# --- 8. Save Features ---
np.save("features_combined.npy", X_combined)
np.save("labels.npy", labels)
np.savetxt("features_combined.csv", X_combined, delimiter=",")
np.savetxt("labels.csv", labels, delimiter=",")

print("✅ Pipeline complete.")
print(f"Feature shape (CSP + ERD/ERS): {X_combined.shape}")
