import mne
from mne.decoding import CSP
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Load EEG Data ---
file_path = r".\..\S001R04.edf"
if not os.path.exists(file_path):
    raise FileNotFoundError("EDF file not found. Place it in the current directory.")

raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
raw.rename_channels(lambda x: x.strip('.').upper())
raw.set_montage('standard_1020', on_missing='ignore')
sfreq = raw.info['sfreq']

# --- 2. Bandpass Filtering ---
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

# Plot PSD (this is safe)
raw.plot_psd(fmin=2, fmax=40, average=True)
plt.savefig("filtered_psd.png")
plt.close()

# --- 3. ICA for Artifact Removal (EOG only) ---
# -------- ICA --------
ica = ICA(n_components=15, random_state=42, method='fastica')
ica.fit(raw)

# -------- Find EOG (Auto) --------
try:
    eog_epochs = create_eog_epochs(raw, reject_by_annotation=True)
    eog_inds, _ = ica.find_bads_eog(eog_epochs)
    if len(eog_inds) > 0:
        print("EOG Components:", eog_inds)
        ica.exclude.extend(eog_inds)
except Exception as e:
    print("No EOG removal:", str(e))

# -------- Find ECG (Optional) --------
try:
    ecg_epochs = create_ecg_epochs(raw, reject_by_annotation=True)
    ecg_inds, _ = ica.find_bads_ecg(ecg_epochs)
    if len(ecg_inds) > 0:
        print("ECG Components:", ecg_inds)
        ica.exclude.extend(ecg_inds)
except Exception as e:
    print("No ECG removal:", str(e))

# -------- Apply ICA --------
ica.apply(raw)

# -------- Save Overlay Plot --------
fig = ica.plot_overlay(raw, exclude=ica.exclude, show=True)
if not os.path.exists("figures"):
    os.makedirs("figures")
fig.savefig("figures/ica_overlay.png", dpi=300)

# --- 4. Event Extraction and Epoching ---
events, event_id = mne.events_from_annotations(raw)
class_ids = {'left_fist': event_id['T1'], 'right_fist': event_id['T2']}

epochs = mne.Epochs(raw, events, event_id=class_ids, tmin=0., tmax=4.,
                    picks='eeg', baseline=None, preload=True)

# Plot average of epochs (mean across trials)
epochs.plot_image(combine='mean')
plt.savefig("epoch_average.png")
plt.close()

# --- 5. CSP Feature Extraction ---
X = epochs.get_data()
y = epochs.events[:, -1]

csp = CSP(n_components=4, log=True)
X_csp = csp.fit_transform(X, y)

# Visualize CSP Patterns
try:
    csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns', size=1.5)
    plt.savefig("csp_patterns.png")
    plt.close()
except ValueError as e:
    print("CSP pattern visualization skipped due to error:", e)

# --- 6. Save Features for SNN ---
np.save("csp_features.npy", X_csp)
np.save("csp_labels.npy", y)
np.savetxt("csp_features.csv", X_csp, delimiter=",")
np.savetxt("csp_labels.csv", y, delimiter=",")

print("Pipeline complete.")
print(f"CSP features shape: {X_csp.shape}")
print("Saved: 'csp_features.npy' and 'csp_labels.npy'")
