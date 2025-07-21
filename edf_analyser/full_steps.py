import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.decoding import CSP
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ------------ Step 1: Load and Prepare Data ----------------
edf_path = r"E:\AMAR\ROBOARM\DATASET\files\S001\S001R03.edf"  # Update your file path
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Standardize channel names and montage
raw.rename_channels(lambda ch: ch.strip('.').upper())
raw.set_montage("standard_1020", on_missing='ignore')

# ------------ Step 2: Filtering ----------------
raw.filter(7., 30., fir_design='firwin')  # Bandpass filter
raw.notch_filter(freqs=50)  # Optional: Notch filter at 50 Hz

# ------------ Step 3: Remove Artifacts with ICA ----------------
# ------------ Step 3: ICA for Artifact Removal ----------------

# Exclude channels with known layout overlap
bad_overlap_chs = ['FCZ', 'CZ', 'CPZ', 'FP1', 'FPZ', 'FP2', 'AFZ', 'FZ', 'PZ', 'POZ', 'OZ', 'IZ']
existing_bad_chs = [ch for ch in bad_overlap_chs if ch in raw.info['ch_names']]
raw.info['bads'].extend(existing_bad_chs)

ica = ICA(n_components=15, random_state=42)
ica.fit(raw)

# Plot ICA with layout-safe info
ica.plot_components(inst=raw)

# Remove EOG artifacts
eog_inds, _ = ica.find_bads_eog(raw)
ica.exclude = eog_inds
ica.apply(raw)

# ------------ Step 4: Pick EEG Channels Only ----------------
raw.pick_types(eeg=True)

# ------------ Step 5: Create Events ----------------
events, _ = mne.events_from_annotations(raw)

if len(events) < 2:
    raise RuntimeError("Not enough events to simulate left/right classes.")

# Simulate left (1) and right (2) class from a single type (e.g., 'T0')
mid = len(events) // 2
events[:mid, 2] = 1  # Left-hand MI
events[mid:, 2] = 2  # Right-hand MI
event_id = {'left': 1, 'right': 2}

# ------------ Step 6: Epoching ----------------
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=4,
                    baseline=None, picks='eeg', preload=True)
epochs.plot(title="Motor Imagery Epochs")


X = epochs.get_data()  
y = epochs.events[:, -1] - 1  

csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
X_csp = csp.fit_transform(X, y)
print("✅ CSP feature shape:", X_csp.shape)

df = pd.DataFrame(X_csp, columns=[f'CSP_{i+1}' for i in range(X_csp.shape[1])])
df['label'] = y
df.to_csv("csp_features.csv", index=False)
print("✅ Saved CSP features to csp_features.csv")

csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (uV)', size=1.5)
csp.plot_filters(epochs.info, ch_type='eeg', units='Filters (AU)', size=1.5)
plt.suptitle('CSP Spatial Patterns')

lda = LinearDiscriminantAnalysis()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lda, X_csp, y, cv=cv, scoring='accuracy')
print(f"🎯 CSP + LDA Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

raw.plot(n_channels=10, title="EEG (Post-ICA)")
raw.plot_psd(fmin=2, fmax=40, average=True)
plt.show()
