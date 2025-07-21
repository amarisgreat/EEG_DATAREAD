import mne
from mne.decoding import CSP
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# ----------- STEP 1: Load EEG EDF File -----------
file_path = r'E:\AMAR\ROBOARM\DATASET\files\S001\S001R03.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# ----------- STEP 2: Channel Standardization -----------
raw.rename_channels(lambda x: x.strip('.').upper())
raw.set_montage('standard_1020', on_missing='ignore')

# ----------- STEP 3: Filtering 7–30 Hz -----------
raw_filtered = raw.copy().filter(7, 30, fir_design='firwin')

# ----------- STEP 4: Mark Bad Channels -----------
bad_overlap_chs = ['FCZ', 'CZ', 'CPZ', 'FP1', 'FPZ', 'FP2', 'AFZ', 'FZ', 'PZ', 'POZ', 'OZ', 'IZ']
existing_bad_chs = [ch for ch in bad_overlap_chs if ch in raw_filtered.ch_names]
raw_filtered.info['bads'].extend(existing_bad_chs)

# ----------- STEP 5: Create Events from Annotations -----------
events, _ = mne.events_from_annotations(raw_filtered)

if len(events) < 2:
    raise RuntimeError("Not enough events to simulate left/right classes.")

# ----------- STEP 6: Simulate Two Classes from 'T0' Annotations -----------
# Split T0 events into left (1) and right (2)
mid = len(events) // 2
events[:mid, 2] = 1  # left class
events[mid:, 2] = 2  # right class
class_ids = dict(left=1, right=2)

# ----------- STEP 7: Create Epochs -----------
tmin, tmax = 0, 4  # seconds
epochs = mne.Epochs(raw_filtered, events, event_id=class_ids,
                    tmin=tmin, tmax=tmax, baseline=None,
                    picks='eeg', preload=True)

# ----------- STEP 8: Extract Data and Labels -----------
X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs.events[:, -1]  # labels (1 or 2)

# ----------- STEP 9: Apply CSP -----------
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
csp.fit(X, y)

# ----------- STEP 10: Plot CSP Spatial Patterns -----------
csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (uV)', size=1.5)
plt.suptitle('CSP Spatial Patterns')

# ----------- STEP 11: Optional: Classification Test -----------
lda = LinearDiscriminantAnalysis()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lda, csp.transform(X), y, cv=cv, scoring='accuracy')
print(f"\nCSP + LDA classification accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

plt.show()
