import mne
from mne.decoding import CSP
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Load Data
file_path = r'E:\AMAR\ROBOARM\DATASET\files\S001\S001R01.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# STEP 2: Standardize Channels and Montage
raw.rename_channels(lambda x: x.strip('.').upper())
raw.set_montage('standard_1020', on_missing='ignore')

# STEP 3: Bandpass Filter
raw_filtered = raw.copy().filter(7, 30, fir_design='firwin')

# STEP 4: Remove Bad Channels
bad_chs = ['FCZ', 'CZ', 'CPZ', 'FP1', 'FPZ', 'FP2', 'AFZ', 'FZ', 'PZ', 'POZ', 'OZ', 'IZ']
raw_filtered.info['bads'].extend([ch for ch in bad_chs if ch in raw_filtered.ch_names])

# STEP 5: Generate Dummy Events (simulate 10 left, 10 right)
sfreq = raw_filtered.info['sfreq']
onsets = np.arange(0, 20 * int(sfreq * 4), int(sfreq * 4))
events = np.array([[onset, 0, 1 if i < 10 else 2] for i, onset in enumerate(onsets)])
event_id = dict(left=1, right=2)

# STEP 6: Create Epochs
tmin, tmax = 0, 4
epochs = mne.Epochs(raw_filtered, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    baseline=None, picks='eeg', preload=True)
X = epochs.get_data()   # Shape: (n_epochs, n_channels, n_times)
y = epochs.events[:, -1]  # Labels (1 or 2)

# STEP 7: Fit CSP
n_components = 4
csp = CSP(n_components=n_components, log=True, norm_trace=False)
X_csp = csp.fit_transform(X, y)  # Shape: (n_epochs, n_components)

# STEP 8: Save for SNN Input
np.save('X_csp.npy', X_csp)     # Features for SNN
np.save('y_labels.npy', y)      # Labels (1 = left, 2 = right)

# Optional: Plot CSP patterns
csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (uV)', size=1.5)
plt.suptitle('CSP Spatial Patterns')
plt.show()

print(f"✅ Saved CSP features: {X_csp.shape}, Labels: {y.shape}")
