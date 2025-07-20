import numpy as np
import mne
import scipy.io
import matplotlib.pyplot as plt

# === Load and parse .mat file ===
mat = scipy.io.loadmat(
    r"E:\AMAR\ROBOARM\DATASET\CLA-SubjectJ-170508-3St-LRHand-Inter.mat",
    struct_as_record=False,
    squeeze_me=True
)
eeg_struct = mat['data'][0]

X = eeg_struct.X             # shape: (n_samples, n_channels)
trials = eeg_struct.trial    # start indices of each trial
y = eeg_struct.y             # labels for each trial
fs = int(eeg_struct.fs)      # sampling rate
classes = eeg_struct.classes # list of class labels

print(f"EEG shape: {X.shape}, fs: {fs}, #trials: {len(trials)}")

# === Segment trials ===
trial_len = 1000  # or known duration (e.g., 1s at 1000 Hz)
segments = []

for start in trials:
    if start + trial_len <= X.shape[0]:
        segments.append(X[start:start + trial_len, :])
    else:
        print(f"Skipping trial starting at {start} — too close to end")

epochs_data = np.stack(segments)  # (n_trials, time, channels)
epochs_data = np.transpose(epochs_data, (0, 2, 1))  # (n_trials, channels, time)

# === Create MNE objects ===
ch_names = [f"EEG {i+1}" for i in range(epochs_data.shape[1])]
info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')

events = np.column_stack((np.arange(len(y)), np.zeros(len(y), dtype=int), y.astype(int)))
event_id = {f"class_{int(lbl)}": int(lbl) for lbl in np.unique(y)}

epochs = mne.EpochsArray(epochs_data, info=info, events=events, event_id=event_id)

# === Plot ERPs ===
for label in event_id:
    title = f"{label}: {classes[int(label.split('_')[1]) - 1]}" if isinstance(classes, (list, np.ndarray)) else label
    epochs[label].average().plot_joint(title=title)

plt.show()
