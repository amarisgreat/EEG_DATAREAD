import numpy as np
import mne
from scipy.io import loadmat

# === Load .mat file ===
mat = loadmat(r"C:\Users\Lenovo\Downloads\s01.mat", simplify_cells=True)
eeg = mat['eeg']

# === Extract variables ===
imagery_left = eeg['imagery_left'][:64] * 1e-6  # Convert µV to V
imagery_right = eeg['imagery_right'][:64] * 1e-6
sfreq = eeg['srate']
n_trials = eeg['n_imagery_trials']
samples_per_trial = imagery_left.shape[1] // n_trials
ch_names = [f"EEG {i+1}" for i in range(64)]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# === Choose a clean trial to visualize ===
trial_index = 0
left_trial = imagery_left[:, trial_index * samples_per_trial : (trial_index + 1) * samples_per_trial]
right_trial = imagery_right[:, trial_index * samples_per_trial : (trial_index + 1) * samples_per_trial]

# === Create RawArray ===
raw_left = mne.io.RawArray(left_trial, info)
raw_right = mne.io.RawArray(right_trial, info)

# === Bandpass filter for clarity (1–40 Hz) ===
raw_left.filter(1., 40., fir_design='firwin')
raw_right.filter(1., 40., fir_design='firwin')

# === Plot smooth waves ===
raw_left.plot(n_channels=16, duration=6.0, title="Left-Hand Imagery - Trial 1", show=True, block=True)
raw_right.plot(n_channels=16, duration=6.0, title="Right-Hand Imagery - Trial 1", show=True, block=True)
