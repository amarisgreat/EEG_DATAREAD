import numpy as np
import mne
from scipy.io import loadmat
import matplotlib.pyplot as plt

# === Load the data ===
mat = loadmat(r"E:/AMAR/ROBOARM/DATASET/CLA-SubjectJ-170508-3St-LRHand-Inter.mat", simplify_cells=True)
eeg = mat["eeg"]
imagery_left = eeg["imagery_left"][:64]   # Use only first 64 channels
imagery_right = eeg["imagery_right"][:64]
sfreq = eeg["srate"]
senloc = eeg["senloc"][:64]  # shape: (64, 3)
n_trials = eeg["n_imagery_trials"]
samples_per_trial = imagery_left.shape[1] // n_trials

# === Stack and label all trials ===
all_data = np.concatenate((imagery_left, imagery_right), axis=1)  # shape: (64, 716800)
ch_names = [f"EEG {i+1}" for i in range(64)]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

# Set montage from 3D sensor locations
montage = mne.channels.make_dig_montage(ch_pos={ch: pos for ch, pos in zip(ch_names, senloc)}, coord_frame='head')
info.set_montage(montage)

# === Create raw ===
raw = mne.io.RawArray(all_data * 1e-6, info)  # convert µV to V
raw.filter(1., 40., fir_design='firwin')

raw.plot_sensors(show_names=True)

# === Create events ===
events = []
for i in range(n_trials):
    latency = i * samples_per_trial
    events.append([latency, 0, 1])  # left
for i in range(n_trials):
    latency = (i + n_trials) * samples_per_trial
    events.append([latency, 0, 2])  # right
events = np.array(events)

# === Epoching ===
epochs = mne.Epochs(raw, events, event_id={'left': 1, 'right': 2},
                    tmin=0, tmax=samples_per_trial / sfreq, baseline=None,
                    preload=True)

# === Time-frequency ERD/ERS ===
freqs = np.arange(6, 30, 2)
n_cycles = freqs / 2.
picks = mne.pick_channels(epochs.info["ch_names"], include=["EEG 1", "EEG 2", "EEG 3"])
power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                                          picks=picks, return_itc=False)
power.apply_baseline(baseline=(0, 1), mode='percent')
power.plot_topo(baseline=(0, 1), mode='percent', title='ERD/ERS - Left vs Right MI')

# === ICA ===
ica = mne.preprocessing.ICA(n_components=15, random_state=42)
ica.fit(raw)
ica.plot_components()
