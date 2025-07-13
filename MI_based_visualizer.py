import h5py
import numpy as np
import mne
import matplotlib.pyplot as plt

# === Load the HDF5-based .mat file ===
mat_file = h5py.File("D:/S01/S01_Se01_CL_R05.mat", "r")

# === Extract EEG data ===
eeg_data = mat_file['eeg']['data'][()]
eeg_data = eeg_data.T  # (n_channels, n_samples)

# === Sampling Frequency ===
sfreq = float(np.squeeze(mat_file['eeg']['fs'][()]))
print(f"Sampling Frequency: {sfreq} Hz")

# === Channel Names ===
label_refs = mat_file['eeg']['channellabels'][0]
ch_names = []
for ref in label_refs:
    label_bytes = mat_file[ref][()]
    label = label_bytes.decode('utf-8') if isinstance(label_bytes, bytes) else ''.join(chr(c) for c in label_bytes[:, 0])
    ch_names.append(label)

print(f"Found {len(ch_names)} EEG Channels:", ch_names)

# === Create RawArray ===
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)

# === Set Montage ===
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False, on_missing='ignore')

# === Bandpass Filter (1–40 Hz) ===
raw.filter(1.0, 40.0, fir_design='firwin')

# === Extract Event Latencies & Labels ===
lat_refs = mat_file['eeg']['event']['latency'][0]
event_latencies = [int(np.squeeze(mat_file[ref][()])) for ref in lat_refs]

event_types_ref = mat_file['eeg']['event']['type'][0]
event_labels = []
for ref in event_types_ref:
    label_bytes = mat_file[ref][()]
    label = label_bytes.decode('utf-8') if isinstance(label_bytes, bytes) else ''.join(chr(c) for c in label_bytes[:, 0])
    event_labels.append(label)

# === Map event labels to integers ===
unique_labels = sorted(set(event_labels))
event_id_map = {label: i + 1 for i, label in enumerate(unique_labels)}
print("Event ID Map:", event_id_map)

# === Build events array for MNE ===
events = np.array([
    [lat, 0, event_id_map[label]]
    for lat, label in zip(event_latencies, event_labels)
])
print(f"Loaded {len(events)} events.")

# === Define motor cortex channels (exact case match) ===
motor_channels = [ch for ch in ['C3', 'C4', 'CZ', 'FCZ'] if ch in raw.ch_names]
print("Using motor cortex channels:", motor_channels)

# === Plot 1: Raw EEG Waveforms ===
raw.plot(
    picks=motor_channels,
    duration=10,
    n_channels=len(motor_channels),
    title="Motor Cortex EEG - C3, C4, CZ, FCZ",
    show=True,
    scalings='auto',
    block=True
)

# === Plot 2: Power Spectral Density ===
psd = raw.compute_psd(fmin=1, fmax=40)
psd.plot(
    picks=motor_channels,
    average=False,
    spatial_colors=True,
    show=True,
    title="PSD: Mu and Beta Activity in Motor Channels"
)

# === Plot 3: Topomap for Mu and Beta Bands ===
psd.plot_topomap(
    ch_type='eeg',
    bands=[(8, 13, 'Mu'), (13, 30, 'Beta')],
    normalize=True,
    size=1.2,
    show=True,
    title="Topomap: Mu and Beta Band Activity"
)

plt.show()
