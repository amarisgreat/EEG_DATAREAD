import h5py
import numpy as np
import mne
import matplotlib.pyplot as plt

# === Load EEG Data ===
mat_path = "D:/S01/S01_Se05_CL_R03.mat"  # 👈 Use a file that has real MI trials
mat_file = h5py.File(mat_path, "r")
eeg_data = mat_file['eeg']['data'][()].T
sfreq = float(np.squeeze(mat_file['eeg']['fs'][()]))
label_refs = mat_file['eeg']['channellabels'][0]

# === Channel names ===
ch_names = []
for ref in label_refs:
    label_bytes = mat_file[ref][()]
    label = label_bytes.decode('utf-8') if isinstance(label_bytes, bytes) else ''.join(chr(c) for c in label_bytes[:, 0])
    ch_names.append(label)

# === Create Raw object ===
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)
raw.set_montage('standard_1020', match_case=False, on_missing='ignore')
raw.drop_channels(['CB1', 'CB2'])  # to avoid layout errors
raw.filter(1.0, 40.0)

# === Get Events and MI Labels ===
lat_refs = mat_file['eeg']['event']['latency'][0]
type_refs = mat_file['eeg']['event']['type'][0]
target_x = mat_file['eeg']['targetpos']['x'][0]
target_y = mat_file['eeg']['targetpos']['y'][0]

def pos_to_label(x, y):
    if x == -1 and y == 0: return 'left'
    if x == 1 and y == 0: return 'right'
    if x == 0 and y == 1: return 'both'
    if x == 0 and y == -1: return 'feet'
    return 'rest'

event_latencies, mi_labels_list = [], []
print("Target positions (x, y):")
for lat_ref, x_ref, y_ref in zip(lat_refs, target_x, target_y):
    lat = int(np.squeeze(mat_file[lat_ref][()]))
    x = int(np.squeeze(x_ref))
    y = int(np.squeeze(y_ref))
    label = pos_to_label(x, y)
    print(f"({x}, {y}) => {label}")
    event_latencies.append(lat)
    mi_labels_list.append(label)

# === Build MNE Events Array ===
label_to_id = {label: i + 1 for i, label in enumerate(sorted(set(mi_labels_list)))}
events = np.array([[lat, 0, label_to_id[mi]] for lat, mi in zip(event_latencies, mi_labels_list)])

# === Epoching ===
epochs = mne.Epochs(raw, events, event_id=label_to_id, tmin=-1.0, tmax=2.0, baseline=(None, 0), preload=True)

# === Compute TFR (Time-Frequency Representation) ===
picks = mne.pick_channels(epochs.info['ch_names'], include=['C3', 'C4', 'CZ', 'FCZ'])
power_dict = {}
freqs = np.arange(6, 31, 1)
n_cycles = freqs / 2.0

for label in label_to_id:
    if len(epochs[label]) == 0:
        print(f"Skipping {label} (no trials)")
        continue
    power = mne.time_frequency.tfr_multitaper(
        epochs[label], picks=picks, freqs=freqs, n_cycles=n_cycles,
        time_bandwidth=2.0, return_itc=False, average=True
    )
    power.apply_baseline(baseline=(-1.0, 0.0), mode='percent')
    power_dict[label] = power

# === Plot ERD/ERS Comparison Across MI Labels ===
n_mi = len(power_dict)
fig, axes = plt.subplots(n_mi, len(picks), figsize=(15, 3 * n_mi))

for i, (label, power) in enumerate(power_dict.items()):
    for j, ch in enumerate(['C3', 'C4', 'CZ', 'FCZ']):
        ax = axes[i, j] if n_mi > 1 else axes[j]
        power.plot([j], baseline=None, mode=None, axes=ax, colorbar=False, show=False)
        if i == 0:
            ax.set_title(ch)
        if j == 0:
            ax.set_ylabel(label)

plt.tight_layout()
plt.show()
