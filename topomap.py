import h5py
import numpy as np
import mne
import matplotlib.pyplot as plt

# === Load .mat EEG file ===
mat_file = h5py.File("D:/S01/S01_Se02_CL_R02.mat", "r")
eeg_data = mat_file['eeg']['data'][()].T
sfreq = float(np.squeeze(mat_file['eeg']['fs'][()]))
label_refs = mat_file['eeg']['channellabels'][0]
ch_names = []

for ref in label_refs:
    label_bytes = mat_file[ref][()]
    label = label_bytes.decode('utf-8') if isinstance(label_bytes, bytes) else ''.join(chr(c) for c in label_bytes[:, 0])
    ch_names.append(label)

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False, on_missing='ignore')
raw.filter(1.0, 40.0)

# === Extract TrialStart events ===
lat_refs = mat_file['eeg']['event']['latency'][0]
type_refs = mat_file['eeg']['event']['type'][0]

event_latencies = []
event_types = []
for lat_ref, type_ref in zip(lat_refs, type_refs):
    lat = int(np.squeeze(mat_file[lat_ref][()]))
    label_bytes = mat_file[type_ref][()]
    label = label_bytes.decode('utf-8') if isinstance(label_bytes, bytes) else ''.join(chr(c) for c in label_bytes[:, 0])
    if label == 'TrialStart':
        event_latencies.append(lat)
        event_types.append(label)

# === Get corresponding MI directions from targetpos ===
target_x = mat_file['eeg']['targetpos']['x'][0]
target_y = mat_file['eeg']['targetpos']['y'][0]

def pos_to_label(x, y):
    if x == -1 and y == 0: return 'left'
    if x == 1 and y == 0: return 'right'
    if x == 0 and y == 1: return 'both'
    if x == 0 and y == -1: return 'feet'
    return 'rest'

mi_labels_list = []
for x_val, y_val in zip(target_x, target_y):
    x = int(np.squeeze(x_val))
    y = int(np.squeeze(y_val))
    mi_labels_list.append(pos_to_label(x, y))

# === MNE-compatible event structure ===
label_to_id = {label: i + 1 for i, label in enumerate(sorted(set(mi_labels_list)))}
events = np.array([
    [lat, 0, label_to_id[mi]]
    for lat, mi in zip(event_latencies, mi_labels_list)
])

# === Remove overlapping channels ===
raw.drop_channels(['CB1', 'CB2'])

# === Epochs per MI type ===
epochs = mne.Epochs(
    raw, events, event_id=label_to_id,
    tmin=-1.0, tmax=2.0,
    baseline=(None, 0), preload=True
)

# === Plot topomaps of Mu and Beta bands per MI type ===
mi_labels = sorted(label_to_id.keys())
fig, axes = plt.subplots(len(mi_labels), 2, figsize=(10, 3 * len(mi_labels)))

# Ensure axes is always 2D
if len(mi_labels) == 1:
    axes = np.array([axes])

# Define bands using dict for compatibility
bands = {'Mu (8–13 Hz)': (8, 13), 'Beta (13–30 Hz)': (13, 30)}

for i, label in enumerate(mi_labels):
    band_psd = epochs[label].compute_psd(fmin=1, fmax=40)
    band_psd.plot_topomap(
        ch_type='eeg',
        bands=bands,
        axes=axes[i],
        show=False,
        normalize=True,
        contours=0,
        size=1.2,
        colorbar=False
    )
    axes[i][0].set_title(f"{label} - Mu (8–13 Hz)")
    axes[i][1].set_title(f"{label} - Beta (13–30 Hz)")

plt.tight_layout()
plt.show()
