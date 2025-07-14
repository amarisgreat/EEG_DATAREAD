# import h5py
# import numpy as np
# import mne
# import matplotlib.pyplot as plt

# # === Load Data ===
# filename = r"D:\S02\S02_Se01_Click_R03.mat"
# f = h5py.File(filename, "r")
# eeg_data = f['eeg']['data'][()].T
# sfreq = float(np.array(f['eeg']['fs']).squeeze())
# label_refs = f['eeg']['channellabels'][0]
# latencies = f['eeg']['event']['latency'][0]
# types = f['eeg']['event']['type'][0]

# # === Get Channel Names ===
# def decode_labels(refs):
#     chs = []
#     for r in refs:
#         raw = f[r][()]
#         if isinstance(raw, bytes):
#             chs.append(raw.decode())
#         else:
#             chs.append(''.join(chr(x) for x in raw[:, 0]))
#     return chs

# ch_names = decode_labels(label_refs)
# info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
# raw = mne.io.RawArray(eeg_data, info)
# raw.drop_channels(['CB1', 'CB2'])
# raw.set_montage('standard_1020', match_case=False)
# raw.filter(1., 40.)

# # === Map (x, y) to MI labels ===
# def pos_to_label(x, y):
#     if x == -1 and y == 0: return 'left'
#     elif x == 1 and y == 0: return 'right'
#     elif x == 0 and y == -1: return 'feet'
#     elif x == 0 and y == 1: return 'both'
#     else: return 'rest'

# # === Extract and Decode Target Positions ===
# try:
#     # Directly extract numeric arrays
#     target_x = np.array(f['eeg']['targetpos']['x']).squeeze()
#     target_y = np.array(f['eeg']['targetpos']['y']).squeeze()
#     positions = [(int(np.round(x)), int(np.round(y))) for x, y in zip(target_x, target_y)]
#     mi_labels = [pos_to_label(x, y) for x, y in positions]

#     # Print target position values with corresponding labels
#     print("Target positions (x, y) and MI labels:")
#     for i, ((x, y), label) in enumerate(zip(positions, mi_labels)):
#         print(f"{i:02d}: x={x}, y={y} → {label}")

# except Exception as e:
#     print("⚠️ Failed to extract MI labels from target positions:", e)
#     mi_labels = ['rest'] * len(latencies)

# # === Construct Events ===
# event_latencies = [int(np.squeeze(f[lat][()])) for lat in latencies]
# label_to_id = {l: i + 1 for i, l in enumerate(sorted(set(mi_labels)))}
# events = np.array([[lat, 0, label_to_id[lab]] for lat, lab in zip(event_latencies, mi_labels)])

# # === Epochs ===
# epochs = mne.Epochs(raw, events, event_id=label_to_id, tmin=-1.0, tmax=2.0,
#                     baseline=(None, 0), preload=True)

# # === ERD/ERS Time-Frequency ===
# picks = mne.pick_channels(epochs.info['ch_names'], include=['C3', 'C4', 'CZ', 'FCZ'])
# freqs = np.arange(6, 31, 1)
# n_cycles = freqs / 2.0
# power_dict = {}

# for label in label_to_id:
#     if len(epochs[label]) == 0:
#         continue
#     power = mne.time_frequency.tfr_multitaper(
#         epochs[label], picks=picks, freqs=freqs, n_cycles=n_cycles,
#         time_bandwidth=2.0, return_itc=False, average=True
#     )
#     power.apply_baseline(baseline=(-1.0, 0.0), mode='percent')
#     power_dict[label] = power

# # === Plot ERD/ERS ===
# fig, axes = plt.subplots(len(power_dict), len(picks), figsize=(15, 4 * len(power_dict)))
# for i, (label, power) in enumerate(power_dict.items()):
#     for j, ch in enumerate(['C3', 'C4', 'CZ', 'FCZ']):
#         ax = axes[i, j] if len(power_dict) > 1 else axes[j]
#         power.plot([j], baseline=None, mode=None, axes=ax, colorbar=False, show=False)
#         ax.set_title(f"{label} - {ch}" if i == 0 else ch)
# plt.tight_layout()
# plt.show()

import h5py
import numpy as np

# === Change this path to your file ===
filename = r"C:\Users\Lenovo\Downloads\s01.mat"
f = h5py.File(filename, "r")

# === Helper: Decode text from MATLAB (HDF5 v7.3) format ===
def decode_ref_string(ref, file):
    try:
        data = file[ref][()]
        if isinstance(data, bytes):
            return data.decode().strip()
        return ''.join(chr(int(c)) for c in data[:, 0]).strip()
    except:
        return str(ref)

# === Extract event types ===
try:
    types = f['eeg']['event']['type'][0]
    event_labels = [decode_ref_string(t, f) for t in types]
    print("✅ Unique Event Labels in this file:")
    print(set(event_labels))
except Exception as e:
    print("⚠️ Couldn't extract event types:", e)

# === Extract target positions (x, y) ===
try:
    target_x_refs = f['eeg']['targetpos']['x'][0]
    target_y_refs = f['eeg']['targetpos']['y'][0]
    positions = []

    for tx_ref, ty_ref in zip(target_x_refs, target_y_refs):
        x = int(np.round(float(f[tx_ref][()])))
        y = int(np.round(float(f[ty_ref][()])))
        positions.append((x, y))

    print("\n✅ Unique Target Positions:")
    unique_positions = sorted(set(positions))
    for x, y in unique_positions:
        label = (
            "left" if (x, y) == (-1, 0) else
            "right" if (x, y) == (1, 0) else
            "feet" if (x, y) == (0, -1) else
            "both" if (x, y) == (0, 1) else
            "rest"
        )
        print(f"x={x}, y={y} → {label}")
except Exception as e:
    print("⚠️ Couldn't extract target positions:", e)
