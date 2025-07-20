import h5py
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage

# -------- CONFIG --------
filename = r"D:\S02\S02_Se01_Click_R03.mat"
# ------------------------

def extract_string_array(ref, file):
    return [''.join(chr(int(c)) for c in file[ref[0, i]][:]).strip()
            for i in range(ref.shape[1])]

def extract_scalar_array(refs, file):
    return [float(file[r][:].squeeze()) for r in refs[0]]

def extract_string_list(refs, file):
    return [''.join(chr(int(c)) for c in file[r][:]).strip() for r in refs[0]]

# === Load .mat file ===
with h5py.File(filename, 'r') as f:
    eeg = f['eeg']
    data = np.array(eeg['data']).T
    fs = int(np.array(eeg['fs'])[0, 0])
    times = np.array(eeg['times']).squeeze()
    labels = extract_string_array(eeg['channellabels'], f)
    cursor_x = np.array(eeg['cursorpos']['x']).squeeze()
    cursor_y = np.array(eeg['cursorpos']['y']).squeeze()
    postimes = np.array(eeg['postimes']).squeeze()
    event = eeg['event']
    latencies = extract_scalar_array(event['latency'], f)
    durations = extract_scalar_array(event['duration'], f)
    event_types = extract_string_list(event['type'], f)

# === Create Raw object ===
info = mne.create_info(ch_names=labels, sfreq=fs, ch_types='eeg')
raw = mne.io.RawArray(data, info)
raw.set_montage('standard_1020', on_missing='ignore')

# ✅ Filter only valid montage channels
montage = make_standard_montage("standard_1020")
valid_chs = [ch for ch in raw.info['ch_names'] if ch in montage.ch_names]
raw_valid = raw.copy().pick_channels(valid_chs)
raw_valid.set_montage(montage)

# === EEG viewer ===
raw.plot(n_channels=10, duration=10, scalings='auto', title='EEG Signals')

# === PSD ===
raw.plot_psd(fmax=60)

# === Safe Topomap at 5s ===
sample = fs * 5
topo_values = raw_valid.get_data()[:, sample]
mne.viz.plot_topomap(topo_values, raw_valid.info, cmap='viridis', show=True)

# === Sensor layout ===
raw_valid.plot_sensors(kind='topomap', show_names=True)

# === Cursor path ===
plt.figure(figsize=(6, 6))
plt.plot(cursor_x, cursor_y)
plt.title("Cursor Trajectory")
plt.xlabel("X (normalized)")
plt.ylabel("Y (normalized)")
plt.axis('equal')
plt.grid(True)
plt.show()

# === Event log ===
print("\nTrial Events:")
for i, (etype, start, dur) in enumerate(zip(event_types, latencies, durations)):
    print(f"Event {i+1}: Type = {etype}, Start = {start/fs:.2f}s, Duration = {dur/1000:.2f}s")
