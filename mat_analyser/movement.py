import h5py
import numpy as np
import mne
import matplotlib.pyplot as plt


filename = r"C:\Users\Lenovo\Downloads\s01.mat"# Update if needed
duration_sec = 3  # Length of each trial
label_channel = ['C3', 'C4', 'CZ', 'FCZ']

# === Load data ===
f = h5py.File(filename, 'r')
data = f['eeg']['data'][()].T  # shape: (channels, time)
sfreq = float(f['eeg']['fs'][()])
label_refs = f['eeg']['channellabels'][0]

# === Extract channel names ===
def get_labels(file, refs):
    labels = []
    for r in refs:
        chars = file[r][()]
        if isinstance(chars, bytes):
            label = chars.decode()
        else:
            label = ''.join(chr(int(c)) for c in chars[:, 0])
        labels.append(label.strip())
    return labels


ch_names = get_labels(f, label_refs)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)
raw.set_montage('standard_1020', match_case=False, on_missing='ignore')
raw.filter(1.0, 40.0)

# === Extract TrialStart events ===
lat_refs = f['eeg']['event']['latency'][0]
type_refs = f['eeg']['event']['type'][0]

def decode(ref):
    val = f[ref][()]
    return val.decode() if isinstance(val, bytes) else ''.join(chr(int(v)) for v in val[:, 0])

events = []
for lat_ref, type_ref in zip(lat_refs, type_refs):
    event_type = decode(type_ref)
    if event_type == 'TrialStart':
        latency = int(np.squeeze(f[lat_ref][()]))
        events.append(latency)

print(f"Found {len(events)} TrialStart events")

# === Build synthetic fixed-length trial ends ===
trial_length = int(duration_sec * sfreq)
event_array = np.array([[lat, 0, 1] for lat in events if lat + trial_length < data.shape[1]])
event_id = {'movement': 1}

# === Epoch the data ===
epochs = mne.Epochs(raw, event_array, event_id=event_id,
                    tmin=0.0, tmax=duration_sec, baseline=None, preload=True)

# === Plot average ERP per channel ===
evoked = epochs.average()
evoked.plot().suptitle("Average MI Response", fontsize=16)

# === Time-Frequency Plot (ERD/ERS) ===
picks = mne.pick_channels(epochs.info['ch_names'], include=label_channel)
freqs = np.arange(6, 31, 1)
n_cycles = freqs / 2.

power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs,
                                          n_cycles=n_cycles, picks=picks,
                                          time_bandwidth=2.0, return_itc=False)
power.apply_baseline(mode='percent', baseline=(0.0, 1.0))
power.plot_topo(baseline=None, mode=None, title='MI ERD/ERS Power Topo')
power.plot(picks=picks, baseline=(0.0, 1.0), mode='percent', title='Time-Frequency Power')

plt.show()
