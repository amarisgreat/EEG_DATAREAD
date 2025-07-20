import h5py
import numpy as np
import matplotlib.pyplot as plt

# === Load .mat file ===
file_path = 'D:/S01/S01_Se01_CL_R05.mat'
f = h5py.File(file_path, 'r')
eeg = f['eeg']

# === Extract and shape EEG data ===
raw_data = eeg['data'][()]
print("Raw EEG shape:", raw_data.shape)

# === Extract channel labels ===
ch_labels = []
for ref in eeg['channellabels'][0]:
    ch = f[ref][()]
    label = ''.join(chr(int(c)) for c in ch.flatten())
    ch_labels.append(label)


if raw_data.shape[0] == len(ch_labels):  
    eeg_data = raw_data.T
else:
    eeg_data = raw_data

print("✅ Total EEG Channels:", len(ch_labels))
print("🧠 Available EEG Channels:", ch_labels)
print("Processed EEG shape:", eeg_data.shape)


times = eeg['times'][()].flatten()

cursor_x = eeg['cursorpos']['x'][()].flatten()
cursor_y = eeg['cursorpos']['y'][()].flatten()
cursor_times = eeg['postimes'][()].flatten()

cursor_indices = [np.searchsorted(times, pt) for pt in cursor_times]


vel_x = np.diff(cursor_x)
vel_y = np.diff(cursor_y)

threshold = 0.01  
movements = []

for i in range(len(vel_x)):
    t = cursor_times[i]
    if vel_x[i] > threshold:
        movements.append((t, 'Right-arm MI'))
    elif vel_x[i] < -threshold:
        movements.append((t, 'Left-arm MI'))
    elif vel_y[i] > threshold:
        movements.append((t, 'Both-arms MI'))

# === Visualize EEG + cursor around MI events ===
window_sec = 2  # 2 seconds before and after
fs = 1000  # Sampling rate

for idx, (event_time, label) in enumerate(movements[:3]):  # Only first 3 for brevity
    center_idx = np.searchsorted(times, event_time)
    start = max(center_idx - window_sec * fs, 0)
    end = min(center_idx + window_sec * fs, len(times))

    t_eeg = times[start:end]
    eeg_segment = eeg_data[start:end, :]

    # Cursor segment around event
    cursor_idx = np.searchsorted(cursor_times, event_time)
    cursor_range = range(max(cursor_idx - 10, 0), min(cursor_idx + 10, len(cursor_times)))
    cursor_x_seg = cursor_x[cursor_range]
    cursor_y_seg = cursor_y[cursor_range]
    cursor_t_seg = cursor_times[cursor_range]

    # === Plot EEG channels with vertical offset ===
    plt.figure(figsize=(16, 8))
    num_channels = eeg_segment.shape[1]

    for i in range(num_channels):
        channel_data = eeg_segment[:, i]
        if len(t_eeg) == len(channel_data):
            plt.plot(t_eeg, channel_data + i * 50, label=ch_labels[i])
        else:
            print(f"Skipping channel {i} due to shape mismatch")

    plt.axvline(event_time, color='red', linestyle='--')
    plt.title(f'All EEG Channels around {label} at {event_time:.2f}s')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μV) (offset per channel)")
    plt.legend(loc='upper right', fontsize='xx-small', ncol=4)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot Cursor Movement ===
    plt.figure(figsize=(10, 3))
    plt.plot(cursor_t_seg, cursor_x_seg, label='X (Left/Right)')
    plt.plot(cursor_t_seg, cursor_y_seg, label='Y (Up/Down)')
    plt.axvline(event_time, color='red', linestyle='--')
    plt.title(f'Cursor Movement around {label}')
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
