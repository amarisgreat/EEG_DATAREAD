import h5py
import numpy as np
import mne
import matplotlib.pyplot as plt


mat_file = h5py.File("D:/S01/S01_Se01_CL_R05.mat", "r")



eeg_data = mat_file['eeg']['data'][()]
eeg_data = eeg_data.T  


sfreq = float(np.squeeze(mat_file['eeg']['fs'][()]))
print(f"Sampling Frequency: {sfreq} Hz")

label_refs = mat_file['eeg']['channellabels'][0]
ch_names = []
for ref in label_refs:
    label_bytes = mat_file[ref][()]
    label = label_bytes.decode('utf-8') if isinstance(label_bytes, bytes) else ''.join(chr(c) for c in label_bytes[:, 0])
    ch_names.append(label)

print(f"✅ Found {len(ch_names)} EEG Channels:", ch_names)


info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)


montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False, on_missing='ignore')


raw.filter(1.0, 40.0, fir_design='firwin')

lat_refs = mat_file['eeg']['event']['latency'][0]
event_latencies = [int(np.squeeze(mat_file[ref][()])) for ref in lat_refs]

event_types_ref = mat_file['eeg']['event']['type'][0]
event_labels = []
for ref in event_types_ref:
    label_bytes = mat_file[ref][()]
    label = label_bytes.decode('utf-8') if isinstance(label_bytes, bytes) else ''.join(chr(c) for c in label_bytes[:, 0])
    event_labels.append(label)

unique_labels = sorted(set(event_labels))
event_id_map = {label: i + 1 for i, label in enumerate(unique_labels)}
print("Event ID Map:", event_id_map)

events = np.array([
    [lat, 0, event_id_map[label]]
    for lat, label in zip(event_latencies, event_labels)
])

print(f"✅ Loaded {len(events)} events.")


# Custom fallback color palette
custom_colors = [
    'red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'
]


color_dict = {
    label: custom_colors[i % len(custom_colors)]
    for i, label in enumerate(event_id_map.keys())
}

raw.plot(
    events=events,
    event_id=event_id_map,
    n_channels=15,
    duration=10,
    scalings='auto',
    show=True,
    title="EEG Signal with Event Markers",
    color='darkblue',
    event_color=color_dict,
    show_options=True,
)

# psd = raw.compute_psd()
# psd.crop(0, 60).plot(
#     average=True,
#     spatial_colors=True,
#     dB=True,
#     show=True,
#     color='indigo'
# )
# Compute and plot PSD
psd = raw.compute_psd(fmin=0, fmax=60)
psd.plot(average=True, spatial_colors=True)


raw.plot_sensors(show_names=True, kind='topomap', sphere=0.1, show=True)

plt.show()
