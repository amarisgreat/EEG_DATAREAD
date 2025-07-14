import h5py
import numpy as np
import mne
import matplotlib.pyplot as plt

# === Load the HDF5-based .mat file ===
mat_file = h5py.File("D:/S01/S01_Se01_CL_R05.mat", "r")

# === Extract EEG data ===
eeg_data = mat_file['eeg']['data'][()]          
eeg_data = eeg_data.T                           

# === Extract sampling frequency ===
sfreq = float(np.squeeze(mat_file['eeg']['fs'][()]))
print(f"Sampling Frequency: {sfreq} Hz")



label_refs = mat_file['eeg']['channellabels'][0]
ch_names = []

for ref in label_refs:
    obj = mat_file[ref]
    label_bytes = obj[()]
    if isinstance(label_bytes, bytes):
        label = label_bytes.decode('utf-8')
    else:
        label = ''.join(chr(c) for c in label_bytes[:, 0])
    ch_names.append(label)

print(f" Found {len(ch_names)} EEG Channels:", ch_names)


info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)


montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False, on_missing='ignore')


raw.filter(1.0, 40.0)
 
raw.plot(n_channels=15, duration=10, scalings='auto', title="EEG Signal")
raw.plot_psd(fmax=60, average=True, spatial_colors=True)
raw.plot_sensors(show_names=True)
raw.plot_topomap()
plt.show()