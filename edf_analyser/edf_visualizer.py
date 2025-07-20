import mne
import matplotlib.pyplot as plt

# ----------- STEP 1: Load EEG EDF File -----------
file_path = r'E:\AMAR\ROBOARM\DATASET\files\S001\S001R01.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# ----------- STEP 2: Standardize Channel Names -----------
raw.rename_channels(lambda x: x.strip('.').upper())
raw.set_montage('standard_1020', on_missing='ignore')

# ----------- STEP 3: Display Info -----------
print(raw.info)

# ----------- STEP 4: Plot Raw EEG -----------
raw.plot(n_channels=30, duration=10, scalings='auto', title="Raw EEG")

raw.plot_psd(fmax=60, average=True)

raw_filtered = raw.copy().filter(7, 30, fir_design='firwin')

bad_overlap_chs = ['FCZ', 'CZ', 'CPZ', 'FP1', 'FPZ', 'FP2', 'AFZ', 'FZ', 'PZ', 'POZ', 'OZ', 'IZ']
existing_bad_chs = [ch for ch in bad_overlap_chs if ch in raw_filtered.ch_names]
raw_filtered.info['bads'].extend(existing_bad_chs)

psd = raw_filtered.compute_psd()
psd.plot_topomap(ch_type='eeg', normalize=True)

raw.plot_sensors(show_names=True)

plt.show()
