import scipy.io
import h5py
import numpy as np
import mne
import os
import matplotlib.pyplot as plt

mat_file_path = 'D:/S01/S01_Se01_CL_R05.mat'  # Replace with your actual .mat file path

def load_mat_file(filepath):
    try:
        mat = scipy.io.loadmat(filepath)
        print("Loaded using scipy.io (MAT < v7.3)")
        return 'scipy', mat
    except NotImplementedError:
        f = h5py.File(filepath, 'r')
        print("Loaded using h5py (MAT ≥ v7.3)")
        return 'h5py', f


loader_type, data = load_mat_file(mat_file_path)

if loader_type == 'scipy':
    # Replace these keys with the actual keys in your .mat file
    eeg_data = data['data']             # (channels, time)
    sfreq = float(data['sfreq'])        # e.g., 256.0 Hz
elif loader_type == 'h5py':
    # Inspect keys using list(data.keys()) or data.visit(print)
    eeg_data = data['eeg']['data'][()]          # shape: (channels, time)
    sfreq = float(data['eeg']['srate'][()])     # or 'sampling_rate', 'sfreq', etc.


if eeg_data.shape[0] > eeg_data.shape[1]:
    eeg_data = eeg_data.T

n_channels = eeg_data.shape[0]
ch_names = [f'EEG {i+1}' for i in range(n_channels)]
ch_types = ['eeg'] * n_channels


info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(eeg_data, info)

raw.filter(l_freq=1.0, h_freq=40.0)  

print(raw.info)
raw.plot(n_channels=8, scalings='auto', duration=5, show=True)
raw.plot_psd(fmax=50)

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False)
raw.plot_sensors(show_names=True)
