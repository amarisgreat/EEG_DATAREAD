# import mne
# from mne.decoding import CSP
# from mne.preprocessing import ICA
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # --- 0. Setup and Parameters ---
# print("Starting BCI Preprocessing Pipeline...")

# # --- 1. Load Data and Apply Montage ---
# # Define the path to the downloaded EEG data file
# file_path = 'S001R04.edf' 

# # Safety check to ensure the data file exists
# if not os.path.exists(file_path):
#     raise FileNotFoundError(
#         f"Data file not found at '{file_path}'. "
#         "Please download the file from the PhysioNet EEGMMI dataset "
#         "and place it in the same directory as this script."
#     )
    
# print(f"--- Starting Preprocessing for {file_path} ---")

# raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
# raw.rename_channels(lambda x: x.strip('.').upper())
# raw.set_montage('standard_1020', on_missing='ignore')

# # **FIX:** Define midline channels to be removed to prevent plotting errors.
# midline_channels = ['FCZ', 'CZ', 'CPZ', 'FPZ', 'AFZ', 'FZ', 'PZ', 'POZ', 'OZ', 'IZ', 'FP1', 'FP2']
# channels_to_drop = [ch for ch in midline_channels if ch in raw.ch_names]
# raw.drop_channels(channels_to_drop)
# print(f"Dropped midline channels to prevent plotting errors: {channels_to_drop}")


# # Visualize the initial sensor locations
# print("Step 1/6: Data loaded. Displaying sensor locations...")
# raw.plot_sensors(show_names=True)
# plt.show() # This will pause the script until you close the plot window

# # --- 2. Initial PSD Visualization ---
# print("Step 2/6: Displaying Power Spectral Density (PSD) before filtering...")
# raw.plot_psd(fmin=2, fmax=40, average=True, picks='eeg')
# plt.show()

# # --- 3. Band-Pass Filtering ---
# raw.filter(7.0, 30.0, fir_design='firwin', skip_by_annotation='edge')

# # Visualize PSD after filtering
# print("Step 3/6: Applied band-pass filter (7-30 Hz). Displaying filtered PSD...")
# raw.plot_psd(fmin=2, fmax=40, average=True, picks='eeg')
# plt.show()

# # --- 4. Robust Artifact Removal with ICA ---
# # ICA works best on data with lower frequencies, so we filter a copy of the raw data.
# raw_for_ica = raw.copy().filter(1.0, 40.0)

# ica = ICA(n_components=15, max_iter='auto', random_state=97)
# ica.fit(raw_for_ica)

# # Visualize the independent components' topographic maps
# print("Step 4/6: Applying ICA for artifact removal. Displaying ICA component topographies...")
# ica.plot_components()
# plt.show()

# # Visualize the properties of each ICA component (including PSD)
# print("Displaying ICA component properties. Look for components with high power at high frequencies to identify muscle artifacts.")
# ica.plot_properties(raw_for_ica, picks=range(0, 15))
# plt.show()


# # Automatically find and mark components related to artifacts
# ica.exclude = []
# eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['FP1', 'FP2'])
# if eog_indices:
#     ica.exclude.extend(eog_indices)
#     print(f"Automatically found and marked EOG components: {eog_indices}")

# # The find_bads_ecg function looks for an existing component that matches a heartbeat pattern
# try:
#     ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
#     if ecg_indices:
#         ica.exclude.extend(ecg_indices)
#         print(f"Automatically found and marked ECG components: {ecg_indices}")
# except Exception as e:
#     print(f"Could not automatically detect ECG components: {e}")

# # Visualize the effect of removing the identified components
# print(f"Total components to exclude: {ica.exclude}. Displaying signal before vs. after cleaning...")
# ica.plot_overlay(raw, exclude=ica.exclude, picks='eeg', title="Signal Before vs. After ICA Cleaning")
# plt.show()

# # Apply the artifact removal to the data
# if ica.exclude:
#     ica.apply(raw)
# print("ICA cleaning complete.")

# # --- 5. Epoching Data into Trials ---
# # Extract events (T1 for left fist, T2 for right fist) from annotations
# events, event_id = mne.events_from_annotations(raw)
# class_ids = {'left_fist': event_id['T1'], 'right_fist': event_id['T2']}

# # Create 4-second epochs (trials) starting at the onset of each event
# tmin, tmax = 0., 4.
# epochs = mne.Epochs(raw, events, event_id=class_ids, tmin=tmin, tmax=tmax,
#                     proj=True, picks='eeg', baseline=None, preload=True)

# # Visualize the epoched data
# print("Step 5/6: Data epoched into trials. Displaying epochs visualization...")
# epochs.plot_image(combine='mean', title='Epoched Data (Mean Across Trials)')
# plt.show()

# # --- 6. Feature Extraction with CSP ---
# print("Step 6/6: Applying CSP for feature extraction. Displaying spatial patterns...")
# X = epochs.get_data()
# y = epochs.events[:, -1]

# # Initialize CSP; it will find 4 spatial filters to best separate the classes
# csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
# X_csp = csp.fit_transform(X, y)

# # Visualize the CSP spatial patterns
# csp.plot_patterns(epochs.info, ch_type='eeg', units='Arbitrary Units', size=1.5)
# plt.suptitle('CSP Spatial Patterns')
# plt.show()

# print("\nPreprocessing and feature extraction complete.")
# print(f"The extracted features (X_csp) have the shape: {X_csp.shape}")
# print("These features are now ready to be used as input for a Spiking Neural Network or other classifier.")


import mne
from mne.decoding import CSP
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 0. Setup and Parameters ---
print("Starting BCI Preprocessing Pipeline...")

# --- 1. Load Data and Apply Montage ---
# Define the path to the downloaded EEG data file
file_path = 'S001R04.edf' 

# Safety check to ensure the data file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Data file not found at '{file_path}'. "
        "Please download the file from the PhysioNet EEGMMI dataset "
        "and place it in the same directory as this script."
    )
    
print(f"--- Starting Preprocessing for {file_path} ---")

raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
raw.rename_channels(lambda x: x.strip('.').upper())
raw.set_montage('standard_1020', on_missing='ignore')

# --- 2. Initial Filtering and PSD Visualization ---
print("Step 1/5: Filtering data and visualizing PSD...")
# Apply a band-pass filter to isolate the mu and beta bands
raw.filter(7.0, 30.0, fir_design='firwin', skip_by_annotation='edge')

# Visualize PSD after filtering
raw.plot_psd(fmin=2, fmax=40, average=True, picks='eeg')
plt.show()

# --- 3. Robust Artifact Removal with ICA ---
# ICA works best on data with lower frequencies, so we filter a copy of the raw data.
raw_for_ica = raw.copy().filter(1.0, 40.0)

ica = ICA(n_components=15, max_iter='auto', random_state=97)
ica.fit(raw_for_ica)

# Automatically find artifact components using the full raw object (with FP1/FP2)
print("Step 2/5: Applying ICA for artifact removal...")
ica.exclude = []
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['FP1', 'FP2'])
if eog_indices:
    ica.exclude.extend(eog_indices)
    print(f"Automatically found and marked EOG components: {eog_indices}")

try:
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
    if ecg_indices:
        ica.exclude.extend(ecg_indices)
        print(f"Automatically found and marked ECG components: {ecg_indices}")
except Exception as e:
    print(f"Could not automatically detect ECG components: {e}")

# Apply the artifact removal to the data
if ica.exclude:
    print(f"Total components to exclude: {ica.exclude}. Applying cleaning...")
    ica.apply(raw)
    print("ICA cleaning complete.")
else:
    print("No artifact components were automatically found to exclude.")


# --- 4. Drop Problematic Channels for Visualization ---
# **FIX:** Now that artifact detection is done, we can safely remove the channels
# that cause plotting errors.
channels_to_drop = ['FCZ', 'CZ', 'CPZ', 'FP1', 'FPZ', 'FP2', 'AFZ', 'FZ', 'PZ', 'POZ', 'OZ', 'IZ']
existing_to_drop = [ch for ch in channels_to_drop if ch in raw.ch_names]

raw.drop_channels(existing_to_drop)
print(f"Step 3/5: Dropped problematic channels for visualization: {existing_to_drop}")

# Now, we can safely visualize the sensor locations of the remaining channels
print("Displaying final sensor locations...")
raw.plot_sensors(show_names=True)
plt.show()

# --- 5. Epoching Data into Trials ---
# Extract events (T1 for left fist, T2 for right fist) from annotations
events, event_id = mne.events_from_annotations(raw)
class_ids = {'left_fist': event_id['T1'], 'right_fist': event_id['T2']}

# Create 4-second epochs (trials) starting at the onset of each event
tmin, tmax = 0., 4.
epochs = mne.Epochs(raw, events, event_id=class_ids, tmin=tmin, tmax=tmax,
                    proj=True, picks='eeg', baseline=None, preload=True)

# Visualize the epoched data
print("Step 4/5: Data epoched into trials. Displaying epochs visualization...")
epochs.plot_image(combine='mean', title='Epoched Data (Mean Across Trials)')
plt.show()

# --- 6. Feature Extraction with CSP ---
print("Step 5/6: Applying CSP for feature extraction. Displaying spatial patterns...")
X = epochs.get_data()
y = epochs.events[:, -1]

# Initialize CSP; it will find 4 spatial filters to best separate the classes
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
X_csp = csp.fit_transform(X, y)

# Visualize the CSP spatial patterns (this will now work correctly)
csp.plot_patterns(epochs.info, ch_type='eeg', units='Arbitrary Units', size=1.5)
plt.suptitle('CSP Spatial Patterns')
plt.show()

print("\nPreprocessing and feature extraction complete.")
print(f"The extracted features (X_csp) have the shape: {X_csp.shape}")
print("These features are now ready to be used as input for a Spiking Neural Network or other classifier.")