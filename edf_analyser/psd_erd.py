import mne
from mne.preprocessing import ICA, create_eog_epochs
from mne.time_frequency import psd_array_welch
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def get_class_ids_from_run(run_number):
    if run_number in [3, 7, 11]:
        return {'rest': 0, 'real_left_fist': 1, 'real_right_fist': 2}
    elif run_number in [4, 8, 12]:
        return {'rest': 0, 'imagined_left_fist': 3, 'imagined_right_fist': 4}
    elif run_number in [5, 9, 13]:
        return {'rest': 0, 'real_both_fists': 5, 'real_both_feet': 6}
    elif run_number in [6, 10, 14]:
        return {'rest': 0, 'imagined_both_fists': 7, 'imagined_both_feet': 8}
    else:
        return None


def preprocess_eeg_file(file_path, output_dir):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    match = re.search(r'R(\d+)\.edf$', file_path)
    if not match:
        print(f"Could not extract run number from {file_path}. Skipping.")
        return
    run_number = int(match.group(1))

    class_name_map = get_class_ids_from_run(run_number)
    if class_name_map is None:
        print(f"Skipping baseline run: {file_path}")
        return

    base_filename = os.path.basename(file_path).replace('.edf', '')
    print(f"Processing {base_filename} with tasks: {list(class_name_map.keys())}")

    # Channel mapping
    new_names = {ch: ch.replace('.', '').upper() for ch in raw.ch_names}
    raw.rename_channels(new_names)
    mapping = {
        'FCZ': 'FCz', 'CZ': 'Cz', 'CPZ': 'CPz', 'FP1': 'Fp1', 'FPZ': 'Fpz',
        'FP2': 'Fp2', 'AFZ': 'AFz', 'FZ': 'Fz', 'PZ': 'Pz', 'POZ': 'POz',
        'OZ': 'Oz', 'IZ': 'Iz'
    }
    raw.rename_channels(mapping)
    raw.set_montage(mne.channels.make_standard_montage('standard_1005'))

    # Filter
    sfreq = raw.info['sfreq']
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    # ICA for EOG removal
    ica = ICA(n_components=15, random_state=42, method='fastica', max_iter=800)
    ica.fit(raw)

    try:
        eog_epochs = create_eog_epochs(raw)
        eog_inds, _ = ica.find_bads_eog(eog_epochs)
        if eog_inds:
            ica.exclude.extend(eog_inds)
    except Exception:
        print(f"No EOG channels or ICA error for {base_filename}.")

    ica.apply(raw)

    # Event extraction
    try:
        events, event_id_from_file = mne.events_from_annotations(raw)
    except ValueError:
        print(f"No events found in {base_filename}. Skipping.")
        return

    event_mapping = {
        'rest': event_id_from_file['T0'],
        list(class_name_map.keys())[1]: event_id_from_file['T1'],
        list(class_name_map.keys())[2]: event_id_from_file['T2'],
    }

    # Epochs
    epochs = mne.Epochs(raw, events, event_id=event_mapping, tmin=-1., tmax=4.,
                        picks='eeg', baseline=(None, 0), preload=True)
    data = epochs.get_data()  # shape: (n_trials, n_channels, n_times)
    labels = epochs.events[:, -1]
    n_trials, n_channels, n_times = data.shape

    # === PSD Feature Extraction ===
    psd_fmin, psd_fmax = 8, 30
    n_fft = int(sfreq * 1.0)
    X_psd_all = []

    for trial in data:
        psd, freqs = psd_array_welch(trial, sfreq=sfreq, fmin=psd_fmin, fmax=psd_fmax,
                                     n_fft=n_fft, average='mean', verbose=False)
        X_psd_all.append(psd.flatten())
    X_psd_all = np.array(X_psd_all)

    # === ERD/ERS Feature Extraction ===
    bands = {'alpha': (8, 12), 'beta': (13, 30)}
    X_erd_all = []

    for _, (fmin, fmax) in bands.items():
        band_epochs = epochs.copy().filter(fmin, fmax, fir_design='firwin', verbose=False)
        band_data = band_epochs.get_data()
        baseline_power = np.mean(band_data[:, :, :int(1 * sfreq)] ** 2, axis=2, keepdims=True)
        task_power = np.mean(band_data[:, :, int(1 * sfreq):] ** 2, axis=2)
        erd_ers = 10 * np.log10(task_power / np.squeeze(baseline_power))
        X_erd_all.append(erd_ers)

    X_erd_combined = np.concatenate(X_erd_all, axis=1)

    # Combine features
    X_combined = np.concatenate([X_psd_all, X_erd_combined], axis=1)

    # Save
    features_dir = os.path.join(output_dir, 'features')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    np.save(os.path.join(features_dir, f"{base_filename}_features.npy"), X_combined)
    np.save(os.path.join(labels_dir, f"{base_filename}_labels.npy"), labels)
    print(f"Saved {base_filename}. Feature shape: {X_combined.shape}, Labels: {np.unique(labels)}")

# === Batch Processing ===
if __name__ == "__main__":
    RECORDS_FILE = r"E:\AMAR\ROBOARM\DATASET\files\RECORDS"
    DATA_DIR = r"E:\AMAR\ROBOARM\DATASET\files"
    OUTPUT_DIR = "preprocessed_data_psd_erd"

    if not os.path.exists(RECORDS_FILE):
        raise FileNotFoundError(f"'{RECORDS_FILE}' not found.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(RECORDS_FILE, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]

    for file_rel_path in file_list:
        file_full_path = os.path.join(DATA_DIR, file_rel_path)
        if os.path.exists(file_full_path):
            preprocess_eeg_file(file_full_path, OUTPUT_DIR)
        else:
            print(f"File not found: {file_full_path}. Skipping.")

    print("\n--- All files processed. ---")
