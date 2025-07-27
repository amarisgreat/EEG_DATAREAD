import mne
from mne.decoding import CSP
from mne.preprocessing import ICA, create_eog_epochs
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def get_class_ids_from_run(run_number):
    """
    Determines the correct class labels based on the experimental run number.
    """
    if run_number in [3, 7, 11]:
        return {'rest': 0, 'real_left_fist': 1, 'real_right_fist': 2}
    elif run_number in [4, 8, 12]:
        return {'rest': 0, 'imagined_left_fist': 1, 'imagined_right_fist': 2}
    elif run_number in [5, 9, 13]:
        return {'rest': 0, 'real_both_fists': 1, 'real_both_feet': 2}
    elif run_number in [6, 10, 14]:
        return {'rest': 0, 'imagined_both_fists': 1, 'imagined_both_feet': 2}
    else:
        # For baseline runs (1, 2) or any other case
        return None

def preprocess_eeg_file(file_path, output_dir):
    """
    Applies the full preprocessing pipeline to a single EDF file.
    """
    # --- 1. Load and Prepare Data ---
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Extract run number from filename
    match = re.search(r'R(\d+)\.edf$', file_path)
    if not match:
        print(f"Could not extract run number from {file_path}. Skipping.")
        return
    run_number = int(match.group(1))

    # Get the correct class mapping for this run
    class_name_map = get_class_ids_from_run(run_number)
    if class_name_map is None:
        print(f"Skipping baseline run: {file_path}")
        return
    
    base_filename = os.path.basename(file_path).replace('.edf', '')
    print(f"Processing {base_filename} with tasks: {list(class_name_map.keys())}")


    # --- 2. Channel Names and Montage ---
    new_names = {ch: ch.replace('.', '').upper() for ch in raw.ch_names}
    raw.rename_channels(new_names)
    mapping = {
        'FCZ': 'FCz', 'CZ': 'Cz', 'CPZ': 'CPz', 'FP1': 'Fp1', 'FPZ': 'Fpz',
        'FP2': 'Fp2', 'AFZ': 'AFz', 'FZ': 'Fz', 'PZ': 'Pz', 'POZ': 'POz',
        'OZ': 'Oz', 'IZ': 'Iz'
    }
    raw.rename_channels(mapping)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    sfreq = raw.info['sfreq']
    print(f" sampling frequency: {sfreq}")
    print(f"Temporal resolution: {1/sfreq:.4f} seconds")
    # --- 3. Filtering ---
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    # --- 4. ICA for Artifact Removal ---
    ica = ICA(n_components=15, random_state=42, method='fastica', max_iter=800)
    ica.fit(raw)

    try:
        eog_epochs = create_eog_epochs(raw)
        eog_inds, _ = ica.find_bads_eog(eog_epochs)
        if eog_inds:
            ica.exclude.extend(eog_inds)
    except Exception:
        print(f"No EOG channels found or other EOG error for {base_filename}.")

    # ICA is not great at finding ECG from EEG data without a dedicated ECG channel
    # This part is often skipped if no ECG channel is present.
    # If you had an ECG channel, you would use ica.find_bads_ecg

    ica.apply(raw)

    # --- 5. Event Extraction ---
    try:
        events, event_id_from_file = mne.events_from_annotations(raw)
    except ValueError:
        print(f"No events found in {base_filename}. Skipping.")
        return
        
    # Map T0, T1, T2 from file to our new labels
# Fix: Use string keys as required by MNE
    event_mapping = {
        'rest': event_id_from_file['T0'],
        list(class_name_map.keys())[1]: event_id_from_file['T1'],
        list(class_name_map.keys())[2]: event_id_from_file['T2'],
    }

    
    # Create epochs with the new labels
    epochs = mne.Epochs(raw, events, event_id=event_mapping, tmin=-1., tmax=4.,
                        picks='eeg', baseline=(None, 0), preload=True)

    labels = epochs.events[:, -1]
    X = epochs.get_data()

    # --- 6. CSP Feature Extraction ---
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print(f"Not enough classes to compute CSP for {base_filename}. Skipping.")
        return
        
    X_csp_all = []
    for class_id in unique_classes:
        if class_id == class_name_map['rest']: continue # Don't compute CSP against rest
        
        # Create a binary problem for one-vs-rest CSP
        y_binary = (labels == class_id).astype(int)
        
        # Check if we have at least two classes in the binary problem
        if len(np.unique(y_binary)) < 2:
            print(f"Skipping CSP for class {class_id} in {base_filename} due to lack of class balance.")
            continue

        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        X_csp_i = csp.fit_transform(X, y_binary)
        X_csp_all.append(X_csp_i)

    if not X_csp_all:
        print(f"CSP could not be computed for any class in {base_filename}. Skipping.")
        return

    X_csp_combined = np.concatenate(X_csp_all, axis=1)

    # --- 7. ERD/ERS Feature Extraction ---
    bands = {'alpha': (8, 12), 'beta': (13, 30)}
    X_erd_all = []
    sfreq = raw.info['sfreq']

    for band_name, (fmin, fmax) in bands.items():
        band_epochs = epochs.copy().filter(fmin, fmax, fir_design='firwin', verbose=False)
        data = band_epochs.get_data()

        # Baseline is from -1 to 0 seconds (the first second of the epoch)
        baseline_power = np.mean(data[:, :, :int(1 * sfreq)] ** 2, axis=2, keepdims=True)
        baseline_power = np.squeeze(baseline_power)  # Removes the last dimension

        # Task is from 0 to 4 seconds
        task_power = np.mean(data[:, :, int(1 * sfreq):] ** 2, axis=2)

        # ERD/ERS formula: log(task_power / baseline_power)
        erd_ers = 10 * np.log10(task_power / baseline_power)
        X_erd_all.append(erd_ers)
    
    X_erd_combined = np.concatenate(X_erd_all, axis=1)

    # --- 8. Combine Features and Save ---
    X_combined = np.concatenate([X_csp_combined, X_erd_combined], axis=1)


    features_dir = os.path.join(output_dir, 'features')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    np.save(os.path.join(features_dir, f"{base_filename}_features.npy"), X_combined)
    np.save(os.path.join(labels_dir, f"{base_filename}_labels.npy"), labels)
    
    print(f"Successfully processed and saved {base_filename}. Feature shape: {X_combined.shape}")



if __name__ == "__main__":
    RECORDS_FILE = r"E:\AMAR\ROBOARM\DATASET\files\RECORDS"
    DATA_DIR = r"E:\AMAR\ROBOARM\DATASET\files" 
    OUTPUT_DIR = "preprocessed_data"

    if not os.path.exists(RECORDS_FILE):
        raise FileNotFoundError(f"'{RECORDS_FILE}' not found. Please ensure it's in the directory.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(RECORDS_FILE, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]

    for file_rel_path in file_list:
        file_full_path = os.path.join(DATA_DIR, file_rel_path)
        if os.path.exists(file_full_path):
            preprocess_eeg_file(file_full_path, OUTPUT_DIR)
        else:
            print(f"File not found: {file_full_path}. Skipping.")

    print("\n\n--- All files processed. ---")