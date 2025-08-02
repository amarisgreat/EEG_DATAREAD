import mne
from mne.preprocessing import ICA, create_eog_epochs
import numpy as np
import os
import re

# === Class Label Mapping ===
CLASS_LABELS = {
    'T0': (0, 'Rest'),
    'T1': {
        3: (1, 'Real Left Fist'), 4: (3, 'Imagined Left Fist'),
        5: (5, 'Real Both Fists'), 6: (7, 'Imagined Both Fists'),
        7: (1, 'Real Left Fist'), 8: (3, 'Imagined Left Fist'),
        9: (5, 'Real Both Fists'), 10: (7, 'Imagined Both Fists'),
        11: (1, 'Real Left Fist'), 12: (3, 'Imagined Left Fist'),
        13: (5, 'Real Both Fists'), 14: (7, 'Imagined Both Fists')
    },
    'T2': {
        3: (2, 'Real Right Fist'), 4: (4, 'Imagined Right Fist'),
        5: (6, 'Real Both Feet'), 6: (8, 'Imagined Both Feet'),
        7: (2, 'Real Right Fist'), 8: (4, 'Imagined Right Fist'),
        9: (6, 'Real Both Feet'), 10: (8, 'Imagined Both Feet'),
        11: (2, 'Real Right Fist'), 12: (4, 'Imagined Right Fist'),
        13: (6, 'Real Both Feet'), 14: (8, 'Imagined Both Feet')
    }
}


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

    base_filename = os.path.basename(file_path).replace('.edf', '')
    print(f"\n📄 Processing: {base_filename} | Run: {run_number}")

    # --- Channel Renaming & Montage ---
    new_names = {ch: ch.replace('.', '').upper() for ch in raw.ch_names}
    raw.rename_channels(new_names)
    mapping = {
        'FCZ': 'FCz', 'CZ': 'Cz', 'CPZ': 'CPz', 'FP1': 'Fp1', 'FPZ': 'Fpz',
        'FP2': 'Fp2', 'AFZ': 'AFz', 'FZ': 'Fz', 'PZ': 'Pz', 'POZ': 'POz',
        'OZ': 'Oz', 'IZ': 'Iz'
    }
    raw.rename_channels(mapping)
    raw.set_montage(mne.channels.make_standard_montage('standard_1005'))

    # --- Filtering ---
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    # --- ICA (Artifact Removal) ---
    ica = ICA(n_components=15, random_state=42, method='fastica', max_iter=800)
    ica.fit(raw)

    try:
        eog_epochs = create_eog_epochs(raw)
        eog_inds, _ = ica.find_bads_eog(eog_epochs)
        if eog_inds:
            ica.exclude.extend(eog_inds)
    except Exception:
        print(f"No EOG data or error in EOG detection for {base_filename}")

    ica.apply(raw)

    # --- Extract Events ---
    try:
        events, event_id_from_file = mne.events_from_annotations(raw)
        event_mapping = {
            'T0': event_id_from_file['T0'],
            'T1': event_id_from_file['T1'],
            'T2': event_id_from_file['T2']
        }
    except Exception as e:
        print(f"Could not extract events from {base_filename}: {e}")
        return

    epochs = mne.Epochs(raw, events, event_id=event_mapping, tmin=-1., tmax=4.,
                        picks='eeg', baseline=(None, 0), preload=True)

    raw_labels = epochs.events[:, -1]  # T0, T1, T2
    label_ids = []

    for raw_label in raw_labels:
        label_key = [k for k, v in event_mapping.items() if v == raw_label][0]
        if label_key == 'T0':
            label_ids.append(CLASS_LABELS['T0'][0])
        else:
            if run_number not in CLASS_LABELS[label_key]:
                print(f" No label defined for {label_key} in run {run_number}")
                continue
            label_ids.append(CLASS_LABELS[label_key][run_number][0])

    if len(label_ids) != len(epochs):
        print(f" Mismatch between labels and epochs for {base_filename}. Skipping.")
        return

    labels = np.array(label_ids)

    # --- ERD/ERS Feature Extraction ---
    sfreq = raw.info['sfreq']
    bands = {'alpha': (8, 12), 'beta': (13, 30)}
    X_erd_all = []

    for band_name, (fmin, fmax) in bands.items():
        band_epochs = epochs.copy().filter(fmin, fmax, fir_design='firwin', verbose=False)
        data = band_epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

        baseline_power = np.mean(data[:, :, :int(1 * sfreq)] ** 2, axis=2, keepdims=True)
        baseline_power = np.squeeze(baseline_power)

        task_power = np.mean(data[:, :, int(1 * sfreq):] ** 2, axis=2)
        erd_ers = 10 * np.log10(task_power / baseline_power)
        X_erd_all.append(erd_ers)

    X_erd_combined = np.concatenate(X_erd_all, axis=1)

    # --- Save Features and Labels ---
    features_dir = os.path.join(output_dir, 'features')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    np.save(os.path.join(features_dir, f"{base_filename}_features.npy"), X_erd_combined)
    np.save(os.path.join(labels_dir, f"{base_filename}_labels.npy"), labels)

    print(f" Saved: {base_filename} | Features: {X_erd_combined.shape} | Labels: {labels.shape}")


if __name__ == "__main__":
    RECORDS_FILE = r"E:\AMAR\ROBOARM\DATASET\files\RECORDS"
    DATA_DIR = r"E:\AMAR\ROBOARM\DATASET\files"
    OUTPUT_DIR = "preprocessed_data_erders_only_custom_labels"

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
            print(f" File not found: {file_full_path}. Skipping.")

    print("\n---  All files processed ---")
