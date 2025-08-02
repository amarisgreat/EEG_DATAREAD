import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.decoding import CSP
from scipy.io import loadmat

# === Constants and mappings ===
class_name_map = {
    'rest': 0,
    'real_left_fist': 1,
    'real_right_fist': 2,
    'real_both_fists': 3,
    'real_feet': 4,
    'imagined_left_fist': 5,
    'imagined_right_fist': 6,
    'imagined_both_fists': 7,
    'imagined_feet': 8,
}
edf_event_map = {
    1: class_name_map['rest'],
    2: class_name_map['real_left_fist'],
    3: class_name_map['real_right_fist'],
    4: class_name_map['real_both_fists'],
    5: class_name_map['real_feet'],
    6: class_name_map['imagined_left_fist'],
    7: class_name_map['imagined_right_fist'],
    8: class_name_map['imagined_both_fists'],
    9: class_name_map['imagined_feet'],
}
sfreq = 160  
fmin, fmax = 8, 30  # frequency band for motor imagery

def preprocess_eeg_file(edf_path, output_dir):
    base_filename = os.path.splitext(os.path.basename(edf_path))[0]

    # === Load and resample ===
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.resample(sfreq, npad="auto")
    raw.pick_types(eeg=True)
    raw.set_montage("standard_1020")

    # === Filter EEG data ===
    raw.filter(fmin, fmax, fir_design='firwin', skip_by_annotation='edge')

    # === Annotations to events ===
    events, event_id = mne.events_from_annotations(raw)
    labels = np.array([edf_event_map.get(code, -1) for code in events[:, 2]])
    valid = labels >= 0
    events = events[valid]
    labels = labels[valid]

    # === Epoching ===
    epochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=2,
                        baseline=None, detrend=1, preload=True, verbose=False)
    X = epochs.get_data()
    y = labels
    unique_classes = np.unique(y)

    print(f"Data shape: X = {X.shape}, y = {y.shape}")
    print(f"Classes present: {unique_classes}")

    # === CSP Topomap Visualization ===
    for class_id in unique_classes:
        if class_id == class_name_map['rest']:
            continue

        y_binary = (y == class_id).astype(int)

        if len(np.unique(y_binary)) < 2:
            print(f"Skipping class {class_id} due to insufficient samples.")
            continue

        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        csp.fit(X, y_binary)

        fig = plt.figure(figsize=(10, 4))
        csp.plot_patterns(epochs.info, ch_type='eeg', components=[0, 1, 2, 3], show=False)
        plt.suptitle(f'CSP Topomap - Class {class_id}', fontsize=14)
        plt.tight_layout()

        vis_dir = os.path.join(output_dir, 'csp_visuals')
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, f"{base_filename}_class{class_id}_csp.png"), dpi=300)
        plt.close(fig)

# === Example usage ===
edf_path = "E:/AMAR/ROBOARM/DATASET/files/S001/S001R04.edf"
output_dir = "preprocessed_data_10051"
preprocess_eeg_file(edf_path, output_dir)
