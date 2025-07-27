import mne
from mne.decoding import CSP
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

base_dir = Path(r"E:\New Volume\AMAR\ROBOARM\DATASET\files")
output_dir = base_dir / "all_subjects_output"
output_dir.mkdir(exist_ok=True)

def get_run_task(run_number):
    """
    Maps run index (1-based) to task description.
    """
    if run_number in [1, 2]:
        return 'baseline'
    elif run_number in [3, 7, 11]:
        return 'real_left_right_fist'
    elif run_number in [4, 8, 12]:
        return 'imagined_left_right_fist'
    elif run_number in [5, 9, 13]:
        return 'real_fists_feet'
    elif run_number in [6, 10, 14]:
        return 'imagined_fists_feet'
    else:
        return 'unknown'

# Loop through all subjects S001 to S109
for subj_id in range(1, 110):
    subject = f"S{subj_id:03d}"
    subject_path = base_dir / subject
    if not subject_path.exists():
        print(f"Skipping {subject}: Folder not found.")
        continue

    edf_files = sorted([f for f in os.listdir(subject_path) if f.endswith(".edf")])
    
    for edf_file in edf_files:
        run_number = int(edf_file.split('R')[-1].split('.')[0])
        run_label = get_run_task(run_number)
        file_path = subject_path / edf_file
        run_name = f"{subject}_R{run_number:02d}_{run_label}"
        print(f"\n=== Processing {run_name} ===")

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

            # Rename channels
            new_names = {ch: ch.replace('.', '').upper() for ch in raw.ch_names}
            raw.rename_channels(new_names)
            mapping = {
                'FCZ': 'FCz', 'CZ': 'Cz', 'CPZ': 'CPz', 'FP1': 'Fp1', 'FPZ': 'Fpz',
                'FP2': 'Fp2', 'AFZ': 'AFz', 'FZ': 'Fz', 'PZ': 'Pz', 'POZ': 'POz',
                'OZ': 'Oz', 'IZ': 'Iz'
            }
            raw.rename_channels(mapping)

            # Set montage
            raw.set_montage(mne.channels.make_standard_montage('standard_1020'))

            # Filter signal
            raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

            # ICA for artifact removal
            ica = ICA(n_components=15, random_state=42, method='fastica')
            ica.fit(raw)
            try:
                eog_epochs = create_eog_epochs(raw)
                eog_inds, _ = ica.find_bads_eog(eog_epochs)
                ica.exclude.extend(eog_inds)
            except Exception:
                pass
            try:
                ecg_epochs = create_ecg_epochs(raw)
                ecg_inds, _ = ica.find_bads_ecg(ecg_epochs)
                ica.exclude.extend(ecg_inds)
            except Exception:
                pass
            ica.apply(raw)

            # Annotations and Epochs
            events, event_id = mne.events_from_annotations(raw)
            if not all(k in event_id for k in ["T1", "T2"]):
                print(f"⚠️ Missing T1/T2 in {run_name}, skipping.")
                continue

            class_ids = {'class1': event_id['T1'], 'class2': event_id['T2']}
            epochs = mne.Epochs(raw, events, event_id=class_ids, tmin=-1., tmax=4.,
                                picks='eeg', baseline=(None, 0), preload=True)
            labels = epochs.events[:, -1]
            X = epochs.get_data()

            # CSP features
            csp = CSP(n_components=4, log=True)
            X_csp = csp.fit_transform(X, labels)

            # ERD features
            sfreq = raw.info['sfreq']
            bands = {'alpha': (8, 12), 'beta': (13, 30)}
            X_erd = []

            for band_name, (fmin, fmax) in bands.items():
                band_epochs = epochs.copy().filter(fmin, fmax, fir_design='firwin')
                data = band_epochs.get_data()
                baseline_power = np.mean(data[:, :, :int(1 * sfreq)] ** 2, axis=2)
                task_power = np.mean(data[:, :, int(1 * sfreq):] ** 2, axis=2)
                erd = 10 * np.log10(task_power / baseline_power)
                X_erd.append(erd)

            X_erd_combined = np.concatenate(X_erd, axis=1)
            X_combined = np.concatenate([X_csp, X_erd_combined], axis=1)

            # Save
            np.save(output_dir / f"{run_name}_features.npy", X_combined)
            np.save(output_dir / f"{run_name}_labels.npy", labels)
            np.savetxt(output_dir / f"{run_name}_features.csv", X_combined, delimiter=",")
            np.savetxt(output_dir / f"{run_name}_labels.csv", labels, delimiter=",")
            print(f"✅ Saved {run_name}: shape={X_combined.shape}")

        except Exception as e:
            print(f"❌ Error processing {run_name}: {e}")
