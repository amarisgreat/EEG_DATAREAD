import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
import os

# === 1. Load EDF ===
edf_path = r"E:\AMAR\ROBOARM\DATASET\files\S001\S001R04.edf"
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
run_number = int(edf_path.split("R")[-1].split(".")[0])

# === 2. Set Montage ===
new_names = {ch: ch.replace('.', '').upper() for ch in raw.ch_names}
raw.rename_channels(new_names)
mapping = {
    'FCZ': 'FCz', 'CZ': 'Cz', 'CPZ': 'CPz', 'FP1': 'Fp1', 'FPZ': 'Fpz',
    'FP2': 'Fp2', 'AFZ': 'AFz', 'FZ': 'Fz', 'PZ': 'Pz', 'POZ': 'POz',
    'OZ': 'Oz', 'IZ': 'Iz'
}
raw.rename_channels(mapping)
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

# === 3. Sampling Info ===
sfreq = raw.info['sfreq']
print(f"Sampling frequency: {sfreq} Hz")
print(f"Temporal resolution: {1/sfreq:.4f} seconds")

# === 4. PSD Plot ===
raw.plot_psd(fmax=60, show=False)
plt.savefig("psd_plot.png", dpi=300)
plt.close()

# === 5. Filtering & ICA ===
raw.filter(1., 40., fir_design='firwin', verbose=False)
ica = mne.preprocessing.ICA(n_components=15, random_state=42)
ica.fit(raw)
ica.exclude = []
raw = ica.apply(raw)

# === 6. Events & Epoching ===
events, event_dict = mne.events_from_annotations(raw)
event_id = {'T0': 1, 'T1': 2, 'T2': 3}
tmin, tmax = 0.0, 3.0
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    proj=True, baseline=(0, 0.5), preload=True, verbose=False)
labels = epochs.events[:, -1]

# === 7. FBCSP: Multi-band CSP Feature Extraction + Plotting ===
filter_banks = {
    'theta': (4, 7),
    'alpha': (8, 12),
    'beta': (13, 30)
}

X_fbcsp_all = []
for band_name, (fmin, fmax) in filter_banks.items():
    band_epochs = epochs.copy().filter(fmin, fmax, fir_design='firwin', verbose=False)
    band_data = band_epochs.get_data()

    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    X_csp = csp.fit_transform(band_data, labels)
    X_fbcsp_all.append(X_csp)

    fig = plt.figure(figsize=(10, 4))
    csp.plot_patterns(band_epochs.info, ch_type='eeg', components=[0, 1, 2, 3], show=False)
    plt.suptitle(f'CSP Patterns - {band_name} Band ({fmin}-{fmax} Hz)', fontsize=12)

    plt.savefig(f'csp_{band_name}_patterns.png', dpi=300)
    plt.close(fig)

X_fbcsp_combined = np.concatenate(X_fbcsp_all, axis=1)

# === 8. ERD/ERS Feature Extraction ===
erd_tmin, erd_tmax = 0.5, 2.5
erd_epochs = epochs.copy().crop(tmin=erd_tmin, tmax=erd_tmax)
erd_data = erd_epochs.get_data()

X_erd_mean = np.mean(erd_data, axis=2)
X_erd_var = np.var(erd_data, axis=2)
X_erd_combined = np.concatenate([X_erd_mean, X_erd_var], axis=1)

# === 9. ERD/ERS Topomap Plots ===
erd_mean_topo = np.mean(erd_data, axis=0).mean(axis=1)
erd_var_topo = np.var(erd_data, axis=0).mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
mne.viz.plot_topomap(erd_mean_topo, epochs.info, axes=axes[0], show=False)
axes[0].set_title("ERD/ERS - Mean Activity")

mne.viz.plot_topomap(erd_var_topo, epochs.info, axes=axes[1], show=False)
axes[1].set_title("ERD/ERS - Variance Activity")

#plt.tight_layout()
plt.savefig("erd_ers_topomaps.png", dpi=300)
plt.close(fig)

# === 10. Combine Final Features ===
X_combined = np.concatenate([X_fbcsp_combined, X_erd_combined], axis=1)
y = labels

print("X_combined shape:", X_combined.shape)
print("Labels shape:", y.shape)

# === 11. Save Features ===
output_dir = "features"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, f"S{run_number:03d}_X.npy"), X_combined)
np.save(os.path.join(output_dir, f"S{run_number:03d}_y.npy"), y)
print(f"Saved: features/S{run_number:03d}_X.npy, features/S{run_number:03d}_y.npy")
