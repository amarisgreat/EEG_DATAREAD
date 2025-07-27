import mne
from mne.decoding import CSP
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
import numpy as np
import matplotlib.pyplot as plt
import os


file_path = r".\steps\S001R04.edf"
if not os.path.exists(file_path):
    raise FileNotFoundError("EDF file not found. Place it in the current directory.")

raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)


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

raw.plot_sensors(kind='topomap', show_names=True)

raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
raw.plot_psd(fmin=2, fmax=40, average=True)
plt.savefig("filtered_psd.png")
plt.close()


ica = ICA(n_components=15, random_state=42, method='fastica')
ica.fit(raw)

try:
    eog_epochs = create_eog_epochs(raw)
    eog_inds, _ = ica.find_bads_eog(eog_epochs)
    ica.exclude.extend(eog_inds)
except Exception:
    print("No EOG artifacts detected")

try:
    ecg_epochs = create_ecg_epochs(raw)
    ecg_inds, _ = ica.find_bads_ecg(ecg_epochs)
    ica.exclude.extend(ecg_inds)
except Exception:
    print("No ECG artifacts detected")

ica.apply(raw)
fig = ica.plot_overlay(raw, exclude=ica.exclude, show=True)
if not os.path.exists("figures"):
    os.makedirs("figures")
fig.savefig("figures/ica_overlay.png", dpi=300)


events, event_id = mne.events_from_annotations(raw)
class_ids = {'left_fist': event_id['T1'], 'right_fist': event_id['T2']}

epochs = mne.Epochs(raw, events, event_id=class_ids, tmin=-1., tmax=4.,
                    picks='eeg', baseline=(None, 0), preload=True)

labels = epochs.events[:, -1]

X = epochs.get_data()
csp = CSP(n_components=4, log=True)

X_csp = csp.fit_transform(X, labels)
# A. Print shape of CSP features
print("CSP output shape:", X_csp.shape)  # Should be (n_epochs, n_components)

# B. Plot CSP patterns (spatial filters)
fig_patterns = csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', show=True)
fig_patterns.savefig("figures/csp_patterns.png", dpi=300)

# C. OPTIONAL: Plot CSP features across trials to see separability
import seaborn as sns
import pandas as pd

# Create DataFrame for plotting
df = pd.DataFrame(X_csp, columns=[f"CSP{i+1}" for i in range(X_csp.shape[1])])
df['label'] = labels

sns.pairplot(df, hue="label", diag_kind="hist")
plt.suptitle("CSP Component Distributions by Class")
plt.savefig("figures/csp_distributions.png", dpi=300)
plt.close()



bands = {'alpha': (8, 12), 'beta': (13, 30)}
X_erd = []

for band_name, (fmin, fmax) in bands.items():
    band_epochs = epochs.copy().filter(fmin, fmax, fir_design='firwin')
    data = band_epochs.get_data()

    baseline_power = np.mean(data[:, :, :int(1*sfreq)] ** 2, axis=2)  # −1 to 0 sec
    task_power = np.mean(data[:, :, int(1*sfreq):] ** 2, axis=2)      # 0 to 4 sec

    erd = 10 * np.log10(task_power / baseline_power)
    X_erd.append(erd)

X_erd_combined = np.concatenate(X_erd, axis=1)  


# ERD/ERS Visualization (Topomap)
erd_mean = np.mean(erd, axis=0)  # average across epochs
erd_std = np.std(erd, axis=0)

for band_name, (fmin, fmax) in bands.items():
    for class_label, class_name in zip([1, 2], ['left_fist', 'right_fist']):
        class_epochs = epochs[labels == class_label].copy().filter(fmin, fmax, fir_design='firwin')
        data = class_epochs.get_data()

        baseline_power = np.mean(data[:, :, :int(1*sfreq)] ** 2, axis=2)  # −1 to 0 sec
        task_power = np.mean(data[:, :, int(1*sfreq):] ** 2, axis=2)      # 0 to 4 sec

        erd = 10 * np.log10(task_power / baseline_power)
        erd_avg = np.mean(erd, axis=0)

        fig, ax = plt.subplots()
        mne.viz.plot_topomap(erd_avg, epochs.info, axes=ax, show=False)
        ax.set_title(f"{band_name.capitalize()} ERD/ERS - {class_name}", fontsize=10)
        fig.savefig(f"figures/erders_{band_name}_{class_name}.png", dpi=300)
        plt.close(fig)



X_combined = np.concatenate([X_csp, X_erd_combined], axis=1)


np.save("features_combined.npy", X_combined)
np.save("labels.npy", labels)
np.savetxt("features_combined.csv", X_combined, delimiter=",")
np.savetxt("labels.csv", labels, delimiter=",")

print("Pipeline complete.")
print(f"Feature shape (CSP + ERD/ERS): {X_combined.shape}")
