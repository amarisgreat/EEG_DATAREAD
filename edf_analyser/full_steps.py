# import mne
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mne.preprocessing import ICA
# from mne.decoding import CSP
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# # ---- Configuration ----
# edf_path = "E:/AMAR/ROBOARM/DATASET/files/S002/S002R04.edf"  # Path to a PhysioNet MI recording

# # ---- Load raw EEG ----
# raw = mne.io.read_raw_edf(edf_path, preload=True)
# new_names = {ch: ch.replace('.', '').upper() for ch in raw.ch_names}
# raw.rename_channels(new_names)
# mapping = {
#     'FCZ': 'FCz', 'CZ': 'Cz', 'CPZ': 'CPz', 'FP1': 'Fp1', 'FPZ': 'Fpz',
#     'FP2': 'Fp2', 'AFZ': 'AFz', 'FZ': 'Fz', 'PZ': 'Pz', 'POZ': 'POz',
#     'OZ': 'Oz', 'IZ': 'Iz'
# }
# raw.rename_channels(mapping)
# print(raw.annotations)
# print(raw.info)
# montage = mne.channels.make_standard_montage('standard_1020')
# raw.set_montage(montage)

# # ---- Filter & Notch ----
# raw.filter(7., 30., fir_design='firwin')
# raw.notch_filter(freqs=50)

# # ---- ICA to remove artifacts ----
# ica = ICA(n_components=15, random_state=42)
# ica.fit(raw)

# try:
#     eog_inds, _ = ica.find_bads_eog(raw)
#     ica.exclude = eog_inds
# except Exception as e:
#     print(f"⚠️ Skipping EOG artifact removal: {e}")

# ica.apply(raw)

# # ---- Event creation from annotations ----
# events, event_id = mne.events_from_annotations(raw)
# if not event_id:
#     raise RuntimeError("No valid event annotations found.")

# # ---- Epoching ----
# epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=4,
#                     baseline=None, picks='eeg', preload=True)
# if len(epochs) < 2:
#     raise RuntimeError("Not enough epochs to continue.")

# # ---- Extract Data ----
# X = epochs.get_data()
# y = epochs.events[:, -1]  # Use true labels from annotations

# # ---- CSP Feature Extraction ----
# csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)
# X_csp = csp.fit_transform(X, y)

# # ---- Save Features ----
# df = pd.DataFrame(X_csp, columns=[f'CSP_{i+1}' for i in range(X_csp.shape[1])])
# df['label'] = y
# df.to_csv("csp_features.csv", index=False)

# # ---- Visualize CSP Patterns ----
# csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (uV)', size=1.5)
# csp.plot_filters(epochs.info, ch_type='eeg', units='Filters (AU)', size=1.5)
# plt.suptitle('CSP Spatial Patterns')

# # ---- Classification with LDA ----
# lda = LinearDiscriminantAnalysis()
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(lda, X_csp, y, cv=cv, scoring='accuracy')
# print(f"CSP + LDA Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

# # ---- EEG & PSD Plot ----
# raw.plot(n_channels=10, title="EEG (Post-ICA)")
# raw.plot_psd(fmin=2, fmax=40, average=True)
# plt.show()


#no ica

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ---- Configuration ----
edf_path = "E:/AMAR/ROBOARM/DATASET/files/S002/S002R04.edf"  # Replace with your file

# ---- Load raw EEG ----

raw = mne.io.read_raw_edf(edf_path, preload=True)

# ---- Channel Name Cleanup ----
new_names = {ch: ch.replace('.', '').upper() for ch in raw.ch_names}
raw.rename_channels(new_names)
mapping = {
    'FCZ': 'FCz', 'CZ': 'Cz', 'CPZ': 'CPz', 'FP1': 'Fp1', 'FPZ': 'Fpz',
    'FP2': 'Fp2', 'AFZ': 'AFz', 'FZ': 'Fz', 'PZ': 'Pz', 'POZ': 'POz',
    'OZ': 'Oz', 'IZ': 'Iz'
}
raw.rename_channels(mapping)
raw.set_montage(mne.channels.make_standard_montage('standard_1020'))

# ---- Filtering ----
raw.filter(7., 30., fir_design='firwin')
raw.notch_filter(freqs=50)

# ---- Event extraction ----
events, event_id = mne.events_from_annotations(raw)
if not event_id:
    raise RuntimeError("No valid event annotations found.")

# ---- Epoching ----
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.0, tmax=4.0,
                    baseline=None, picks='eeg', preload=True)

if len(epochs) < 2:
    raise RuntimeError("Not enough epochs to continue.")

# ---- Extract Data ----
X = epochs.get_data()           # shape = (n_trials, n_channels, n_times)
y = epochs.events[:, -1]        # labels (e.g., 1 = T0, 2 = T1, 3 = T2)

# ---- CSP Feature Extraction ----
csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)
X_csp = csp.fit_transform(X, y)

# ---- Save CSP features ----
df = pd.DataFrame(X_csp, columns=[f'CSP_{i+1}' for i in range(X_csp.shape[1])])
df['label'] = y
df.to_csv("csp_features_no_ica.csv", index=False)

# ---- CSP Visualization ----
csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (uV)', size=1.5)
csp.plot_filters(epochs.info, ch_type='eeg', units='Filters (AU)', size=1.5)
plt.suptitle('CSP Spatial Patterns (No ICA)')

# ---- Classification ----
lda = LinearDiscriminantAnalysis()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lda, X_csp, y, cv=cv, scoring='accuracy')
print(f"CSP + LDA Accuracy (No ICA): {np.mean(scores):.2f} ± {np.std(scores):.2f}")

# ---- Raw EEG and PSD ----
raw.plot(n_channels=10, title="EEG (Raw, No ICA)")
raw.plot_psd(fmin=2, fmax=40, average=True)
plt.show()
