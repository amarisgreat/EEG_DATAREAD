import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report


raw = mne.io.read_raw_edf(r"E:\AMAR\ROBOARM\DATASET\files\S001\S001R03.edf", preload=True)
raw.rename_channels(lambda ch: ch.strip('.'))
raw.filter(7., 30., fir_design='firwin') 
raw.set_montage('standard_1020', match_case=False)
raw.pick_types(eeg=True, exclude='bads')


event_id = {'T1': 1, 'T2': 2}  
events, _ = mne.events_from_annotations(raw, event_id=event_id)



epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0., tmax=4.0,
                    baseline=None, preload=True)

labels = epochs.events[:, -1] - 1  
X = epochs.get_data()

motor_channels = ['C3', 'C4', 'Cz', 'Fcz']
epochs_motor = epochs.copy().pick_channels(motor_channels)


epochs_motor['T1'].average().plot(spatial_colors=True, titles='Left Fist (T1) - Motor Channels')
epochs_motor['T2'].average().plot(spatial_colors=True, titles='Right Fist (T2) - Motor Channels')



epochs['T1'].average().plot(spatial_colors=True, titles='Left Fist (T1)')
epochs['T2'].average().plot(spatial_colors=True, titles='Right Fist (T2)')


epochs['T1'].average().plot_topomap(times=[1.0], ch_type='eeg', title='Left MI (T1) @ 1s')
epochs['T2'].average().plot_topomap(times=[1.0], ch_type='eeg', title='Right MI (T2) @ 1s')


psds_T1, freqs = mne.time_frequency.psd_welch(epochs['1'], fmin=8, fmax=30, n_fft=256)
psds_T2, _     = mne.time_frequency.psd_welch(epochs['2'], fmin=8, fmax=30, n_fft=256)


psds_T1_db = 10 * np.log10(psds_T1.mean(axis=0))
psds_T2_db = 10 * np.log10(psds_T2.mean(axis=0))

plt.figure(figsize=(10, 5))
for ch in range(4):  # First 4 EEG channels
    plt.plot(freqs, psds_T1_db[ch], label=f'Left - {epochs.ch_names[ch]}')
    plt.plot(freqs, psds_T2_db[ch], linestyle='--', label=f'Right - {epochs.ch_names[ch]}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Mu/Beta Band Power: Left vs Right Motor Imagery')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# csp = CSP(n_components=4, log=True, norm_trace=False)
# clf = Pipeline([('csp', csp), ('lr', LogisticRegression(solver='liblinear'))])

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, target_names=["Left Fist (T1)", "Right Fist (T2)"]))


# csp.plot_patterns(epochs.info, ch_type='eeg', units='AU', size=1.5)
# plt.show()
