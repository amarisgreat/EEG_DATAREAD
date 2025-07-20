import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# -------- Load and Filter Data --------
raw = mne.io.read_raw_edf(r"E:\AMAR\ROBOARM\DATASET\files\S001\S001R03.edf", preload=True)
raw.rename_channels(lambda x: x.strip('.'))  # Remove trailing dots if any
raw.filter(7., 30., fir_design='firwin')     # Filter to Mu (8–12 Hz) + Beta (13–30 Hz) bands
raw.set_montage("standard_1020", match_case=False)

# -------- Select Only Motor Cortex Channels --------
print(raw.info['ch_names'])

channels_of_interest = ['C3', 'Cz', 'C4', 'Cpz']
raw.pick_channels(channels_of_interest)

# Optional: Visualize sensor positions
raw.plot_sensors(kind='topomap', show_names=True)

# -------- Extract Events --------
event_id = {'T1': 1, 'T2': 2}  # T1 = left fist, T2 = right fist
events, _ = mne.events_from_annotations(raw, event_id=event_id)

# -------- Epoch the Data --------
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0., tmax=4.,
                    baseline=None, preload=True)

labels = epochs.events[:, -1] - 1  # Convert T1 (1) → 0 and T2 (2) → 1
X = epochs.get_data()  # Shape: (n_trials, n_channels, n_times)

# -------- Train/Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# -------- Classification with CSP + Logistic Regression --------
csp = CSP(n_components=4, log=True, norm_trace=False)
clf = Pipeline([('csp', csp), ('lr', LogisticRegression(solver='liblinear'))])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# -------- Evaluation --------
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=["Left Fist (T1)", "Right Fist (T2)"]
))

# -------- CSP Topomap Patterns --------
csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
plt.show()
