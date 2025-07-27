import mne

# Load raw EEG from EDF
raw = mne.io.read_raw_edf(r"./steps/S001R04.edf", preload=True)

# Fix channel names: strip dots, uppercase
raw.rename_channels({ch: ch.replace('.', '').upper() for ch in raw.ch_names})

# Drop channels that cause 3D plot overlap
overlapping_chs = ['FCZ', 'CZ', 'CPZ', 'FP1', 'FPZ', 'FP2', 'AFZ', 'FZ', 'PZ', 'POZ', 'OZ', 'IZ']
raw.drop_channels([ch for ch in overlapping_chs if ch in raw.ch_names])

# Set standard montage (after dropping)
raw.set_montage("standard_1020", on_missing="ignore")

# Apply bandpass filter
raw.filter(1, 40)

# Run ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter="auto")
ica.fit(raw)

# Plot ICA components (all of them)
ica.plot_components()
