import mne

# Load raw EEG from EDF
raw = mne.io.read_raw_edf(r"./steps/S001R04.edf", preload=True)


raw.rename_channels({ch: ch.replace('.', '').upper() for ch in raw.ch_names})


overlapping_chs = ['FCZ', 'CZ', 'CPZ', 'FP1', 'FPZ', 'FP2', 'AFZ', 'FZ', 'PZ', 'POZ', 'OZ', 'IZ']
raw.drop_channels([ch for ch in overlapping_chs if ch in raw.ch_names])


raw.set_montage("standard_1020", on_missing="ignore")


raw.filter(1, 40)

ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter="auto")
ica.fit(raw)


ica.plot_components()
