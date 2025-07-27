import mne
import matplotlib.pyplot as plt
raw = mne.io.read_raw_edf(r".\steps\S001R04.edf", preload=True)
print(raw.info)
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

raw.plot_sensors(kind='topomap', show_names=True)
plt.show()