import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap
from mne.viz.topomap import _prepare_topomap_plot

# ---------- Load EDF EEG ----------
edf_path = "./steps/S001R04.edf"  # Update path
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
raw.rename_channels(lambda x: x.strip('.').upper())
raw.set_montage('standard_1020', on_missing='ignore')
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

# ---------- ICA to remove artifacts ----------
ica = mne.preprocessing.ICA(n_components=15, random_state=42)
ica.fit(raw)
for fn in [mne.preprocessing.create_eog_epochs, mne.preprocessing.create_ecg_epochs]:
    try:
        epochs_temp = fn(raw)
        bads, _ = ica.find_bads_eog(epochs_temp) if 'eog' in fn.__name__ else ica.find_bads_ecg(epochs_temp)
        ica.exclude.extend(bads)
    except Exception:
        continue
ica.apply(raw)

# ---------- Events & Epochs ----------
events, event_id = mne.events_from_annotations(raw)
class_map = {'left': event_id['T1'], 'right': event_id['T2']}
epochs = mne.Epochs(raw, events, event_id=class_map, tmin=-1.0, tmax=4.0,
                    baseline=(None, 0), picks='eeg', preload=True, detrend=1)

# ---------- Compute TFR ----------
freqs = np.linspace(8, 30, 23)
n_cycles = freqs / 2
os.makedirs("figures", exist_ok=True)

for label in class_map:
    class_epochs = epochs[label]
    power = class_epochs.compute_tfr(freqs=freqs,  # <-- FIXED HERE
                                     method='multitaper',
                                     n_cycles=n_cycles,
                                     use_fft=True,
                                     return_itc=False,
                                     decim=2,
                                     n_jobs=1)
    power.apply_baseline(baseline=(-1.0, 0.0), mode='logratio')

    # Get data of shape: (n_channels, n_freqs, n_times)
    data = power.data
    times = power.times
    freqs = power.freqs

    # Focus on 10–25 Hz, 0.5–1.0s window
    fmask = (freqs >= 10) & (freqs <= 25)
    tmask = (times >= 0.5) & (times <= 1.0)

    # Average power in this band and time window
    avg_power = data[:, fmask][:, :, tmask].mean(axis=(1, 2))  # shape: (n_channels,)

    # Get channel positions
    layout = mne.find_layout(power.info)
    pos, outlines = _prepare_topomap_plot(power.info, layout)
    ch_names = power.info['ch_names']
    ch_idx = [ch_names.index(ch) for ch in layout.names if ch in ch_names]
    data_plot = avg_power[ch_idx]
    pos_plot = pos[ch_idx]
    ch_names_plot = [ch_names[i] for i in ch_idx]

    # ---------- Plot with matplotlib ----------
    fig, ax = plt.subplots()
    im, cn = plot_topomap(data_plot, pos_plot, axes=ax, names=ch_names_plot,
                          show=False, contours=0, cmap='RdBu_r', vlim=(-1, 1))
    ax.set_title(f"{label.capitalize()} ERD/ERS (10–25Hz, 0.5–1.0s)")
    plt.colorbar(im, ax=ax)
    fig.savefig(f"figures/erd_ers_topomap_{label}.png", dpi=300)
    plt.close(fig)

print("Topomap ERD/ERS plots saved in 'figures/' folder.")
