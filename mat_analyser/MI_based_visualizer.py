import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mne.channels import make_dig_montage

def visualize_motor_imagery_from_mat(filepath):
    """
    Loads motor imagery EEG data from a .mat file, processes it, and
    visualizes the brain signals (ERD/ERS) corresponding to motor commands.

    Args:
        filepath (str): The path to the subject's .mat file.
    """
    # --- 1. Load the .mat file ---
    print(f"Loading data from {filepath}...")
    mat_data = loadmat(filepath, squeeze_me=True, struct_as_record=False)

    data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
    data = mat_data[data_key]
    print(f"Accessing data structure: '{data_key}'")

    # Transpose EEG data to (samples, channels)
    eeg_data_left = data.imagery_left.T
    eeg_data_right = data.imagery_right.T

    # Derive channel count directly from data shape
    num_channels = data.imagery_left.shape[0]
    print(f"Found {num_channels} channels and {eeg_data_left.shape[0]} samples per trial block.")

    # Create the gap array for combining data
    sampling_rate = data.srate
    gap_duration_samples = int(sampling_rate * 2)
    gap = np.zeros((gap_duration_samples, num_channels))

    eeg_data_combined = np.vstack([eeg_data_left, gap, eeg_data_right])

    # --- 2. Create MNE Info Object ---
    try:
        ch_names = list(data.senloc.labels)
        ch_positions_raw = data.senloc.electrodeposition
        coords_xyz = {ch: pos for ch, pos in zip(ch_names, ch_positions_raw)}
        montage = make_dig_montage(ch_pos=coords_xyz, coord_frame='head')
        print("Successfully loaded channel names and positions.")
    except Exception:
        print("Could not load channel names/locations. Using generic names.")
        ch_names = [f'EEG {i+1}' for i in range(num_channels)]
        montage = None

    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data_combined.T, info)

    if montage:
        raw.set_montage(montage)

    # --- 3. Create Events and Epochs ---
    n_trials_left = data.n_imagery_trials
    trial_duration_samples = eeg_data_left.shape[0] // n_trials_left

    events = []
    for i in range(n_trials_left):
        events.append([i * trial_duration_samples, 0, 2])
    start_sample_right = eeg_data_left.shape[0] + gap_duration_samples
    for i in range(n_trials_left):
        events.append([start_sample_right + (i * trial_duration_samples), 0, 3])

    events = np.array(events)
    event_id = dict(left_hand=2, right_hand=3)

    # --- 4. Preprocess and Analyze ---
    print("Filtering data and creating epochs...")
    raw.filter(l_freq=8., h_freq=30., fir_design='firwin', verbose=False)
    epochs = mne.Epochs(raw, events, event_id, tmin=-1., tmax=4.,
                        baseline=(-1, 0), preload=True, verbose=False)

    print("Calculating power for each frequency and time point...")
    freqs = np.arange(8., 31., 1.)
    n_cycles = freqs / 2.
    
    epochs_tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                             use_fft=True, return_itc=False,
                                             average=False, decim=2, verbose=False)

    # --- 5. Visualize the Results ---
    print("Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('Motor Command Signatures from .mat File', fontsize=16)

    # --- FIX for ValueError ---
    # Use the generic channel names that correspond to the C3/C4 locations.
    # From the paper's Figure 1: C4 is channel 51, C3 is channel 13.
    pick_c4 = 'EEG 51'
    pick_c3 = 'EEG 13'

    epochs_tfr['left_hand'].average().plot(picks=[pick_c4], axes=axes[0], colorbar=True,
                                           show=False)
    axes[0].set_title(f'Left-Hand Imagery (Channel {pick_c4})')
    axes[0].axvline(0, linestyle='--', color='red', label='Cue')
    axes[0].legend()

    epochs_tfr['right_hand'].average().plot(picks=[pick_c3], axes=axes[1], colorbar=True,
                                            show=False)
    axes[1].set_title(f'Right-Hand Imagery (Channel {pick_c3})')
    axes[1].axvline(0, linestyle='--', color='red')
    # --- End of Fix ---

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()

# --- RUN THE ANALYSIS ---
try:
    visualize_motor_imagery_from_mat(r'E:\AMAR\ROBOARM\DATASET\GigaDB\s03.mat')
except FileNotFoundError:
    print("\nERROR: The specified .mat file was not found.")
    print("Please ensure the file path is correct.")