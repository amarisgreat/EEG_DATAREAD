import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mne.channels import make_dig_montage
from MI_based_visualizer import  visualize_motor_imagery_from_mat 


def draw_sensors_and_wave(filepath):
    """
    Loads EEG data from a .mat file, then visualizes the sensor layout
    and the averaged brainwave for key motor channels.

    Args:
        filepath (str): The path to the subject's .mat file.
    """
    # --- 1. Load the .mat file ---
    print(f"Loading data from {filepath}...")
    mat_data = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
    data = mat_data[data_key]
    print(f"Accessing data structure: '{data_key}'")

    # Transpose EEG data and get info
    eeg_data_left = data.imagery_left.T
    eeg_data_right = data.imagery_right.T
    num_channels = data.imagery_left.shape[0]
    sampling_rate = data.srate

    # Combine data
    gap = np.zeros((int(sampling_rate * 2), num_channels))
    eeg_data_combined = np.vstack([eeg_data_left, gap, eeg_data_right])

    # --- 2. Create MNE Raw Object ---
    montage = None
    try:
        # Attempt to load channel names and positions
        ch_names = list(data.senloc.labels)
        ch_positions_raw = data.senloc.electrodeposition
        coords_xyz = {ch: pos for ch, pos in zip(ch_names, ch_positions_raw)}
        montage = make_dig_montage(ch_pos=coords_xyz, coord_frame='head')
        print("Successfully loaded channel names and positions.")
    except Exception as e:
        # If it fails, create generic channel names and proceed without a montage
        print(f"Could not load channel names/locations due to error: {e}. Using generic names.")
        ch_names = [f'EEG {i+1}' for i in range(num_channels)]

    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(eeg_data_combined.T, info, verbose=False)

    if montage:
        raw.set_montage(montage)
    else:
        # Fallback to a standard montage if custom one fails, for plotting purposes
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)


    # --- 3. Create Events and Epochs ---
    n_trials = data.n_imagery_trials
    trial_duration_samples = eeg_data_left.shape[0] // n_trials
    events = []
    for i in range(n_trials):
        events.append([i * trial_duration_samples, 0, 2]) # Left
    start_sample_right = eeg_data_left.shape[0] + gap.shape[0]
    for i in range(n_trials):
        events.append([start_sample_right + (i * trial_duration_samples), 0, 3]) # Right

    events = np.array(events)
    event_id = dict(left_hand=2, right_hand=3)

    # Filter and create epochs
    raw.filter(l_freq=1., h_freq=40., fir_design='firwin', verbose=False)
    epochs = mne.Epochs(raw, events, event_id, tmin=-1., tmax=4.,
                        baseline=(-1, 0), preload=True, verbose=False)

    # --- 4. Generate Plots ---
    print("Generating plots...")

    # --- FIX for RuntimeError ---
    # Since the file's digitization points couldn't be loaded, we use a standard
    # montage for visualization. This provides a representative sensor layout.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Sensor Layout
    picks_vis = ['C3', 'C4']
    raw.plot_sensors(ch_type='eeg', axes=axes[0], show_names=True, picks=picks_vis, show=False)
    axes[0].set_title('Standard EEG Sensor Layout')
    
    # Plot 2: The Brainwave
    # For the wave plot, we must use the actual channel names present in the info object.
    # From the paper, C3=13 and C4=51. Our generic names are 1-based.
    # The actual names in the montage are 'C3' and 'C4'.
    evoked_right = epochs['right_hand'].average()
    evoked_right.plot(picks=picks_vis, axes=axes[1], show=False, spatial_colors=True)
    axes[1].set_title('Averaged Brainwave for Right-Hand Imagery')
    axes[1].axvline(0, linestyle='--', color='red', label='Cue')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# --- RUN THE ANALYSIS ---
try:
    draw_sensors_and_wave(r'E:\AMAR\ROBOARM\DATASET\GigaDB\s03.mat')
    visualize_motor_imagery_from_mat(r'E:\AMAR\ROBOARM\DATASET\GigaDB\s03.mat')

except FileNotFoundError:
    print("\nERROR: The specified .mat file was not found.")
    print("Please ensure the file path is correct.")