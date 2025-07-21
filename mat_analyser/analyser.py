# import numpy as np
# import matplotlib.pyplot as plt
# from pymatreader import read_mat
# import mne

# # ---------- STEP 1: Load the .mat file ----------
# file_path = r'E:\AMAR\ROBOARM\DATASET\A01T.mat'  # Replace with your actual file
# mat = read_mat(file_path)
# print("Keys in .mat file:", list(mat.keys()))

# # ---------- STEP 2: Find EEG-like array ----------
# def find_2d_array(d, path=""):
#     """Recursively find the first 2D ndarray and track its path."""
#     if isinstance(d, dict):
#         for k, v in d.items():
#             new_path = f"{path}.{k}" if path else k
#             result = find_2d_array(v, new_path)
#             if result is not None:
#                 # This print statement is intentionally inside the recursive call's success check
#                 # to report the actual key that led to the found array.
#                 print(f"Found EEG array under key: {new_path}, shape: {np.shape(result)}")
#                 return result
#     elif isinstance(d, np.ndarray) and d.ndim == 2:
#         return d
#     return None

# # ---------- STEP 3: Find Sampling Frequency ----------
# def find_sfreq(d):
#     """Recursively search for sampling frequency."""
#     if isinstance(d, dict):
#         for k, v in d.items():
#             key_lower = k.lower()
#             if key_lower in ['sfreq', 'fs', 'sampling_rate', 'samplingrate']:
#                 if isinstance(v, (float, int, np.integer, np.floating)):
#                     print(f"Found sfreq under key: {k}: {v}")
#                     return float(v)
#                 elif isinstance(v, np.ndarray) and v.size == 1: # Ensure it's a single value array
#                     print(f"Found sfreq under key: {k}: {v.item()}") # Use .item() for scalar arrays
#                     return float(v.item())
#             elif isinstance(v, dict):
#                 val = find_sfreq(v)
#                 if val is not None:
#                     return val
#     return None

# # ---------- STEP 4: Process EEG ----------
# eeg = find_2d_array(mat)
# if eeg is None:
#     raise ValueError("EEG 2D array not found!")

# # MNE expects (n_channels, n_samples)
# # Check if the number of rows is significantly smaller than columns, suggesting
# # that rows are channels and columns are samples. Or vice-versa.
# # A common pattern for EEG is many samples, few channels.
# # If rows (shape[0]) are much larger than columns (shape[1]), it's likely (n_samples, n_channels)
# if eeg.shape[0] > eeg.shape[1] * 2: # Heuristic: if rows are more than twice columns, assume (samples, channels)
#     eeg = eeg.T  # Transpose to (n_channels, n_samples)
#     print(f"Transposed EEG from {eeg.shape[::-1]} to {eeg.shape}") # Print original and new shape
# elif eeg.shape[1] > eeg.shape[0] * 2: # If columns are much larger, assume it's already (channels, samples)
#     pass # No transpose needed
# else:
#     print(f"Warning: EEG array dimensions ({eeg.shape}) are not clearly (samples, channels) or (channels, samples). Assuming (channels, samples).")


# n_channels, n_samples = eeg.shape
# print(f"Final EEG shape: {eeg.shape}")

# # ---------- STEP 5: Get or default sfreq ----------
# sfreq = find_sfreq(mat)
# if sfreq is None:
#     sfreq = 1000.0  # Default to 1000 Hz if not found
#     print(f"Sampling frequency not found. Using default: {sfreq} Hz")

# # ---------- STEP 6: Create MNE RawArray ----------
# ch_names = [f"EEG {i+1}" for i in range(n_channels)]
# ch_types = ['eeg'] * n_channels
# info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
# raw = mne.io.RawArray(eeg, info)

# # ---------- STEP 7: Plot EEG ----------
# if __name__ == '__main__':
#     import matplotlib
#     matplotlib.use('Qt5Agg') # Ensure Qt5Agg is used

#     print("\nDisplaying EEG plot. Close the plot window to continue/exit.")
#     raw.plot(n_channels=min(10, n_channels), duration=5, scalings='auto', title="EEG Viewer")
    
#     plt.close('all') # Explicitly close all plot windows.





import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pymatreader import read_mat
import mne
import sys


file_path = r'E:\AMAR\ROBOARM\DATASET\GigaDB\s03.mat'#r'E:\AMAR\ROBOARM\DATASET\A01T.mat'#r'C:\Users\Lenovo\Downloads\s01.mat' 
try:
    mat = read_mat(file_path)
    print("Successfully loaded .mat file.")
    print("Top-level keys found:", list(mat.keys()))
except FileNotFoundError:
    print(f"Error: The file was not found at the path: {file_path}")
    print("Please ensure the file path is correct and try again.")
    sys.exit(1) 
except Exception as e:
    print(f"An unexpected error occurred while loading the .mat file: {e}")
    sys.exit(1)


def find_2d_array(d, path=""):
    """
    Recursively find the largest 2D ndarray and return it and its path.
    This version now searches inside lists/tuples.
    """
    if isinstance(d, dict):

        for k, v in d.items():
            new_path = f"{path}.{k}" if path else k
            arr, found_path = find_2d_array(v, new_path)
            if arr is not None:
                return arr, found_path
    elif isinstance(d, (list, tuple)):

        for i, item in enumerate(d):
            new_path = f"{path}[{i}]"
            arr, found_path = find_2d_array(item, new_path)
            if arr is not None:
                return arr, found_path
    elif isinstance(d, np.ndarray) and d.ndim == 2:

        return d, path
        

    return None, None

def find_sfreq(d):
    """
    Recursively search for sampling frequency in dicts, lists, and tuples.
    Looks for common names like 'fs', 'sfreq', etc.
    """
    if isinstance(d, dict):
        for k, v in d.items():

            key_lower = str(k).lower()
            if key_lower in ['sfreq', 'fs', 'samplingrate', 'sampling_rate', 'samplingfrequency']:
                if isinstance(v, (float, int, np.integer, np.floating)):
                    return float(v)

                elif isinstance(v, np.ndarray) and v.size == 1:
                    return float(v.item())

            if isinstance(v, (dict, list, tuple)):
                result = find_sfreq(v)
                if result is not None:
                    return result
    elif isinstance(d, (list, tuple)):

        for item in d:
            result = find_sfreq(item)
            if result is not None:
                return result
    return None


eeg, eeg_path = find_2d_array(mat)

if eeg is None:
    print("\nError: Could not automatically find a suitable 2D EEG data array in the file.")
    print("Please inspect the .mat file structure manually to identify the correct key.")
    sys.exit(1)
else:
    print(f"\nFound potential EEG array under key: '{eeg_path}', with initial shape: {eeg.shape}")


if eeg.shape[0] > eeg.shape[1]:
    eeg = eeg.T
    print(f"Transposed array to {eeg.shape} to conform to MNE's (channels, samples) format.")

n_channels, n_samples = eeg.shape
print(f"Final EEG shape for MNE: {n_channels} channels, {n_samples} samples.")

sfreq = find_sfreq(mat)
if sfreq is None:

    sfreq = 250.0 
    print(f"\nSampling frequency not found automatically. Using a default of {sfreq} Hz.")
else:
    print(f"\nFound sampling frequency: {sfreq} Hz")

ch_names = [f"EEG {i+1}" for i in range(n_channels)]
ch_types = ['eeg'] * n_channels
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)


if np.mean(np.abs(eeg)) < 1e-3:
    eeg = eeg * 1e6
    print("Data values are very small. Assuming units are Volts, converting to microvolts for plotting.")

raw = mne.io.RawArray(eeg, info)
print("\nCreated MNE RawArray object successfully.")


if __name__ == '__main__':

    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        print("\nWarning: Qt5Agg backend not found. Matplotlib will use its default backend.")
        print("If the plot does not appear, you may need to install a GUI backend for matplotlib (e.g., pip install pyqt5).")

    print("\nDisplaying EEG plot. Close the plot window to terminate the script.")
    
    try:

        raw.plot(
            n_channels=min(n_channels, 25), 
            duration=10,                     
            scalings='auto',                
            title=f"EEG Data from: {eeg_path}",
            show=True, 
            block=True
        )
    except Exception as e:
        print(f"\nCould not display the plot. An error occurred: {e}")
        print("Plotting requires a graphical user interface. This may fail if running in a non-GUI environment (like a remote server terminal).")

    print("\nScript finished.")
