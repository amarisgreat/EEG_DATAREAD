import mne
import numpy as np

def read_event_file_manually(filepath):
    """Manually read .edf.event file and parse into (sample, duration, type) format."""
    events = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                parts = [int(float(p)) for p in line.strip().split()]
                if len(parts) >= 3:
                    events.append(parts[:3])
            except (ValueError, IndexError):
                print(f"Skipping malformed line: {line.strip()}")
    return np.array(events, dtype=int)

# Paths
edf_file_path = r'.\steps\S001R04.edf'
event_file_path = r'.\steps\S001R04.edf.event'

# Load EDF file
raw = mne.io.read_raw_edf(edf_file_path, preload=False, verbose=False)
sfreq = raw.info['sfreq']

# Extract events from EDF annotations
events_edf, event_id_map = mne.events_from_annotations(raw, verbose=False)

print("\n🔹 Events from EDF Annotations:")
print(f"{'Index':<6} {'Time (s)':<12} {'Sample':<10} {'Type':<10}")
print("-" * 40)
for i, event in enumerate(events_edf):
    sample, _, event_code = event
    time_sec = sample / sfreq
    event_name = next((k for k, v in event_id_map.items() if v == event_code), f"Code {event_code}")
    print(f"{i:<6} {time_sec:<12.3f} {sample:<10} {event_name}")

# Load events from .edf.event manually
events_manual = read_event_file_manually(event_file_path)

print("\n🔹 Events from .edf.event File:")
print(f"{'Index':<6} {'Time (s)':<12} {'Sample':<10} {'Type':<10}")
print("-" * 40)
for i, event in enumerate(events_manual):
    sample, _, event_type = event
    time_sec = sample / sfreq
    print(f"{i:<6} {time_sec:<12.3f} {sample:<10} {event_type}")
