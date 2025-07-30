import mne
import numpy as np

# === Load EDF ===
edf_path = r"E:\AMAR\ROBOARM\DATASET\files\S001\S001R04.edf"
run_number = int(edf_path.split('R')[-1].split('.')[0])

# === Load EEG data ===
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
annotations = raw.annotations

# === Define class label and color maps ===
CLASS_LABELS = {
    'T0': (0, 'Rest'),
    'T1': {
        3: (1, 'Real Left Fist'),
        4: (3, 'Imagined Left Fist'),
        5: (5, 'Real Both Fists'),
        6: (7, 'Imagined Both Fists'),
        7: (1, 'Real Left Fist'),
        8: (3, 'Imagined Left Fist'),
        9: (5, 'Real Both Fists'),
        10: (7, 'Imagined Both Fists'),
        11: (1, 'Real Left Fist'),
        12: (3, 'Imagined Left Fist'),
        13: (5, 'Real Both Fists'),
        14: (7, 'Imagined Both Fists')
    },
    'T2': {
        3: (2, 'Real Right Fist'),
        4: (4, 'Imagined Right Fist'),
        5: (6, 'Real Both Feet'),
        6: (8, 'Imagined Both Feet'),
        7: (2, 'Real Right Fist'),
        8: (4, 'Imagined Right Fist'),
        9: (6, 'Real Both Feet'),
        10: (8, 'Imagined Both Feet'),
        11: (2, 'Real Right Fist'),
        12: (4, 'Imagined Right Fist'),
        13: (6, 'Real Both Feet'),
        14: (8, 'Imagined Both Feet')
    }
}

CLASS_COLORS = {
    -1: 'gray',    # Rest
    1: 'blue',    # Real Left Fist
    2: 'red',     # Real Right Fist
    3: 'cyan',    # Imagined Left Fist
    4: 'orange',  # Imagined Right Fist
    5: 'purple',  # Real Both Fists
    6: 'green',   # Real Both Feet
    7: 'magenta', # Imagined Both Fists
    8: 'brown'    # Imagined Both Feet
}

# === Extract task events based on actual annotation onset/duration ===
events = []
event_id = {}

print(f"\n[Extracted Events from {edf_path.split('/')[-1]} — RUN {run_number}]\n")
sfreq = raw.info['sfreq']

for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
    if desc in CLASS_LABELS:
        if desc == 'T0':
            class_id, class_name = CLASS_LABELS['T0']
        else:
            if run_number not in CLASS_LABELS[desc]:
                continue
            class_id, class_name = CLASS_LABELS[desc][run_number]

        # Store onset sample for plotting
        sample = int(onset * sfreq)
        events.append([sample, 0, class_id])
        event_id[class_name] = class_id

        print(f"Label: {class_name:22s} | ID: {class_id} | Onset: {onset:7.2f}s | Duration: {duration:5.2f}s")

# Convert to NumPy array
events = np.array(events)

# === Plot ===
raw.plot(
    events=events,
    event_id=event_id,
    event_color=CLASS_COLORS,
    title=f"EEG Task Visualization — Run {run_number}",
    scalings='auto',
    n_channels=32,
    duration=10.0,
    show=True,
    block=True
)

# === Class Legend ===
print("\n[CLASS MAPPING USED]")
for class_id, color in CLASS_COLORS.items():
    names = [name for name, cid in event_id.items() if cid == class_id]
    for name in names:
        print(f"ID {class_id}: {name:22s} | Color: {color}")
