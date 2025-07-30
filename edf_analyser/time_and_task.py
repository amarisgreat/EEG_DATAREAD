# import mne


# edf_path = 'E:\AMAR\git\EEG_DATAREAD\edf_analyser\steps\S001R04.edf'  
# raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

# annotations = raw.annotations


# print("Event Time (s)\tLabel")
# for onset, desc in zip(annotations.onset, annotations.description):
#     print(f"{onset:.2f}\t\t{desc}")

# import mne
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Patch


# edf_path = 'E:\AMAR\git\EEG_DATAREAD\edf_analyser\steps\S001R04.edf'  
# run_number = int(edf_path.split('R')[-1].split('.')[0])  

# real_runs = [3, 5, 7, 9, 11, 13]
# imagined_runs = [4, 6, 8, 10, 12, 14]

# if run_number in real_runs:
#     task_type = 'real'
# elif run_number in imagined_runs:
#     task_type = 'imagined'
# else:
#     task_type = 'other'


# raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
# data, times = raw.get_data(return_times=True)
# channel_names = raw.ch_names


# annotations = raw.annotations
# onsets = annotations.onset
# durations = annotations.duration
# descriptions = annotations.description


# if task_type == 'real':
#     task_colors = {'T0': 'lightgray', 'T1': 'lightgreen', 'T2': 'lightgreen'}
#     legend_desc = {
#         'T0': 'Rest',
#         'T1': 'Real Movement (Left/Both Fists)',
#         'T2': 'Real Movement (Right/Both Feet)'
#     }
# elif task_type == 'imagined':
#     task_colors = {'T0': 'lightgray', 'T1': 'orange', 'T2': 'green'}
#     legend_desc = {
#         'T0': 'Rest',
#         'T1': 'Imagined Movement (Left/Both Fists)',
#         'T2': 'Imagined Movement (Right/Both Feet)'
#     }
# else:
#     task_colors = {'T0': 'gray', 'T1': 'blue', 'T2': 'red'}
#     legend_desc = {
#         'T0': 'Rest',
#         'T1': 'T1',
#         'T2': 'T2'
#     }

# plt.figure(figsize=(20, 18))
# offset = 150  

# for i, ch_data in enumerate(data):
#     plt.plot(times, ch_data * 1e6 + i * offset, label=channel_names[i], linewidth=0.8)


# for onset, duration, desc in zip(onsets, durations, descriptions):
#     if desc in task_colors:
#         plt.axvspan(onset, onset + duration, color=task_colors[desc], alpha=0.3)


# plt.yticks(np.arange(0, len(channel_names) * offset, offset), channel_names)
# plt.xlabel("Time (s)")
# plt.ylabel("EEG Channels (µV + offset)")
# plt.title(f"EEG (All Channels) — Task: {task_type.upper()} — File: {edf_path.split('/')[-1]}")


# legend_elements = [Patch(facecolor=color, edgecolor='k', label=label)
#                    for key, color in task_colors.items()
#                    for desc, label in legend_desc.items() if key == desc[0]]
# plt.legend(handles=legend_elements, loc='upper right')

# plt.grid(True, axis='x', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()


# import mne
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# import ipywidgets as widgets
# from IPython.display import display

# # Load a single EDF file
# edf_path = r"E:\AMAR\git\EEG_DATAREAD\edf_analyser\steps\S001R04.edf"  # Update path if needed
# run_number = int(edf_path.split('R')[-1].split('.')[0])

# # Determine task type
# real_runs = [3, 5, 7, 9, 11, 13]
# imagined_runs = [4, 6, 8, 10, 12, 14]
# task_type = 'real' if run_number in real_runs else 'imagined' if run_number in imagined_runs else 'other'

# # Load EEG data
# raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
# data, times = raw.get_data(return_times=True)
# channel_names = raw.ch_names
# annotations = raw.annotations

# # Annotation color mapping
# color_map = {
#     'T0': 'lightgray',
#     'T1': 'lightgreen' if task_type == 'real' else 'orange',
#     'T2': 'lightgreen' if task_type == 'real' else 'orange'
# }
# legend_desc = {
#     'T0': 'Rest',
#     'T1': f'{task_type.capitalize()} Movement: Left/Both Fists',
#     'T2': f'{task_type.capitalize()} Movement: Right/Both Feet'
# }

# # Widgets
# channel_slider = widgets.IntSlider(
#     min=0, max=len(channel_names)-1, step=1, value=0, description='Channel')
# time_slider = widgets.FloatSlider(
#     min=0, max=times[-1]-10, step=1, value=0, description='Time (s)', continuous_update=False)

# # Interactive plotting function
# def plot_channel(channel_idx, start_time):
#     plt.clf()
#     fig, ax = plt.subplots(figsize=(12, 4))
    
#     end_time = start_time + 10  # 10-second window
#     sfreq = raw.info['sfreq']
#     start_sample = int(start_time * sfreq)
#     end_sample = int(end_time * sfreq)
    
#     signal = data[channel_idx, start_sample:end_sample] * 1e6  # Convert to µV
#     time_range = times[start_sample:end_sample]
    
#     ax.plot(time_range, signal, label=channel_names[channel_idx], linewidth=1.2)
    
#     for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
#         if desc in color_map and onset < end_time and (onset + duration) > start_time:
#             x_start = max(onset, start_time)
#             x_end = min(onset + duration, end_time)
#             ax.axvspan(x_start, x_end, color=color_map[desc], alpha=0.3)

#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Amplitude (µV)")
#     ax.set_title(f"EEG Channel: {channel_names[channel_idx]} ({task_type.upper()} Task)")
#     ax.grid(True)

#     handles = [Patch(facecolor=color_map[k], label=legend_desc[k]) for k in legend_desc]
#     ax.legend(handles=handles, loc='upper right')
    
#     plt.tight_layout()
#     plt.show()

# # Activate interactivity
# widgets.interact(plot_channel, channel_idx=channel_slider, start_time=time_slider)


import mne
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.patches import Patch

edf_path = r'E:\AMAR\git\EEG_DATAREAD\edf_analyser\steps\S001R04.edf'
run_number = int(edf_path.split('R')[-1].split('.')[0])

real_runs = [3, 5, 7, 9, 11, 13]
imagined_runs = [4, 6, 8, 10, 12, 14]

if run_number in real_runs:
    task_type = 'real'
elif run_number in imagined_runs:
    task_type = 'imagined'
else:
    task_type = 'other'

raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
data, times = raw.get_data(return_times=True)
channel_names = raw.ch_names
n_channels = data.shape[0]

annotations = raw.annotations
onsets = annotations.onset
durations = annotations.duration
descriptions = annotations.description

# Define colors and legends
if task_type == 'real':
    task_colors = {'T0': 'lightgray', 'T1': 'lightgreen', 'T2': 'lightgreen'}
    legend_desc = {
        'T0': 'Rest',
        'T1': 'Real Movement (Left/Both Fists)',
        'T2': 'Real Movement (Right/Both Feet)'
    }
elif task_type == 'imagined':
    task_colors = {'T0': 'lightgray', 'T1': 'orange', 'T2': 'green'}
    legend_desc = {
        'T0': 'Rest',
        'T1': 'Imagined Movement (Left/Both Fists)',
        'T2': 'Imagined Movement (Right/Both Feet)'
    }
else:
    task_colors = {'T0': 'gray', 'T1': 'blue', 'T2': 'red'}
    legend_desc = {
        'T0': 'Rest',
        'T1': 'T1',
        'T2': 'T2'
    }

# Visualization parameters
channels_per_page = 8
offset = 150  # vertical offset for each channel

# Create figure and axis
fig, ax = plt.subplots(figsize=(20, 10))
plt.subplots_adjust(bottom=0.2)

lines = []
yticks = []
yticklabels = []

# Function to update visible channel set
def plot_channels(start_idx):
    ax.clear()
    lines.clear()
    yticks.clear()
    yticklabels.clear()

    end_idx = min(start_idx + channels_per_page, n_channels)

    for i, ch_idx in enumerate(range(start_idx, end_idx)):
        ch_data = data[ch_idx] * 1e6  # Convert to µV
        y_offset = i * offset
        ax.plot(times, ch_data + y_offset, label=channel_names[ch_idx], linewidth=0.8)
        yticks.append(y_offset)
        yticklabels.append(channel_names[ch_idx])

    for onset, duration, desc in zip(onsets, durations, descriptions):
        if desc in task_colors:
            ax.axvspan(onset, onset + duration, color=task_colors[desc], alpha=0.3)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EEG Channels (µV + offset)")
    ax.set_title(f"EEG Visualization — Task: {task_type.upper()} — File: {edf_path.split('/')[-1]}")
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5)

    # Legend
    legend_elements = [Patch(facecolor=color, edgecolor='k', label=label)
                       for key, color in task_colors.items()
                       for desc, label in legend_desc.items() if key == desc[0]]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.canvas.draw_idle()

# Initial plot
plot_channels(0)

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Channel Group', 0, n_channels - channels_per_page, valinit=0, valstep=channels_per_page)

def update(val):
    start_idx = int(slider.val)
    plot_channels(start_idx)

slider.on_changed(update)

plt.show()
