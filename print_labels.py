import os
import h5py
import numpy as np

# Define the folder containing .mat files
folder = "D:/S10"  # <-- Change this to your actual path

# Define the label mapping based on position
def pos_to_label(x, y, tol=0.1):
    if abs(x + 0.5) < tol and abs(y) < tol: return 'left'
    if abs(x - 0.5) < tol and abs(y) < tol: return 'right'
    if abs(x) < tol and abs(y - 0.5) < tol: return 'both'
    if abs(x) < tol and abs(y + 0.5) < tol: return 'feet'
    return 'rest'


# Traverse all files in the folder
for file in sorted(os.listdir(folder)):
    if not file.endswith(".mat"):
        continue

    path = os.path.join(folder, file)
    try:
        mat_file = h5py.File(path, "r")

        # Extract target position references
        x_refs = mat_file['eeg']['targetpos']['x'][0]
        y_refs = mat_file['eeg']['targetpos']['y'][0]

        unique_positions = set()

        # Iterate over all target positions
        for x_ref, y_ref in zip(x_refs, y_refs):
            try:
                x_val = int(np.squeeze(mat_file[x_ref][()]))
                y_val = int(np.squeeze(mat_file[y_ref][()]))
                unique_positions.add((x_val, y_val))
            except Exception as e:
                continue  # Skip bad entries

        # Map positions to labels
        labels = sorted({pos_to_label(x, y) for (x, y) in unique_positions})
        positions_str = ", ".join([f"({x}, {y})" for (x, y) in sorted(unique_positions)])

        print(f"{file}:")
        print(f"  Unique positions: {positions_str}")
        print(f"  MI labels: {labels}\n")

    except Exception as e:
        print(f"{file}: Error - {e}")


# import h5py
# import numpy as np

# path = "D:/S01/S01_Se01_CL_R05.mat"  # Use the path to your MI file
# f = h5py.File(path, "r")

# # Inspect keys
# print("== Keys in file ==")
# f.visit(lambda x: print(x))

# # Read actual float data directly
# target_x_vals = f['eeg']['targetpos']['x'][()].flatten()
# target_y_vals = f['eeg']['targetpos']['y'][()].flatten()

# def pos_to_label(x, y):
#     if x == -1 and y == 0: return 'left'
#     if x == 1 and y == 0: return 'right'
#     if x == 0 and y == 1: return 'both'
#     if x == 0 and y == -1: return 'feet'
#     return 'rest'

# print("\n== First 10 MI labels from target positions ==")
# for i in range(min(10, len(target_x_vals))):
#     x = int(round(target_x_vals[i]))
#     y = int(round(target_y_vals[i]))
#     label = pos_to_label(x, y)
#     print(f"x[{i}] = {x}, y[{i}] = {y} => {label}")
