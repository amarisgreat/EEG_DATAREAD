# import h5py

# file = h5py.File(r"E:\AMAR\ROBOARM\DATASET\A01T.mat", "r")
# file.visit(print)  # Shows all keys in the file



# Print top-level keys in the .mat file


# from scipy.io import loadmat

# filename = r"C:\Users\Lenovo\Downloads\s01.mat"  # or update path accordingly
# mat = loadmat(filename, simplify_cells=True)

# print("\nTop-level keys in .mat file:")
# print(mat.keys())

# if 'eeg' in mat:
#     print("\n'eeg' subkeys:")
#     print(mat['eeg'].keys())
# else:
#     print("\n'eeg' key not found.")


import scipy.io

mat = scipy.io.loadmat(r"E:\AMAR\ROBOARM\DATASET\A01T.mat")

print("Available keys:")
for key in mat:
    if not key.startswith('__'):
        print(f"- {key}")

# Inspect the 'data' key
eeg = mat['data']

print("\nShape of 'data':", eeg.shape)
print("Data type:", type(eeg))

print("Available keys:")
for key in mat:
    if not key.startswith('__'):
        print(f"- {key}")
eeg_struct = mat['data'][0, 0]

print("\nFields in 'data':")
print(eeg_struct.dtype)

