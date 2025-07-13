import h5py

file = h5py.File("D:/S01/S01_Se01_CL_R05.mat", "r")
file.visit(print)  # Shows all keys in the file
