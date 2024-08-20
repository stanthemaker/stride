import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import argparse

def plot(path, save_path):
    with h5py.File(path, "r") as f:
        data = f["data"][:] 
        plt.figure(figsize=(10, 8))
        plt.imshow(data, aspect="auto", interpolation="none")
        plt.colorbar()  # Adds a color bar to indicate the scale
        plt.title("Speed of sound map")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
        plt.savefig(save_path)

def parse_args():
    parser = argparse.ArgumentParser(description="read folder")
    parser.add_argument("folder", type=str, help="Path to the folder.")
    return parser.parse_args()

args = parse_args()
files = sorted(os.listdir(args.folder))
    
# Loop through each file
for filename in files:
    if filename.endswith(".h5") and "Vp" in filename:
        savename = filename.split("-")[-1].split(".")[0]
        file_idx = int(savename)
        file_path = os.path.join(args.folder, filename)
        if file_idx % 10 != 0:
            os.remove(file_path) 
        else: 
            save_path = os.path.join(args.folder, savename)
            plot(file_path,save_path)