import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Plot heatmap from HDF5 file.")
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file.")
    return parser.parse_args()

def main():
    args = parse_args()
    with h5py.File(args.file_path, "r") as f:
        data = f["data"][:] 
        plt.figure(figsize=(10, 8))
        plt.imshow(data, aspect="auto", interpolation="none")
        plt.colorbar()  # Adds a color bar to indicate the scale
        plt.title("Speed of sound map")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
        plt.savefig('tmp.png')

if __name__ == "__main__":
    main()
