import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="read .h5 file")
    parser.add_argument("filepath", type=str, help="Path to the mat file.")
    return parser.parse_args()


# Load the MATLAB file
args = parse_args()
data = loadmat(args.filepath)

# Extract vp_true and vp_estimated
vp_true = data["vp_true"]
vp_estimated = data["vp_estimated"]

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot vp_true
axes[0].imshow(vp_true, cmap="viridis", vmin = 1400, vmax = 1600)
axes[0].set_title("VP Ground Truth")
axes[0].set_xlabel("X Dimension")
axes[0].set_ylabel("Y Dimension")

# Plot vp_estimated
axes[1].imshow(vp_estimated, cmap="viridis", vmin = 1400, vmax = 1600)
axes[1].set_title("VP Estimated")
axes[1].set_xlabel("X Dimension")
axes[1].set_ylabel("Y Dimension")

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig("tmp.png")
# plt.show()
