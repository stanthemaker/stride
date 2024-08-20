import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

grid_size = int(2100 / 2)
map_path = "/home/stan/data/stride/stride_examples/tutorials/data/maps/realmap.npy"
map = np.load(map_path)
pad_height = (grid_size - map.shape[0]) // 2
pad_width = (grid_size - map.shape[1]) // 2

data_true = np.pad(
    map,
    pad_width=((pad_height, pad_height), (pad_width, pad_width)),
    constant_values=1500,
)

data_filtered = gaussian_filter(data_true, sigma=2)

data = np.stack((data_true, data_filtered))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax1.imshow(data_true, cmap='viridis')
ax1.set_title('Original Image')
ax1.axis('off')
fig.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

im2 = ax2.imshow(data_filtered, cmap='viridis')
ax2.set_title('Gaussian Filtered Image')
ax2.axis('off')
fig.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.046, pad=0.04)

plt.savefig("tmp.png")