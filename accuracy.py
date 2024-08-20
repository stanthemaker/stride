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

    with h5py.File(args.file_path, "r") as f:
        data_recon = f["data"][:] 
    mse = np.mean((data_true - data_recon)**2)
    rms = np.sqrt(np.mean(data_true**2))
    rrmse = np.sqrt(mse) / rms
    print(f"mse:{mse:.4f}, rms:{rms:.4f}, relative rmse:{rrmse:.4f}")

    data = np.stack((data_true, data_recon))

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for i in range(data.shape[0]):
        im = axs[i].imshow(data[i], cmap='viridis')
        axs[i].axis("off")  
        fig.colorbar(im, ax=axs[i], orientation='vertical')

    plt.savefig("tmp.png")

if __name__ == "__main__":
    main()
