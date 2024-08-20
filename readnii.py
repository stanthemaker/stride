import nibabel as nib
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, gaussian_filter, zoom
from nibabel.affines import apply_affine
import os
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Mapping ranges to speed of sound
mapping = {
    (-1024, -501): 331,
    (-500, -101): 400,
    (-100, -50): 1450,
    (-49, 29): 1500,
    (30, 60): 1560,
    (100, 300): 1540,
    (700, 3000): 3000,
}


def largest_connected_component(binary_array):
    labeled_array, num_features = label(binary_array)
    LCC_size = 0
    LCC_label = 0

    for i in range(1, num_features + 1):  # Start from 1 as 0 is the background
        component_size = np.sum(labeled_array == i)
        if component_size > LCC_size:
            LCC_size = component_size
            LCC_label = i

    LCC = (labeled_array == LCC_label).astype(int)
    return LCC


def ct2sos(hu_array):
    speed_array = np.zeros_like(hu_array, dtype=int)

    for (lower_bound, upper_bound), speed in mapping.items():
        mask = (hu_array >= lower_bound) & (hu_array <= upper_bound)
        speed_array[mask] = speed

    return speed_array


def parse_args():
    parser = argparse.ArgumentParser(description="generate 2d ndarray from nii")
    parser.add_argument("file_path", type=str, help="Path to the nii file.")
    parser.add_argument(
        "save_folder", type=str, help="Folder to save the output 2d array"
    )
    parser.add_argument("filename", type=str, help="Saved file name")
    return parser.parse_args()


def apply_lowpass_fft(data, cutoff):
    # Transform to frequency domain
    data_fft = fft2(data)
    data_fft_shifted = fftshift(data_fft)

    # Create a lowpass filter mask
    H, W = data.shape
    X, Y = np.ogrid[:H, :W]
    center_x, center_y = H // 2, W // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= cutoff**2

    filtered_fft_shifted = data_fft_shifted * mask

    filtered_fft = ifftshift(filtered_fft_shifted)
    filtered_data = ifft2(filtered_fft)
    return np.abs(filtered_data)


def main():
    args = parse_args()
    vol = nib.load(args.file_path)
    np.set_printoptions(precision=4, suppress=True)
    vol_data = vol.get_fdata()

    p1 = np.array([255, 255, 31])
    p2 = np.array([256, 255, 31])

    p1_real = apply_affine(vol.affine, p1)
    p2_real = apply_affine(vol.affine, p2)

    slice = vol_data[:, :, 30]  # 30 kidney
    binary_array = (slice > -500).astype(int)
    LCC = largest_connected_component(binary_array)
    background_mask = largest_connected_component(1 - LCC).astype(bool)
    slice[background_mask] = 0
    slice_sos = ct2sos(slice)
    slice_sos[slice_sos < 1450] = 1450
    slice_sos[slice_sos > 1550] = 1550

    zoom_factor = 600 / 512
    slice_sos = zoom(slice_sos, zoom_factor, order=3)
    slice_sos = apply_lowpass_fft(slice_sos, 250)

    savefile_path = os.path.join(args.save_folder, args.filename)
    np.save(savefile_path, slice_sos)
    print(f"sos map saved at : {savefile_path}")

    plt.imshow(slice_sos, cmap="viridis", vmin=1450, vmax=1600)
    plt.colorbar()
    plt.axis("off")
    plt.title("Vp ground truth")
    plt.savefig("tmp.png")


if __name__ == "__main__":
    main()
