import os
import re
import cv2
import numpy as np
from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Global variables for cropping
cropping_coordinates = None
reference_image_path = None
compare_images_folder = None
output_folder = None

def select_rectangle(eclick, erelease):
    """Callback function for rectangle selection."""
    global cropping_coordinates
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    cropping_coordinates = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def save_cropped_images(cropping_coordinates):
    """Crop and save images based on selected region."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    x1, y1, x2, y2 = cropping_coordinates
    cropped_reference = reference_image[y1:y2, x1:x2]

    metrics = []
    float_values = []

    for image_name in os.listdir(compare_images_folder):
        compare_image_path = os.path.join(compare_images_folder, image_name)
        compare_image = cv2.imread(compare_image_path)

        if compare_image is None:
            print(f"Skipping {image_name}, unable to read.")
            continue

        cropped_compare = compare_image[y1:y2, x1:x2]
        output_image_path = os.path.join(output_folder, f"cropped_{image_name}")
        cv2.imwrite(output_image_path, cropped_compare)

        # Extract float value from the filename
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", image_name)
        float_value = float(match.group()) if match else None
        if float_value is not None:
            float_values.append(float_value)

        # Compute metrics
        im_mse = mse(cropped_reference, cropped_compare)
        im_psnr = psnr(cropped_reference, cropped_compare, data_range=cropped_reference.max() - cropped_reference.min())
        im_ssim = ssim(cropped_reference, cropped_compare, multichannel=True)

        metrics.append((float_value, im_mse, im_psnr, im_ssim))

    return metrics, float_values


def plot_metrics(metrics, float_values):
    """Plot the computed metrics."""
    sorted_metrics = sorted(zip(float_values, metrics), key=lambda x: x[0])
    sorted_float_values = [x[0] for x in sorted_metrics]
    sorted_metrics = [x[1] for x in sorted_metrics]

    im_mse = [m[1] for m in sorted_metrics]
    im_psnr = [m[2] for m in sorted_metrics]
    im_ssim = [m[3] for m in sorted_metrics]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(sorted_float_values, im_mse, color='blue', marker='o')
    plt.title('Mean Squared Error (MSE)')
    plt.xlabel('Float Value')

    plt.subplot(1, 3, 2)
    plt.plot(sorted_float_values, im_psnr, color='green', marker='o')
    plt.title('Peak Signal-to-Noise Ratio (PSNR)')
    plt.xlabel('Float Value')

    plt.subplot(1, 3, 3)
    plt.plot(sorted_float_values, im_ssim, color='red', marker='o')
    plt.title('Structural Similarity Index (SSIM)')
    plt.xlabel('Float Value')

    plt.tight_layout()
    plt.show()


def crop_and_compare(reference_image_path, compare_images_folder, output_folder):
    global reference_image

    # Load the reference image
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        raise FileNotFoundError(f"Unable to read the reference image: {reference_image_path}")

    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    # Display the image for cropping
    fig, ax = plt.subplots()
    ax.imshow(reference_image)
    rect_selector = RectangleSelector(ax, select_rectangle, drawtype='box', useblit=True, button=[1], interactive=True)
    plt.title("Select the region to crop and close the window.")
    plt.show()

    if cropping_coordinates is None:
        raise RuntimeError("No cropping area selected.")

    metrics, float_values = save_cropped_images(cropping_coordinates)
    plot_metrics(metrics, float_values)

if __name__ == "__main__":
    reference_image_path = input("Enter the path to the reference image: ")
    compare_images_folder = input("Enter the folder containing images to compare: ")
    output_folder = input("Enter the folder to save cropped images: ")

    crop_and_compare(reference_image_path, compare_images_folder, output_folder)