import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from tqdm import tqdm

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if image is not None:
                images.append((filename, image))
    return images

# Display images side by side with zoom functionality
def display_image_pairs(folder1, folder2):
    # Load images from folders
    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)

    if len(images1) != len(images2):
        print("Error: Number of images in both folders must be the same.")
        return

    # Function to display a zoomed-in region
    def zoom_in(region):
        x1, y1, x2, y2 = region
        img1_zoom = img1[int(y1):int(y2), int(x1):int(x2)]
        img2_zoom = img2[int(y1):int(y2), int(x1):int(x2)]

        if img1_zoom.size > 0 and img2_zoom.size > 0:
            plt.figure("Zoomed View")
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img1_zoom, cv2.COLOR_BGR2RGB))
            plt.title(filename1)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img2_zoom, cv2.COLOR_BGR2RGB))
            plt.title(filename2)
            plt.axis("off")
            plt.show()

            cv2.imwrite("cropped_no_dome_port.png", img1_zoom)
            cv2.imwrite("cropped_w_dome_port.png", img2_zoom)

    # Callback for rectangle selection
    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        zoom_in((x1, y1, x2, y2))

    # Display each pair of images
    for (filename1, img1), (filename2, img2) in zip(images1, images2):
        print(f"Displaying: {filename1} and {filename2}")

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"No Dome Port\n{filename1}")
        ax[0].axis("off")

        ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f"Dome Port No Offset\n{filename2}")
        ax[1].axis("off")

        # Rectangle selector to zoom in on Image 1
        rect_selector = RectangleSelector(
            ax[0],
            onselect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True
        )

        plt.show()

# Paths to the two folders
folder1 = "dome_port_test\\no_dome_port"
folder2 = "dome_port_test\\dome_port_centered"

# Run the program
display_image_pairs(folder1, folder2)