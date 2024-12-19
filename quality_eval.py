import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

def compute_imms(img, reference):
    """Compute the IMMS (Image Mean Squared Error)"""
    mse = np.mean((img - reference) ** 2)
    return mse

def compute_psnr(img, reference):
    """Compute PSNR (Peak Signal-to-Noise Ratio)"""
    return psnr_metric(reference, img, data_range=255)

def compute_ssim(img, reference):
    """Compute SSIM (Structural Similarity Index)"""
    return ssim_metric(reference, img, data_range=255)

def main():
    # Load reference and test images
    ref_img_path = "focus_cycle_cropped\cropped_reference.png"
    test_img_path = "focus_cycle_cropped\cropped_render_focus_distance_0.53.png"

    # Load images as grayscale and ensure datatype consistency
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

    if ref_img.shape != test_img.shape:
        print("Shape mismatch, resizing to match.")
        test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))

    mse_val = compute_imms(test_img.astype(np.float32), ref_img.astype(np.float32))
    psnr_val = compute_psnr(test_img.astype(np.uint8), ref_img.astype(np.uint8))
    ssim_val = compute_ssim(test_img.astype(np.uint8), ref_img.astype(np.uint8))

    print("Focal Length: 6mm; Focus Distance: 4m, f-stop:5.8")
    print(f'Image MSE: {mse_val}')
    print(f'PSNR value: {psnr_val} dB')
    print(f'SSIM value: {ssim_val}')

if __name__ == "__main__":
    main()