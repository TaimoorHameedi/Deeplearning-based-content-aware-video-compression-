import cv2
import numpy as np
import os

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    orig_path = "VIDEO_FOR_TESTING.mp4"
    roi_path = "temp_cavc_hls/roi_stream.mp4"
    
    if not os.path.exists(orig_path) or not os.path.exists(roi_path):
        print(f"Error: Files not found. Ensure {orig_path} and {roi_path} exist.")
        return

    cap_o = cv2.VideoCapture(orig_path)
    cap_r = cv2.VideoCapture(roi_path)
    
    ret_o, f_o = cap_o.read()
    ret_r, f_r = cap_r.read()
    
    if not ret_o or not ret_r:
        print("Error: Could not read frames.")
        return

    # Resize to match (1920, 1080) as in pipeline
    PROCESSING_RESOLUTION = (1920, 1080)
    f_o = cv2.resize(f_o, PROCESSING_RESOLUTION)
    f_r = cv2.resize(f_r, PROCESSING_RESOLUTION)

    # Generate ROI mask from ROI stream
    gray_r = cv2.cvtColor(f_r, cv2.COLOR_BGR2GRAY)
    mask = (gray_r > 5).astype(np.uint8)
    
    # Baseline PSNR
    mse_base = np.mean(((f_o.astype(float) - f_r.astype(float))**2)[mask > 0])
    psnr_base = 10 * np.log10(255**2 / mse_base) if mse_base > 0 else 100
    print(f"Baseline ROI PSNR: {psnr_base:.2f} dB")

    # 1. Color Swap Test (BGR -> RGB)
    f_r_swapped = cv2.cvtColor(f_r, cv2.COLOR_BGR2RGB)
    mse_swap = np.mean(((f_o.astype(float) - f_r_swapped.astype(float))**2)[mask > 0])
    psnr_swap = 10 * np.log10(255**2 / mse_swap) if mse_swap > 0 else 100
    print(f"Color Swapped ROI PSNR: {psnr_swap:.2f} dB")

    # 2. Range Shift Test (TV -> Full)
    # Estimate if comp is in 16-235 range and scaled back
    f_r_scaled = np.clip((f_r.astype(float) - 16) * (255.0 / (235.0 - 16.0)), 0, 255).astype(np.uint8)
    mse_scale = np.mean(((f_o.astype(float) - f_r_scaled.astype(float))**2)[mask > 0])
    psnr_scale = 10 * np.log10(255**2 / mse_scale) if mse_scale > 0 else 100
    print(f"Range Corrected ROI PSNR: {psnr_scale:.2f} dB")

    # 3. Shift Test (X, Y)
    best_psnr = 0
    best_shift = (0, 0)
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if dx == 0 and dy == 0: continue
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            f_r_shifted = cv2.warpAffine(f_r, M, PROCESSING_RESOLUTION)
            # Re-generate mask for shifted frame
            m_s = (cv2.cvtColor(f_r_shifted, cv2.COLOR_BGR2GRAY) > 5).astype(np.uint8)
            common_mask = (mask > 0) & (m_s > 0)
            if np.sum(common_mask) == 0: continue
            mse_s = np.mean(((f_o.astype(float) - f_r_shifted.astype(float))**2)[common_mask])
            psnr_s = 10 * np.log10(255**2 / mse_s) if mse_s > 0 else 100
            if psnr_s > best_psnr:
                best_psnr = psnr_s
                best_shift = (dx, dy)
    print(f"Best Shifted ROI PSNR: {best_psnr:.2f} dB at shift {best_shift}")

    cap_o.release()
    cap_r.release()

if __name__ == "__main__":
    main()
