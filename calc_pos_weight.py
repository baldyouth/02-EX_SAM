import os
import cv2
import numpy as np

def compute_pos_weight(mask_root, keyword=''):
    total_pixels = 0
    positive_pixels = 0

    mask_files = [f for f in os.listdir(mask_root) if (keyword in f and f.endswith(('.png', '.jpg', '.bmp', '.tif')))]
    for filename in mask_files:
        mask_path = os.path.join(mask_root, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: failed to load {mask_path}")
            continue

        mask = (mask > 0).astype(np.uint8)  # 非0为正类

        positive_pixels += mask.sum()
        total_pixels += mask.size

    if positive_pixels == 0:
        raise ValueError("No positive pixels found!")

    pos_weight = (total_pixels - positive_pixels) / positive_pixels
    print(f"Total pixels: {total_pixels}")
    print(f"Positive pixels: {positive_pixels}")
    print(f"pos_weight = {pos_weight:.4f}")

    return pos_weight


mask_root = "/home/swjtu/workspace_01/data/crack_segmentation_dataset/train/masks"
compute_pos_weight(mask_root, keyword='DeepCrack')
