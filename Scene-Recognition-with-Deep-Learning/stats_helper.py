import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################
    sum_pixels = 0.0
    sum_squared_pixels = 0.0
    count = 0

    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path).convert('L')
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    pixels = img_array.flatten()
                    sum_pixels += pixels.sum()
                    sum_squared_pixels += (pixels ** 2).sum()
                    count += pixels.size
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    if count > 0:
        mean = sum_pixels / count
        variance = (sum_squared_pixels / count) - (mean ** 2)
        std = np.sqrt(variance)
    else:
        print("No images found or all images failed to process.")
        mean, std = 0.0, 0.0
    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
