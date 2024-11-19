import numpy as np
import cv2

def apply_smoothing_filter(keypoints, window_size=5):
    # Apply a smoothing filter (e.g., moving average)
    smoothed_keypoints = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window_size) / window_size, mode='valid'),
        axis=0, arr=keypoints
    )
    return smoothed_keypoints


def apply_gaussian_filter(keypoints, kernel_size=(3, 3)):
    # Apply Gaussian blur filter (can be implemented with OpenCV)
    
    smoothed_keypoints = cv2.GaussianBlur(keypoints, kernel_size, 0)
    return smoothed_keypoints


def normalize_keypoints(keypoints):
    # Normalize keypoint coordinates (x, y, z) to the range [0, 1] or [-1, 1]
    normalized_keypoints = (keypoints - np.min(keypoints)) / (np.max(keypoints) - np.min(keypoints))
    return normalized_keypoints


def preprocess_keypoints(keypoints, filter_type=None, normalize=True):
    # Apply chosen filter
    if filter_type == 'smoothing':
        keypoints = apply_smoothing_filter(keypoints)
    elif filter_type == 'gaussian':
        keypoints = apply_gaussian_filter(keypoints)
    elif filter_type is None:
        pass  # No filter applied


    # Normalize keypoints if required
    if normalize:
        keypoints = normalize_keypoints(keypoints)


    return keypoints