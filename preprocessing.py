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


def apply_kalman_filter(keypoints):
    """
    Applies a Kalman filter to smooth the keypoints data.
    The Kalman filter assumes that keypoints are a sequence of (x, y) positions.
    
    Args:
        keypoints (np.ndarray): Input keypoints, shape (n_samples, n_features).
        
    Returns:
        np.ndarray: Smoothed keypoints, same shape as input.
    """
    num_samples, num_features = keypoints.shape
    smoothed_keypoints = np.zeros_like(keypoints)

    # Kalman filter parameters
    Q = 1e-5  # Process noise covariance
    R = 1e-2  # Measurement noise covariance
    P = 1.0   # Estimate error covariance
    K = 0.0   # Kalman gain
    x = 0.0   # State estimate

    for feature_idx in range(num_features):
        x = keypoints[0, feature_idx]  # Initialize state estimate
        for t in range(num_samples):
            z = keypoints[t, feature_idx]  # Measurement
            # Prediction step
            P = P + Q
            # Update step
            K = P / (P + R)
            x = x + K * (z - x)
            P = (1 - K) * P
            # Store the smoothed value
            smoothed_keypoints[t, feature_idx] = x
    
    return smoothed_keypoints