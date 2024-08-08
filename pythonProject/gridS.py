import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid


def load_image(path):
    """Load an image from a file path"""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image from {path}")
    return image


def preprocess_image(image, kernel_size):
    """Apply Gaussian blur to reduce noise"""
    return cv2.GaussianBlur(image, kernel_size, 0)


def edge_detection(image, low_threshold, high_threshold):
    """Perform Canny edge detection"""
    return cv2.Canny(image, low_threshold, high_threshold)


def calculate_score(edges):
    """Calculate a score based on edge density"""
    return np.sum(edges) / (edges.shape[0] * edges.shape[1])


def grid_search(image, param_grid):
    """Perform grid search to find the best hyperparameters"""
    best_score = 0
    best_params = {}

    for params in ParameterGrid(param_grid):
        blurred = preprocess_image(image, params['kernel_size'])
        edges = edge_detection(blurred, params['low_threshold'], params['high_threshold'])
        score = calculate_score(edges)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


def main():
    # Path to the fingerprint image
    image_path = r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Impression_2\fp_1\55.jpg"

    # Load the image
    fingerprint_image = load_image(image_path)

    # Define parameter grid
    param_grid = {
        'kernel_size': [(3, 3), (5, 5), (7, 7)],
        'low_threshold': [50, 100, 150],
        'high_threshold': [150, 200, 250]
    }

    # Perform grid search
    best_params, best_score = grid_search(fingerprint_image, param_grid)

    # Apply the best parameters
    best_blurred = preprocess_image(fingerprint_image, best_params['kernel_size'])
    best_edges = edge_detection(best_blurred, best_params['low_threshold'], best_params['high_threshold'])

    # Display the original and edge-detected images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(fingerprint_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(best_edges, cmap='gray')
    plt.title("Edge Detection (Best Parameters)")
    plt.axis('off')

    plt.show()

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)


if __name__ == "__main__":
    main()
