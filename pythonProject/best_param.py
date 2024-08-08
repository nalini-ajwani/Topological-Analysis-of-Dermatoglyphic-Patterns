import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist


# Function to generate semantic sketch
def generate_sketch(image, kernel_size, canny_threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, kernel_size, 0)
    edges = cv2.Canny(blurred, canny_threshold[0], canny_threshold[1])
    return edges


# Function to measure similarity
def measure_similarity(original, sketch):
    mse = mean_squared_error(original, sketch)
    ssim_value = ssim(original, sketch, data_range=sketch.max() - sketch.min())
    return mse, ssim_value


# Function to calculate the Sinkhorn distance
def sinkhorn_distance(M, reg, num_iters=100):
    a = np.ones((M.shape[0],)) / M.shape[0]
    b = np.ones((M.shape[1],)) / M.shape[1]

    K = np.exp(-M / reg)
    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(num_iters):
        u = a / (K @ v)
        v = b / (K.T @ u)

    distance = np.sum(u[:, None] * K * v[None, :] * M)

    return distance


# Function to calculate pairwise distance matrix
def calculate_distance_matrix(points):
    distance_matrix = cdist(points, points, metric='euclidean')
    return distance_matrix


# Function to extract key points using SIFT
def extract_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(image, None)
    return keypoints


# Function to optimize parameters
def optimize_parameters(image, kernel_sizes, canny_thresholds, reg=0.1):
    best_params = None
    best_score = -float('inf')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract points from original image for Sinkhorn distance
    keypoints = extract_keypoints(gray)
    if len(keypoints) == 0:
        raise ValueError("No keypoints found in the original image.")
    original_points = np.array([kp.pt for kp in keypoints])
    original_dist_matrix = calculate_distance_matrix(original_points)

    for kernel_size in kernel_sizes:
        for threshold in canny_thresholds:
            sketch = generate_sketch(image, kernel_size, threshold)
            sketch_resized = cv2.resize(sketch, (gray.shape[1], gray.shape[0]))
            mse, ssim_value = measure_similarity(gray, sketch_resized)

            # Extract points from sketch for Sinkhorn distance
            keypoints_sketch = extract_keypoints(sketch)
            if len(keypoints_sketch) == 0:
                continue
            sketch_points = np.array([kp.pt for kp in keypoints_sketch])
            sketch_dist_matrix = calculate_distance_matrix(sketch_points)

            # Calculate Sinkhorn distance
            sinkhorn_dist = sinkhorn_distance(original_dist_matrix, reg)

            score = ssim_value - mse - sinkhorn_dist  # Combining SSIM, MSE, and Sinkhorn distance to form a single score
            if score > best_score:
                best_score = score
                best_params = {'kernel_size': kernel_size, 'threshold': threshold}

    return best_params, best_score, best_params['kernel_size'], best_params['threshold'], sinkhorn_dist


# Function to display images
def display_images(original, sketch):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sketch, cmap='gray')
    plt.title("Semantic Sketch (Best Parameters)")
    plt.axis('off')

    plt.show()
# Load the image
image_path = r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Impression_2\fp_1\55.jpg"
image = cv2.imread(image_path)

# Define parameter grids
gaussian_kernel_sizes = [(3, 3), (5, 5), (7, 7)]
canny_thresholds = [(10, 100), (20, 150), (30, 200)]

# Optimize parameters
best_params, best_score, best_kernel_size, best_threshold, best_sinkhorn_dist = optimize_parameters(image, gaussian_kernel_sizes, canny_thresholds)

# Generate the semantic sketch using the best parameters
best_sketch = generate_sketch(image, best_params['kernel_size'], best_params['threshold'])

# Display the original image and the best semantic sketch
display_images(image, best_sketch)

# Print the best parameters and score
print("Best Parameters:", best_params)
print("Best Gaussian Kernel Size:", best_kernel_size)
print("Best Canny Thresholds:", best_threshold)
print("Best Score (Combined SSIM - MSE - Sinkhorn Distance):", best_score)
print("Best Sinkhorn Distance:", best_sinkhorn_dist)
