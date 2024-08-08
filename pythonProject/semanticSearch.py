import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Impression_3\fp_1\12.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define parameter grids
gaussian_kernel_sizes = [(3, 3), (5, 5), (7, 7)]
canny_thresholds = [(10, 100), (20, 150), (30, 200)]

best_score = 0
best_params = {}

# Perform grid search
for kernel_size in gaussian_kernel_sizes:
    for threshold in canny_thresholds:
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, kernel_size, 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, threshold[0], threshold[1])

        # Calculate score (e.g., edge density)
        score = np.sum(edges) / (edges.shape[0] * edges.shape[1])

        # Update best parameters if score is better
        if score > best_score:
            best_score = score
            best_params = {'kernel_size': kernel_size, 'threshold': threshold}

# Apply the best parameters to generate the semantic sketch
blurred = cv2.GaussianBlur(gray, best_params['kernel_size'], 0)
edges = cv2.Canny(blurred, best_params['threshold'][0], best_params['threshold'][1])

# Display the original image and the semantical sketch
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Semantic Sketch (Best Parameters)")
plt.axis('off')

plt.show()

# Print the best parameters and score
print("Best Parameters:")
print("Best Gaussian Kernel Size:", best_params['kernel_size'])
print("Best Canny Thresholds:", best_params['threshold'])
print("Best Score (Sinkhorn Distance):", best_score)
