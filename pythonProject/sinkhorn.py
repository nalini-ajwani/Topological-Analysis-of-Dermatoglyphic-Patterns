import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def extract_keypoints(image_path):
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use SIFT to extract keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints


def calculate_distance_matrix(keypoints):
    # Extract the coordinates of the keypoints
    points = np.array([kp.pt for kp in keypoints])

    # Calculate the pairwise distance matrix
    distance_matrix = cdist(points, points, metric='euclidean')

    return distance_matrix


def sinkhorn_distance(a, b, M, reg, num_iters=100):
    # a and b are histograms (probability distributions) over the points
    # M is the distance matrix
    # reg is the regularization parameter
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    # Ensure a and b are normalized
    a /= a.sum()
    b /= b.sum()

    K = np.exp(-M / reg)
    U = K * M

    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(num_iters):
        u = a / (K @ v)
        v = b / (K.T @ u)

    # Compute the Sinkhorn distance
    distance = np.sum(u[:, None] * K * v[None, :] * M)

    return distance


# Path to the fingerprint image
image_path = r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Impression_2\fp_1\55.jpg"

# Extract keypoints from the fingerprint image
keypoints = extract_keypoints(image_path)

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(keypoints)

# Define the histograms (uniform distribution in this case)
num_points = len(keypoints)
a = np.ones(num_points) / num_points
b = np.ones(num_points) / num_points

# Set the regularization parameter
reg = 0.1

# Calculate the Sinkhorn distance
distance = sinkhorn_distance(a, b, distance_matrix, reg)

print("Sinkhorn Distance:", distance)
