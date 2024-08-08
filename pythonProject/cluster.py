import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_image(path):
    """Load an image from a file path"""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image from {path}")
    return image


def extract_minutiae(image):
    """Extract minutiae from an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def display_image_with_keypoints(image, keypoints):
    """Display an image with keypoints using Matplotlib"""
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_with_keypoints = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_keypoints)
    plt.axis('off')
    plt.show()


def cluster_minutiae(keypoints, num_clusters):
    """Cluster minutiae datapoints using K-Means"""
    keypoints_array = np.array([kp.pt for kp in keypoints])
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(keypoints_array)
    return kmeans.labels_, kmeans.cluster_centers_


def plot_clusters(image, keypoints, labels, centers):
    """Plot clusters of keypoints"""
    keypoints_array = np.array([kp.pt for kp in keypoints])

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(keypoints_array[:, 0], keypoints_array[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='red', s=100, label='Cluster Centers')
    plt.colorbar()
    plt.title('Minutiae Clustering')
    plt.legend()
    plt.axis('off')
    plt.show()


def main():
    # Load the fingerprint image
    image_path = r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Fingerprints\fp_1\164.jpg"
    fingerprint_image = load_image(image_path)

    # Extract minutiae from the fingerprint image
    fingerprint_keypoints, _ = extract_minutiae(fingerprint_image)

    # Cluster minutiae datapoints
    num_clusters = 5
    fingerprint_labels, cluster_centers = cluster_minutiae(fingerprint_keypoints, num_clusters)

    # Plot clusters of keypoints
    plot_clusters(fingerprint_image, fingerprint_keypoints, fingerprint_labels, cluster_centers)


if __name__ == "__main__":
    main()
