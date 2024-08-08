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
    return kmeans.labels_


def main():
    # Load sample and real fingerprint images
    sample = load_image("SOCOFing\\Altered\\Altered-Hard\\50__M_Left_index_finger_Obl.BMP")
    fingerprint_image = load_image("SOCOFing\\Real\\1__M_Left_index_finger.BMP")

    # Extract minutiae from sample and real fingerprint images
    sample_keypoints, _ = extract_minutiae(sample)
    fingerprint_keypoints, _ = extract_minutiae(fingerprint_image)

    # Display images with keypoints
    display_image_with_keypoints(sample, sample_keypoints)
    display_image_with_keypoints(fingerprint_image, fingerprint_keypoints)

    # Cluster minutiae datapoints
    num_clusters = 5
    sample_labels = cluster_minutiae(sample_keypoints, num_clusters)
    fingerprint_labels = cluster_minutiae(fingerprint_keypoints, num_clusters)

    # Print clustering results
    print("Sample Minutiae Clustering Labels:", sample_labels)
    print("Fingerprint Minutiae Clustering Labels:", fingerprint_labels)


if __name__ == "__main__":
    main()
