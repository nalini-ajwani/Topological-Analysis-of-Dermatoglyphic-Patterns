import cv2
import numpy as np
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
    """Display an image with keypoints"""
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Image with Keypoints", image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cluster_minutiae(keypoints, num_clusters):
    """Cluster minutiae datapoints using K-Means"""
    keypoints_array = np.array([kp.pt for kp in keypoints])
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(keypoints_array)
    return kmeans.labels_

def main():
    # Load fingerprint image
    fingerprint_image_path = r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Impression_2\fp_1\55.jpg"
    fingerprint_image = load_image(fingerprint_image_path)

    # Extract minutiae from fingerprint image
    fingerprint_keypoints, _ = extract_minutiae(fingerprint_image)

    # Display image with keypoints
    display_image_with_keypoints(fingerprint_image, fingerprint_keypoints)

    # Cluster minutiae datapoints
    num_clusters = 5
    fingerprint_labels = cluster_minutiae(fingerprint_keypoints, num_clusters)

    # Print clustering results
    print("Fingerprint Minutiae Clustering Labels:", fingerprint_labels)

if __name__ == "__main__":
    main()
