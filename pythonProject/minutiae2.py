import cv2
import numpy as np


def extract_minutiae(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return image_with_keypoints


# Read the sample fingerprint image
sample = cv2.imread("SOCOFing\\Altered\\Altered-Hard\\50__M_Left_index_finger_Obl.BMP")

# Read the real fingerprint image
fingerprint_image = cv2.imread("SOCOFing\\Real\\1__M_Left_index_finger.BMP")

# Extract minutiae from the sample image
sample_minutiae_image = extract_minutiae(sample)

# Extract minutiae from the real fingerprint image
fingerprint_minutiae_image = extract_minutiae(fingerprint_image)

# Display the images with detected keypoints
cv2.imshow("Sample Minutiae", sample_minutiae_image)
cv2.imshow("Fingerprint Minutiae", fingerprint_minutiae_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
