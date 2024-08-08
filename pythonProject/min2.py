import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

    return image_blurred

def extract_minutiae(image):
    # Thresholding to create a binary image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw minutiae
    minutiae_image = np.zeros_like(image)

    # Loop through contours to find minutiae
    for contour in contours:
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # Calculate the angle at the far point
                angle = np.arctan2(far[1] - start[1], far[0] - start[0]) * 180.0 / np.pi

                # Filter minutiae based on angle
                if angle < 165 and angle > 15:
                    cv2.circle(minutiae_image, far, 5, (255, 255, 255), -1)

    return minutiae_image

# Path to the fingerprint image
image_path = r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Impression_1\fp_1\24.jpg"

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Extract minutiae
minutiae_image = extract_minutiae(preprocessed_image)

# Display the result using matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(preprocessed_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Minutiae Image')
plt.imshow(minutiae_image, cmap='gray')
plt.axis('off')

plt.show()
