'''import cv2
import numpy as np
from skimage.restoration import denoise_tv_bregman

def load_image(path):
    """Load an image from a file path"""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image from {path}")
    return image

def denoise_image(image):
    """Remove noise from an image using Total Variation denoising"""
    denoised_image = denoise_tv_bregman(image, weight=0.6)
    return denoised_image

def main():
    # Load high-resolution fingerprint image
    fingerprint_image = load_image(r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Impression_2\fp_1\55.jpg")
    #SOCOFing\\Real\\1__M_Left_index_finger.BMP

    # Denoise the image
    denoised_image = denoise_image(fingerprint_image)

    # Display original and denoised images
    cv2.imshow("Original Image", fingerprint_image)
    cv2.imshow("Denoised Image", denoised_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    """Load an image from a file path"""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image from {path}")
    return image


def convert_to_sketch(image):
    """Convert an image to a pencil sketch using OpenCV"""
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_image = cv2.bitwise_not(gray_image)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)

    # Invert the blurred image
    inverted_blurred_image = cv2.bitwise_not(blurred_image)

    # Create the pencil sketch
    sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)

    return sketch_image


def main():
    # Path to the fingerprint image
    image_path = r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Impression_2\fp_1\55.jpg"

    # Load the image
    fingerprint_image = load_image(image_path)

    # Convert the image to a sketch
    sketch_image = convert_to_sketch(fingerprint_image)

    # Display the original and sketched images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sketch_image, cmap='gray')
    plt.title("Sketch Image")
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()

