import cv2
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 20, 150)

    return edges


def get_edge_points(edges, sample_size=10000):
    # Get the coordinates of edge points
    points = np.column_stack(np.where(edges > 0))

    # Subsample the points to reduce memory usage
    if len(points) > sample_size:
        points = points[np.random.choice(points.shape[0], sample_size, replace=False)]

    # Normalize the points
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    return points_scaled


def compute_persistence_diagrams(points_scaled):
    # Compute the persistence diagrams
    diagrams = ripser(points_scaled)['dgms']
    return diagrams


def plot_persistence_diagrams_and_barcodes(diagrams):
    # Disable LaTeX
    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = ''

    plt.figure(figsize=(12, 6))

    # Plot persistence diagrams
    plt.subplot(1, 2, 1)
    plot_diagrams(diagrams, show=False)
    plt.title("Persistence Diagrams", fontsize=12)

    # Plot barcodes
    plt.subplot(1, 2, 2)
    for i, dgm in enumerate(diagrams):
        for pt in dgm:
            if pt[1] < np.inf:
                plt.plot([pt[0], pt[1]], [i, i], color='b')
    plt.title("Barcodes", fontsize=12)
    plt.xlabel("Birth", fontsize=10)
    plt.ylabel("Homology Dimension", fontsize=10)
    plt.show()


def main():
    image_path = r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Impression_2\fp_1\55.jpg"

    # Load and preprocess the image
    edges = load_and_preprocess_image(image_path)

    # Get edge points
    points_scaled = get_edge_points(edges, sample_size=5000)

    # Compute persistence diagrams
    diagrams = compute_persistence_diagrams(points_scaled)

    # Plot persistence diagrams and barcodes
    plot_persistence_diagrams_and_barcodes(diagrams)


if __name__ == "__main__":
    main()
