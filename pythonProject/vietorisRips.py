import cv2
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd


def load_image(path):
    """Load an image from a file path"""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image from {path}")
    return image


def preprocess_image(image):
    """Preprocess the image to extract feature points"""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def extract_feature_points(edges):
    """Extract feature points from edge-detected image"""
    points = np.column_stack(np.where(edges > 0))
    return points


def vietoris_rips_filtration(points, max_edge_length, max_dimension):
    """Construct Vietoris-Rips complex and return the simplex tree"""
    rips_complex = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    return simplex_tree


def plot_persistence_diagram(simplex_tree):
    """Plot persistence diagram using matplotlib"""
    diag = simplex_tree.persistence()
    gd.plot_persistence_diagram(diag)
    plt.show()

    # Alternatively, plot using matplotlib directly
    intervals = simplex_tree.persistence_intervals_in_dimension(1)
    plt.figure()
    for interval in intervals:
        plt.plot([interval[0], interval[1]], [1, 1], 'b-')
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Persistence Diagram")
    plt.show()


def main():
    # Path to the fingerprint image
    image_path = r"D:\CS\minor\minutiaeDetection\pythonProject\Anguli Fingerprints\Class - Natural\Impression_2\fp_1\55.jpg"

    # Load the image
    fingerprint_image = load_image(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(fingerprint_image)

    # Extract feature points
    feature_points = extract_feature_points(preprocessed_image)

    # Apply Vietoris-Rips filtration
    max_edge_length = 20  # Maximum edge length for Vietoris-Rips complex
    max_dimension = 2  # Maximum dimension for simplices in the complex
    simplex_tree = vietoris_rips_filtration(feature_points, max_edge_length, max_dimension)

    # Plot the persistence diagram
    plot_persistence_diagram(simplex_tree)

    print("Vietoris-Rips complex created with filtration")
    print("Persistence diagram plotted")


if __name__ == "__main__":
    main()
