import cv2
import numpy as np

def calculate_stroke_width(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # Calculate the distance transform
    dist_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 3)

    # Normalize the distance transform
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Threshold the distance transform to get the stroke width
    _, stroke_width = cv2.threshold(dist_transform, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return stroke_width

# Example usage
image_path = "record_0131_1/picture.png"
stroke_width = calculate_stroke_width(image_path)
cv2.imshow("Stroke Width", stroke_width)
cv2.waitKey(0)
cv2.destroyAllWindows()

