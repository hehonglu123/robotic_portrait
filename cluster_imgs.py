import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Read image
image_path = Path("imgs/me_out.png")
image = cv2.imread(str(image_path))
# show image
cv2.imshow("Image", image)
cv2.waitKey(0)

# Apply morphological opening to remove smaller strokes
kernel = np.ones((5, 5), np.uint8)
image_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# show image
cv2.imshow("Image", image_opened)
cv2.waitKey(0)

# Apply Flow-based Difference-of Gaussian (FDoG) filtering
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_fdog = gaussian_filter(image_gray, sigma=1) - gaussian_filter(image_gray, sigma=3)

# show image
cv2.imshow("Image", image_fdog)
cv2.waitKey(0)

### give values to each black pixels, and use them for path planning, like a*