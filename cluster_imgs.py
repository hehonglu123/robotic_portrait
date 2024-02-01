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

## convert image to gray
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## thresholding
_, image_thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
# show image
plt.imshow(image_thresh, cmap='gray')
plt.show()

## find all black pixels
black_pixels = np.where(image_thresh == 0)
image_values = np.zeros_like(image_thresh)
## loop through all black pixels
for i in range(len(black_pixels[0])):
    print(i/len(black_pixels[0])*100)
    ## get x and y of black pixel
    x = black_pixels[1][i]
    y = black_pixels[0][i]
    
    for radias in range(5,0,-1):
        image_white = np.ones_like(image_thresh)
        image_circle = cv2.circle(image_white, (x, y), radias, 0, -1)
        
        ## element-wise OR between image_thresh and image_circle
        image_or = np.logical_or(image_thresh, image_circle)
        ## if the OR operation is different from image_circle, then continue
        if (image_or != image_circle).any():
            continue
        else:
            ## image_values at x y has value of radias
            image_values[y, x] = radias
            break
plt.imshow(image_values)
plt.show()

# # Apply morphological opening to remove smaller strokes
# kernel = np.ones((5, 5), np.uint8)
# image_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# # show image
# cv2.imshow("Image", image_opened)
# cv2.waitKey(0)