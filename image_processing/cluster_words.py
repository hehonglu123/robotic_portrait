import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

img_name = 'wen_name_out.png'
img_dir = '../imgs/'

# Read image
image_path = Path(img_dir+img_name)
image = cv2.imread(str(image_path))
# show image
cv2.imshow("Image", image)
cv2.waitKey(0)

## convert image to gray
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## thresholding
_, image_thresh = cv2.threshold(image_gray, 15, 255, cv2.THRESH_BINARY)
# show image
plt.imshow(image_thresh, cmap='gray')
plt.show()

# invert image_thresh
image_thresh_flip = cv2.bitwise_not(image_thresh)
## skeletonize
image_skeleton = cv2.ximgproc.thinning(image_thresh_flip)
plt.imshow(image_skeleton+image_thresh, cmap='gray')
plt.show()