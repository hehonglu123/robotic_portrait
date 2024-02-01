import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

def check_new_pixel_covered(image_covered, image_circle, bpx, bpy, radias):
    
    y_len, x_len = image_thresh.shape
    # get circle boundaries
    x_min = max(0,bpx-radias)
    y_min = max(0,bpy-radias)
    x_max = min(x_len,bpx+radias+1)
    y_max = min(y_len,bpy+radias+1)
    
    ## element-wise OR between image_covered and image_circle
    image_or = np.logical_or(image_covered[y_min:y_max,x_min:x_max], image_circle[y_min:y_max,x_min:x_max])
    
    ## if the OR operation is different from image_circle, then continue
    if (image_or == False).any():
        return True
    else:
        # if radias<5:
        #     plt.imshow(image_covered[y_min:y_max,x_min:x_max], cmap='gray')
        #     plt.show()
        #     plt.imshow(image_circle[y_min:y_max,x_min:x_max], cmap='gray')
        #     plt.show()
        #     plt.imshow(image_or, cmap='gray')
        #     plt.show()
        #     input("Not covered")
        return False

def check_circle_valid(image_thresh, image_circle, bpx, bpy, radias):
    
    y_len, x_len = image_thresh.shape
    # get circle boundaries
    x_min = max(0,bpx-radias)
    y_min = max(0,bpy-radias)
    x_max = min(x_len,bpx+radias+1)
    y_max = min(y_len,bpy+radias+1)
    
    ## element-wise OR between image_thresh and image_circle
    image_or = np.logical_or(image_thresh[y_min:y_max,x_min:x_max], image_circle[y_min:y_max,x_min:x_max])
    ## if the OR operation is different from image_circle, then continue
    if (image_or != image_circle[y_min:y_max,x_min:x_max]).any():
        return False
    else:
        return True

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

# # invert image_thresh
# image_thresh = cv2.bitwise_not(image_thresh)
# ## skeletonize
# image_skeleton = cv2.ximgproc.thinning(image_thresh)
# plt.imshow(image_skeleton, cmap='gray')
# plt.show()

## find all black pixels
black_pixels = np.where(image_thresh == 0)

##### covered the image with circles ####
max_radias = 8
min_radias = 2
## create image_covered, image_vis, image_node
image_covered = deepcopy(image_thresh)
image_vis = np.ones_like(image_thresh)
image_node = np.zeros_like(image_thresh)
## loop through radias max_radias to 1
for radias in range(max_radias,min_radias-1,-1):
    print("Radias: ", radias)
    ## loop through all black pixels
    for i in range(len(black_pixels[0])):
        if i % 1000 == 0:
            print(i/len(black_pixels[0])*100)
        ## get x and y of black pixel
        x = black_pixels[1][i]
        y = black_pixels[0][i]
        
        image_white = np.ones_like(image_thresh)
        image_circle = cv2.circle(image_white, (x, y), radias, 0, -1)
        # check if circle is valid
        # valid_circle = check_circle_valid(image_thresh, image_circle, x, y, radias)
        valid_circle = check_circle_valid(image_covered, image_circle, x, y, radias)
        if not valid_circle:
            continue
        ## check if new pixel covered
        covered_pixel = check_new_pixel_covered(image_covered, image_circle, x, y, radias)
        if not covered_pixel:
            continue
        
        ## image_covered at x y has value of radias
        image_circle = np.logical_not(image_circle)
        image_covered = np.logical_or(image_covered, image_circle)

        ## image_vis at x y has value of radias
        image_vis = cv2.circle(image_vis, (x, y), radias, 0, -1)
        
        ## add node
        image_node[y,x] = radias

plt.imshow(image_vis, cmap='gray')
plt.show()

plt.imshow(image_node*10)
plt.show()

# # Apply morphological opening to remove smaller strokes
# kernel = np.ones((5, 5), np.uint8)
# image_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# # show image
# cv2.imshow("Image", image_opened)
# cv2.waitKey(0)