import numpy as np
import cv2
import matplotlib.pyplot as plt

source_image_name = 'honglu_0221.png'
target_image_name = '../imgs/me_out_resized.png'
# source_image_name = 'wen_name_0221.png'
# target_image_name = '../imgs/wen_name_out_resized.png'

# Read the images
image1 = cv2.imread(source_image_name)
image2 = cv2.imread(target_image_name)

# Convert the images to grayscale
img_drawn = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img_template = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Apply template Matching
res = cv2.matchTemplate(img_drawn,img_template,cv2.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
draw_w, draw_h = img_template.shape[::-1]
top_left = min_loc
bottom_right = (top_left[0] + draw_w, top_left[1] + draw_h)
# gray1_viz=cv2.rectangle(img_drawn,top_left, bottom_right, 125, 2)

# turn the original image as the size of drawn image
image_origin = np.ones_like(img_drawn).astype(np.uint8)*255
# put the template matching result on the original image
image_origin[top_left[1]:top_left[1]+draw_h, top_left[0]:top_left[0]+draw_w] = img_template

# Compute the pixel-wise difference between the images
diff_img = image_origin.astype(np.int32) - img_drawn.astype(np.int32)
plt.imshow(diff_img)
plt.colorbar()
plt.show()

# Create a new image to visualize the differences
diff_viz = np.zeros_like(image1, dtype=np.uint8)
# Set red color for negative differences
diff_viz[diff_img < 0] = (0, 0, 255)
# Set blue color for positive differences
diff_viz[diff_img > 0] = (255, 0, 0)
# Set black color for zero differences
diff_viz[diff_img == 0] = (0, 0, 0)

# Display the difference image
plt.imshow(diff_viz)
plt.axis('off')
plt.show()