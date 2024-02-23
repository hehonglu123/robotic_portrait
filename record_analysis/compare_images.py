import numpy as np
import cv2
import matplotlib.pyplot as plt

# source_image_name = 'honglu_0221.png'
source_image_name = 'honglu_0207.png'
target_image_name = '../imgs/me_out_resized.png'
# source_image_name = 'wen_name_0221.png'
# source_image_name = 'wen_name_0207.png'
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

image_stack = np.zeros_like(image1, dtype=np.uint8)
image_stack[:,:,0] = image_origin
image_stack[:,:,1] = img_drawn
image_stack = image_stack[top_left[1]:top_left[1]+draw_h, top_left[0]:top_left[0]+draw_w]
plt.imshow(image_stack)
plt.show()

# Compute the pixel-wise difference between the images
diff_img = image_origin.astype(np.int32) - img_drawn.astype(np.int32)
# plt.imshow(diff_img)
# plt.colorbar()
# plt.show()

total_blackpix = np.count_nonzero(img_template <200)
total_whitepix = np.count_nonzero(img_template >= 200)
print("Total pixels: ", img_template.shape[0]*img_template.shape[1])
print("Total black pixels: ", total_blackpix)
print("Total white pixels: ", total_whitepix)
print("True drawing rate (%):",round(np.count_nonzero(np.bitwise_and(diff_img==0,image_origin<200))/total_blackpix*100,2))
print("False drawing rate (%):",round(np.count_nonzero(np.bitwise_and(diff_img>0,image_origin>=200))/total_whitepix*100,2))