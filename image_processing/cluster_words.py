import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from pathlib import Path
import sys
sys.setrecursionlimit(10**6)
sys.path.append('../search_algorithm')
from dfs import DFS

img_name = 'wen_name_out'
# img_name = 'name_cali'
img_dir = '../imgs/'

# Read image
image_path = Path(img_dir+img_name+'.png')
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

## find the distance closest white pixel in image_thresh using distance transform
dist_transform = cv2.distanceTransform(image_thresh_flip, cv2.DIST_L2, 5)

## find white pixels in image_skeleton and loop through them
white_pixels = np.where(image_skeleton == 255)

edge_count = 0
edges = []
for i in range(len(white_pixels[0])):
    x = white_pixels[1][i]
    y = white_pixels[0][i]
    white_n = np.sum(image_skeleton[y-1:y+2, x-1:x+2] > 0)
    if white_n==2:
        edge_count+=1
        edges.append([x, y])
print(f"Number of white pixels with 2 white neighbors: {edge_count}")

## find min max stroke width
max_dist = max(dist_transform[white_pixels])
min_dist = min(dist_transform[white_pixels])
print(f"Max distance: {max_dist}, Min distance: {min_dist}")

## find strokes with deep first search, starting from the edge pixels
dfs = DFS(image_skeleton, edges)
strokes = dfs.search(from_edge=True)
## split strokes into segments, and find the width of each segment
strokes_split = []
for m in range(len(strokes)):
    indices=[]
    widths=[]
    for i in range(len(strokes[m])-1):
        if np.linalg.norm(strokes[m][i]-strokes[m][i+1])>2:
            indices.append(i+1)
        # get the width of the point on the stroke
        widths.append(dist_transform[strokes[m][i][1],strokes[m][i][0]])
    widths.append(dist_transform[strokes[m][-1][1],strokes[m][-1][0]])
    
    #split path
    path_split=np.split(strokes[m],indices)
    widths_split=np.split(widths,indices)
    for i in range(len(path_split)):
        strokes_split.append(np.hstack((path_split[i],widths_split[i].reshape(-1,1))))
strokes = strokes_split

img_viz = np.ones_like(image_thresh_flip)*255
for stroke in strokes:
    for x, y, w in stroke:
        img_viz = cv2.circle(img_viz, (int(x), int(y)), round(w), 0, -1)
    # cv2.imshow("Image", img_viz)
    # cv2.waitKey(0)

## save to strokes to file
Path('../path/pixel_path/'+img_name+'/').mkdir(parents=True, exist_ok=True)
for i in range(len(strokes)):
    np.savetxt('../path/pixel_path/'+img_name+'/'+str(i)+'.csv', strokes[i], delimiter=',')

## plot out estimated output
image_out = np.ones_like(image_thresh_flip)
for i in range(len(white_pixels[0])):
    ## get the pixel coordinates
    x = white_pixels[1][i]
    y = white_pixels[0][i]    
    image_out = cv2.circle(image_out, (x, y), round(dist_transform[y, x]), 0, -1)
plt.imshow(image_out, cmap='gray')
plt.show()