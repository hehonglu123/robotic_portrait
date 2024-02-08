import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import networkx as nx
import sys
sys.path.append('../search_algorithm')
from dfs import DFS

img_name = 'me_out'
img_dir = '../imgs/'
max_width = 11 # in pixels

# Read image
image_path = Path(img_dir+img_name+'.png')
image = cv2.imread(str(image_path))
## convert image to gray
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## thresholding
_, image_thresh = cv2.threshold(image_gray, 15, 255, cv2.THRESH_BINARY)
# resize image
size_ratio = 2
print("Original image size: ", image_thresh.shape)
image_thresh = cv2.resize(image_thresh, (int(image_thresh.shape[1]*size_ratio), int(image_thresh.shape[0]*size_ratio)), interpolation = cv2.INTER_NEAREST)
print("Resized image size: ", image_thresh.shape)
# show image
plt.imshow(image_thresh, cmap='gray')
plt.show()

image_filled = deepcopy(image_thresh)
image_vis = deepcopy(image_thresh)*2/3

while True:
    ## invert image_thresh
    image_thresh_flip = cv2.bitwise_not(image_filled)
    ## skeletonize
    image_skeleton = cv2.ximgproc.thinning(image_thresh_flip)
    plt.imshow(image_vis+image_skeleton, cmap='gray')
    # plt.imshow(image_filled, cmap='gray')
    plt.show()

    ## find the distance closest black pixel in image_thresh using distance transform
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

    for i in range(len(white_pixels[0])):
        x = white_pixels[1][i]
        y = white_pixels[0][i]
        # if dist_transform[y][x]<=max_width:
        image_filled = cv2.circle(image_filled, (x, y), min(max(int(dist_transform[y][x]),1),max_width)+1, 255, -1)
        image_vis = cv2.circle(image_vis, (x, y), min(max(int(dist_transform[y][x]),1),max_width)+1, 120, -1)
    
    ## find strokes with deep first search, starting from the edge pixels
    plt.imshow(image_skeleton, cmap='gray')
    plt.show()

    dfs = DFS(image_skeleton, edges)
    strokes = dfs.search(from_edge=True)
    ## split strokes into segments, and find the width of each segment
    strokes_split = []
    all_wdiths = []
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
            all_wdiths.extend(widths_split[i])
    strokes = strokes_split