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
import networkx as nx

img_name = 'wen_name_out'
# img_name = 'eric_name_out'
img_dir = '../imgs/'

save_paths = True

# Read image
image_path = Path(img_dir+img_name+'.png')
image = cv2.imread(str(image_path))
## convert image to gray
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## thresholding
_, image_thresh = cv2.threshold(image_gray, 15, 255, cv2.THRESH_BINARY)
# show image
plt.imshow(image_thresh, cmap='gray')
plt.show()

max_width = 11 # in pixels
while True:

    # invert image_thresh
    image_thresh_flip = cv2.bitwise_not(image_thresh)
    ## skeletonize
    image_skeleton = cv2.ximgproc.thinning(image_thresh_flip)
    plt.imshow(image_thresh+image_skeleton, cmap='gray')
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
    
    if round(max_dist) <= max_width:
        print("Max distance is less than max width")
        break
    
    resize_ratio = max_width/max_dist
    # resize the image
    image_thresh = cv2.resize(image_thresh, (int(image_thresh.shape[1]*resize_ratio), int(image_thresh.shape[0]*resize_ratio)), interpolation = cv2.INTER_NEAREST)
    image_thresh = cv2.resize(image_thresh, (int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)), interpolation = cv2.INTER_NEAREST)

## find strokes with deep first search, starting from the edge pixels
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

img_viz = np.ones_like(image_thresh_flip)*255
for stroke in strokes:
    for x, y, w in stroke:
        img_viz = cv2.circle(img_viz, (int(x), int(y)), round(w), 0, -1)
    # cv2.imshow("Image", img_viz)
    # cv2.waitKey(0)

if save_paths:
    ## save to strokes to file
    Path('../path/pixel_path/'+img_name+'/').mkdir(parents=True, exist_ok=True)
    for i in range(len(strokes)):
        np.savetxt('../path/pixel_path/'+img_name+'/'+str(i)+'.csv', strokes[i], delimiter=',')
    ## save resized image
    cv2.imwrite('../imgs/'+img_name+'_resized.png', image_thresh)

## plot out estimated output
image_out = np.ones_like(image_thresh_flip)
for i in range(len(white_pixels[0])):
    ## get the pixel coordinates
    x = white_pixels[1][i]
    y = white_pixels[0][i]    
    image_out = cv2.circle(image_out, (x, y), round(dist_transform[y, x]), 0, -1)
plt.imshow(image_out, cmap='gray')
plt.show()


#### test building graph
directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
white_pix_arr = np.array([white_pixels[1],white_pixels[0]]).T
print(white_pix_arr)
graph = nx.Graph()
draw_graph_pos={}
for i in range(len(white_pix_arr)):
    graph.add_node((white_pix_arr[i][0],white_pix_arr[i][1]),width=round(dist_transform[white_pix_arr[i][1], white_pix_arr[i][0]]))
    draw_graph_pos[(white_pix_arr[i][0],white_pix_arr[i][1])]=(white_pix_arr[i][0],white_pix_arr[i][1])
    for direction in directions:
        if (white_pix_arr[i]+direction).tolist() in white_pix_arr.tolist():
            graph.add_edge((white_pix_arr[i][0],white_pix_arr[i][1]),tuple(white_pix_arr[i]+direction),weight=np.linalg.norm(direction))
for i in range(len(edges)):
    for j in range(i+1,len(edges)):
        dist = np.sqrt((edges[i][0]-edges[j][0])**2 + (edges[i][1]-edges[j][1])**2)
        graph.add_edge((edges[i][0],edges[i][1]),(edges[j][0],edges[j][1]),weight=dist*2)

options = {
    'node_color': 'black',
    'node_size': 10,
}  
nx.draw(graph, draw_graph_pos, **options)
plt.show()
print(graph)

print("Start tsp...")
draw_path = nx.approximation.traveling_salesman_problem(graph, cycle=False)
print("End tsp...")

strokes = []
stroke=[]
for i in range(len(draw_path)):
    stroke.append(draw_path[i])
    if i==len(draw_path)-1:
        strokes.append(stroke)
    elif np.linalg.norm(np.array(draw_path[i])-np.array(draw_path[i+1]))>2:
        strokes.append(stroke)
        stroke=[]

image_out = np.ones_like(image_thresh_flip)*255
for n in draw_path:
    print(n)
    image_out = cv2.circle(image_out, n, round(graph.nodes[n]['width']), 0, -1)
    cv2.imshow("Image", image_out)
    cv2.waitKey(0)