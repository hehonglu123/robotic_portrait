import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import networkx as nx

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

save_paths=True

# Read image
img_name='me_out'
image_path = Path("../imgs/"+img_name+".png")
image = cv2.imread(str(image_path))
# show image
# cv2.imshow("Image", image)
# cv2.waitKey(0)

## convert image to gray
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## thresholding
_, image_thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
# show image
plt.imshow(image_thresh, cmap='gray')
plt.show()

# resize image
size_ratio = 2
print("Original image size: ", image_thresh.shape)
image_thresh = cv2.resize(image_thresh, (int(image_thresh.shape[1]*size_ratio), int(image_thresh.shape[0]*size_ratio)), interpolation = cv2.INTER_NEAREST)
print("Resized image size: ", image_thresh.shape)

## find all black pixels
black_pixels = np.where(image_thresh == 0)

##### covered the image with circles ####
max_radias = 10
min_radias = 2
## create image_covered, image_vis, image_node
image_covered = deepcopy(image_thresh)
image_vis = np.ones_like(image_thresh)
image_node = np.zeros_like(image_thresh)
nodes = []
graph = nx.Graph()
draw_graph_pos={}
## loop through radias max_radias to 1
for radias in range(max_radias,min_radias-1,-1):
    print("Radias: ", radias)
    ## loop through all black pixels
    for i in range(len(black_pixels[0])):
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
        nodes.append([x,y,radias])
        graph.add_node((x,y),width=radias) ## add circle center pixel as nodes
        draw_graph_pos[(x,y)]=(x,y)

## find the distance closest black pixel in image_viz using distance transform
dist_transform = cv2.distanceTransform(image_vis, cv2.DIST_L2, 5)

plt.imshow(image_vis, cmap='gray')
plt.show()

print("Total nodes: ", len(nodes))
blank_thres = 1.5
leave_paper_weight=2
for i in range(len(nodes)):
    for j in range(i+1,len(nodes)):
        dist = np.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2)
        if dist < nodes[i][2]+nodes[j][2]+blank_thres:
            # graph.add_node(((nodes[i][0]+nodes[j][0])/2,(nodes[i][1]+nodes[j][1])/2))
            # graph.add_edge((nodes[i][0],nodes[i][1]),((nodes[i][0]+nodes[j][0])/2,(nodes[i][1]+nodes[j][1])/2))
            # graph.add_edge((nodes[j][0],nodes[j][1]),((nodes[i][0]+nodes[j][0])/2,(nodes[i][1]+nodes[j][1])/2))
            graph.add_edge((nodes[i][0],nodes[i][1]),(nodes[j][0],nodes[j][1]),weight=dist)
            image_node = cv2.line(image_node, (nodes[i][0],nodes[i][1]), (nodes[j][0],nodes[j][1]), max(nodes[i][2],nodes[j][2]), 1)
        else:
            # graph.add_edge((nodes[i][0],nodes[i][1]),(nodes[j][0],nodes[j][1]),weight=dist*leave_paper_weight)
            pass
print("graph",graph)

options = {
    'node_color': 'black',
    'node_size': 10,
}  
nx.draw(graph, draw_graph_pos, **options)
plt.show()

## check isolation
graph.remove_nodes_from(list(nx.isolates(graph)))
nx.draw(graph,pos=draw_graph_pos, **options)
plt.show()

## check connectivity
all_graphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

print("Start tsp...")
total_g=len(all_graphs)
all_paths=[]
count=1
for subg in all_graphs:    
    print("graph: ",count,"/",total_g," nodes: ",subg.number_of_nodes(), " edges: ", subg.number_of_edges())
    draw_path = nx.approximation.traveling_salesman_problem(subg, cycle=False)
    all_paths.append(draw_path)
    count+=1
print("End tsp...")

strokes_split = []
for i in range(len(all_paths)):
    stroke=[]
    for j in range(len(all_paths[i])):
        width = graph.nodes[all_paths[i][j]]['width']
        this_p = list(all_paths[i][j])
        this_p.append(width)
        stroke.append(this_p)
    strokes_split.append(stroke)
strokes=strokes_split

if save_paths:
    ## save to strokes to file
    Path('../path/pixel_path/'+img_name+'/').mkdir(parents=True, exist_ok=True)
    for i in range(len(strokes)):
        np.savetxt('../path/pixel_path/'+img_name+'/'+str(i)+'.csv', strokes[i], delimiter=',')
    ## save resized image
    cv2.imwrite('../imgs/'+img_name+'_resized.png', image_thresh)

# options = {
#     'node_color': 'black',
#     'node_size': 100,
#     'width': 3,
# }  
# nx.draw_random(graph, **options)
# plt.show()

image_node_viz=image_node*10
image_node_viz[0,0]=1
plt.imshow(image_node*10)
plt.show()

# # Apply morphological opening to remove smaller strokes
# kernel = np.ones((5, 5), np.uint8)
# image_opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
# # show image
# cv2.imshow("Image", image_opened)
# cv2.waitKey(0)