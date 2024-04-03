import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import networkx as nx
import sys
sys.path.append('../search_algorithm')
from dfs import DFS

# img_name = 'me_out'
# img_dir = '../imgs/'
img_name = 'img_out.jpg'
img_dir = '../temp_data/'
max_width = 11 # in pixels
min_stroke_length = 20 # in pixels
min_white_pixels = 50
save_paths = True

# Read image
image_path = Path(img_dir+img_name)
image = cv2.imread(str(image_path))
## convert image to gray
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## add white boarder
image_gray = cv2.copyMakeBorder(image_gray, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
## thresholding
_, image_thresh = cv2.threshold(image_gray, 15, 255, cv2.THRESH_BINARY)
# resize image
# size_ratio = 2
target_size=[1200,800]
size_ratio=np.mean(np.divide(target_size,image.shape[:2]))
print("Original image size: ", image_thresh.shape)
image_thresh = cv2.resize(image_thresh, (int(image_thresh.shape[1]*size_ratio), int(image_thresh.shape[0]*size_ratio)), interpolation = cv2.INTER_NEAREST)
print("Resized image size: ", image_thresh.shape)
# show image
plt.imshow(image_thresh, cmap='gray')
plt.show()

image_filled = deepcopy(image_thresh)
image_vis = deepcopy(image_thresh)*2/3

graph=nx.Graph()
in_graph_pix = []
in_graph_radius = []
draw_graph_pos={}

strokes_split = []
count=1
while True:
    ## invert image_thresh
    image_thresh_flip = cv2.bitwise_not(image_filled)
    ## skeletonize
    image_skeleton = cv2.ximgproc.thinning(image_thresh_flip)
    # plt.imshow(image_vis+image_skeleton, cmap='gray')
    # # plt.imshow(image_filled, cmap='gray')
    # plt.show()

    ## find the distance closest black pixel in image_thresh using distance transform
    dist_transform = cv2.distanceTransform(image_thresh_flip, cv2.DIST_L2, 5)

    ## find white pixels in image_skeleton and loop through them
    white_pixels = np.where(image_skeleton == 255)
    image_skeleton = np.zeros_like(image_skeleton)
    white_pixels_removed = []
    image_viz_boarder = deepcopy(image_skeleton)
    for i in range(len(white_pixels[0])):
        x = white_pixels[1][i]
        y = white_pixels[0][i]
        # filter out white pixels with distance smaller than 1
        if dist_transform[y,x]>1:
            white_pixels_removed.append([y,x])
            image_skeleton[y,x]=255
        image_viz_boarder[y,x]=255-dist_transform[y,x]
    
    if count==1:
        white_pixels = np.where(image_skeleton == 255)
    else:
        white_pixels = white_pixels_removed
        white_pixels = np.array(white_pixels).T
        white_pixels = tuple(white_pixels)
    
    # plt.imshow(image_vis+image_viz_boarder, cmap='gray')
    # plt.show()
    
    # plt.imshow(image_vis+image_skeleton, cmap='gray')
    # plt.show()
    
    
    if len(white_pixels)==0:
        print("Number of white pixels is less than min_white_pixels")
        break
    print(f"Number of white pixels: {len(white_pixels[0])}")
    if len(white_pixels[0])<min_white_pixels:
        print("Number of white pixels is less than min_white_pixels")
        break

    edge_count = 0
    edges = []
    for i in range(len(white_pixels[0])):
        x = white_pixels[1][i]
        y = white_pixels[0][i]
        white_n = np.sum(image_skeleton[y-1:y+2, x-1:x+2] > 0)
        if white_n==2:
            edge_count+=1
            edges.append([x, y])
        elif white_n==3:
            white_surrounds = np.argwhere(image_skeleton[y-1:y+2, x-1:x+2] > 0)
            white_surrounds = white_surrounds.tolist()
            white_surrounds.remove([1,1])
            nei_dist = np.linalg.norm(np.diff(white_surrounds, axis=0))
            if nei_dist==1:
                edge_count+=1
                edges.append([x, y])
            
    print(f"Number of white pixels with 2 white neighbors: {edge_count}")
    

    ## find min max stroke width
    max_dist = max(dist_transform[white_pixels])
    min_dist = min(dist_transform[white_pixels])
    print(f"Max distance: {max_dist}, Min distance: {min_dist}")

    ##### draw circle and find nodes and edges
    out_graph_pix = np.array([white_pixels[1],white_pixels[0]]).T
    start_draw = True
    while out_graph_pix.shape[0]>0:
        # move to in graph
        if len(in_graph_pix)==0:
        # if start_draw:
            start_draw = False
            # add first node
            graph.add_node(tuple(np.append(edges[0],min(max(dist_transform[edges[0][1]][edges[0][0]],1),max_width))))
            in_graph_pix = np.array([edges[0]]) # add the in graph pixel liset
            out_graph_pix = np.delete(out_graph_pix,np.all(out_graph_pix==edges[0],axis=1),axis=0) # remove from outside graph list
        else:
            closest_i = np.argmin(np.linalg.norm(out_graph_pix-in_graph_pix[-1],axis=1)) # find the closest pixel to the "draw nodes"
            in_graph_pix = np.vstack((in_graph_pix,out_graph_pix[closest_i])) # add the in graph pixel liset
            out_graph_pix = np.delete(out_graph_pix,closest_i,axis=0) # remove from outside graph list
        this_node_name = tuple(np.append(in_graph_pix[-1],min(max(dist_transform[in_graph_pix[-1][1]][in_graph_pix[-1][0]],1),max_width))) # node name
        draw_radius = min(max(dist_transform[in_graph_pix[-1][1]][in_graph_pix[-1][0]],1),max_width) # drawing radius
        in_graph_radius = np.append(in_graph_radius,draw_radius) # add the in graph radius list
        draw_graph_pos[this_node_name]=tuple(in_graph_pix[-1]) # add the drawing position
        # remove all the pixels that are too close to the in graph
        close_i = np.where(np.linalg.norm(out_graph_pix-in_graph_pix[-1],axis=1)<=in_graph_radius[-1])[0]
        out_graph_pix = np.delete(out_graph_pix,close_i,axis=0)
        # draw the circle
        image_filled = cv2.circle(image_filled, in_graph_pix[-1], min(max(round(dist_transform[in_graph_pix[-1][1]][in_graph_pix[-1][0]]),1),max_width), 255, -1)
        image_vis = cv2.circle(image_vis, in_graph_pix[-1], min(max(round(dist_transform[in_graph_pix[-1][1]][in_graph_pix[-1][0]]),1),max_width), 120, -1)
        
        if len(in_graph_pix)==1:
            continue
        # draw edge if there's one
        edges_node_ids = np.where(np.linalg.norm(in_graph_pix[:-1]-in_graph_pix[-1],axis=1)<=in_graph_radius[:-1]+in_graph_radius[-1])[0]
        for eni in edges_node_ids:
            graph.add_edge(tuple(np.append(in_graph_pix[eni],in_graph_radius[eni])),this_node_name\
                ,weight=np.linalg.norm(in_graph_pix[eni]-in_graph_pix[-1]))
        
        if False:
            # if len(edges_node_ids)==0:
            print("Previous node: ",in_graph_pix[-2])
            print("Previous radius: ",in_graph_radius[-2])
            print("Add node",in_graph_pix[-1]," to graph")
            print("build edges: ",in_graph_pix[edges_node_ids])
            print("=================")
            print(out_graph_pix.shape[0])
            cv2.imshow("Image", image_vis)
            cv2.waitKey(0)
        
    ## find strokes with deep first search, starting from the edge pixels
    # plt.imshow(image_filled, cmap='gray')
    # plt.show()
    # plt.imshow(image_vis, cmap='gray')
    # plt.show()
    
    count+=1

## check isolation
graph.remove_nodes_from(list(nx.isolates(graph)))
print("Total nodes: ", graph.number_of_nodes(), "Total edges: ", graph.number_of_edges())
options = {
'node_color': 'black',
'node_size': 10,
}  
nx.draw(graph, draw_graph_pos, **options)
plt.show()

# find subgraphs
subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
total_g=len(subgraphs)
print('Total subgraphs: ', total_g)
graph_count=1
for subg in subgraphs:    
    print("graph: ",graph_count,"/",total_g," nodes: ",subg.number_of_nodes(), " edges: ", subg.number_of_edges())
    connetion_path = nx.approximation.traveling_salesman_problem(subg, cycle=True)
    connetion_path = list(connetion_path)
    if subg.number_of_nodes()>2:
        connetion_path.append(connetion_path[0])
    
    connetion_path_arr = np.array(connetion_path)
    lam_pix = np.linalg.norm(np.diff(connetion_path_arr[:,:2],axis=0),axis=1).sum()
    if lam_pix<min_stroke_length:
        print("Stroke length is less than min_stroke_length")
        continue
    strokes_split.append(connetion_path_arr)
    
    graph_count+=1
print("End tsp...")

strokes = strokes_split

if save_paths:
    ## save to strokes to file
    Path('../path/pixel_path/'+img_name+'/').mkdir(parents=True, exist_ok=True)
    for i in range(len(strokes)):
        np.savetxt('../path/pixel_path/'+img_name+'/'+str(i)+'.csv', strokes[i], delimiter=',')
    ## save resized image
    cv2.imwrite('../imgs/'+img_name+'_resized.png', image_thresh)

image_out = np.ones_like(image_thresh_flip)*255
for stroke in strokes:
    for n in stroke:
        image_out = cv2.circle(image_out, (int(n[0]), int(n[1])), round(n[2]), 0, -1)
        cv2.imshow("Image", image_out)
        if cv2.waitKey(1) == ord('q'): 
            # press q to terminate the loop 
            cv2.destroyAllWindows() 
            break 