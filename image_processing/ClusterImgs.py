import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import networkx as nx
import time

def check_new_pixel_covered(image_covered, image_circle, bpx, bpy, radias):
    
    y_len, x_len = image_covered.shape
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

def travel_pixel_skeletons():
    # TODO: add code to cluster the skeletons
    pass

def travel_pixel_dots(image,resize_ratio=2,max_radias=10,min_radias=2,max_nodes=1500,max_edges=2500,max_pixel=250000,
                      blank_thres=1.5,leave_paper_weight=2,face_mask=None,face_drawing_order=[],skip_background=False,SHOW_TSP=False):
    
    ## convert image to gray
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    shrink_flag = False
    dot_st=time.time()
    while True:
        ## shrink size
        if shrink_flag:
            resize_ratio = resize_ratio*0.95
            shrink_flag = False
        ## thresholding
        _, image_thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)

        # resize image
        image_thresh = cv2.resize(image_thresh, (int(image_thresh.shape[1]*resize_ratio), int(image_thresh.shape[0]*resize_ratio)), interpolation = cv2.INTER_NEAREST)
        if face_mask is not None:
            face_mask = cv2.resize(face_mask, (int(image_thresh.shape[1]*resize_ratio), int(image_thresh.shape[0]*resize_ratio)), interpolation = cv2.INTER_NEAREST)
        print("image size: ",image_thresh.shape) if SHOW_TSP else None

        ## find all black pixels
        black_pixels = np.where(image_thresh == 0)
        print("black pixels: ",len(black_pixels[0])) if SHOW_TSP else None
        if len(black_pixels[0]) > max_pixel:
            print("skip... too many pixels", len(black_pixels[0]))
            shrink_flag = True
            continue

        ##### covered the image with circles ####
        ## create image_covered, image_vis, image_node
        image_covered = deepcopy(image_thresh)
        image_node = np.zeros_like(image_thresh)
        # nodes = []
        nodes = {0:[]}
        if face_mask is not None or len(face_drawing_order) > 0:
            for i in range(len(face_drawing_order)):
                nodes[i] = []
        graph = nx.Graph()
        draw_graph_pos={}
        black_pixels_not_drew = deepcopy(black_pixels)
        ## loop through radias max_radias to 1
        for radias in range(max_radias,min_radias-1,-1):
            print("radias: ",radias,"/",max_radias) if SHOW_TSP else None
            image_thresh_not=np.logical_not(image_thresh)
            dist_img = cv2.distanceTransform(image_thresh_not.astype(np.uint8), cv2.DIST_L2, 3)
            ## loop through all black pixels
            st=time.time()
            black_pixels_not_drew_next = [[],[]]
            for i in range(len(black_pixels_not_drew[0])):
                ## get x and y of black pixel
                x = black_pixels[1][i]
                y = black_pixels[0][i]
                
                if i%10000==0:
                    print("time pass",time.time()-st,i,"/",len(black_pixels_not_drew[0])," pixels") if SHOW_TSP else None
                    
                if dist_img[y,x] < radias-2:
                    black_pixels_not_drew_next[0].append(y)
                    black_pixels_not_drew_next[1].append(x)
                    continue
                
                image_white = np.ones_like(image_thresh)
                image_circle = cv2.circle(image_white, (x, y), radias, 0, -1)
                # check if circle is valid
                # valid_circle = check_circle_valid(image_thresh, image_circle, x, y, radias)
                valid_circle = check_circle_valid(image_covered, image_circle, x, y, radias)
                if not valid_circle:
                    black_pixels_not_drew_next[0].append(y)
                    black_pixels_not_drew_next[1].append(x)
                    continue
                ## check if new pixel covered
                covered_pixel = check_new_pixel_covered(image_covered, image_circle, x, y, radias)
                if not covered_pixel:
                    black_pixels_not_drew_next[0].append(y)
                    black_pixels_not_drew_next[1].append(x)
                    continue
                
                ## image_covered at x y has value of radias
                image_circle = np.logical_not(image_circle)
                image_covered = np.logical_or(image_covered, image_circle)
                
                ## add node
                image_node[y,x] = radias
                if face_mask is not None or len(face_drawing_order) > 0:
                    face_seg = face_mask[y,x]
                    nodes[face_seg].append([x,y,radias])
                else:
                    nodes[0].append([x,y,radias])
                graph.add_node((x,y),width=radias) ## add circle center pixel as nodes
                draw_graph_pos[(x,y)]=(x,y)
            black_pixels_not_drew = black_pixels_not_drew_next
        
        print("dotting time pass",time.time()-dot_st) if SHOW_TSP else None
        
        tsp_st=time.time()
        if 0 not in face_drawing_order:
            face_drawing_order.append(0)
        
        for face_seg in face_drawing_order:
            for i in range(len(nodes[face_seg])):
                for j in range(i+1,len(nodes[face_seg])):
                    dist = np.sqrt((nodes[face_seg][i][0]-nodes[face_seg][j][0])**2 + (nodes[face_seg][i][1]-nodes[face_seg][j][1])**2)
                    if dist < nodes[face_seg][i][2]+nodes[face_seg][j][2]+blank_thres:
                        # graph.add_node(((nodes[face_seg][i][0]+nodes[face_seg][j][0])/2,(nodes[face_seg][i][1]+nodes[face_seg][j][1])/2))
                        # graph.add_edge((nodes[face_seg][i][0],nodes[face_seg][i][1]),((nodes[face_seg][i][0]+nodes[face_seg][j][0])/2,(nodes[face_seg][i][1]+nodes[face_seg][j][1])/2))
                        # graph.add_edge((nodes[face_seg][j][0],nodes[face_seg][j][1]),((nodes[face_seg][i][0]+nodes[face_seg][j][0])/2,(nodes[face_seg][i][1]+nodes[face_seg][j][1])/2))
                        graph.add_edge((nodes[face_seg][i][0],nodes[face_seg][i][1]),(nodes[face_seg][j][0],nodes[face_seg][j][1]),weight=dist)
                        image_node = cv2.line(image_node, (nodes[face_seg][i][0],nodes[face_seg][i][1]), (nodes[face_seg][j][0],nodes[face_seg][j][1]), max(nodes[face_seg][i][2],nodes[face_seg][j][2]), 1)
                    else:
                        # graph.add_edge((nodes[face_seg][i][0],nodes[face_seg][i][1]),(nodes[face_seg][j][0],nodes[face_seg][j][1]),weight=dist*leave_paper_weight)
                        pass

        ## check isolation
        graph.remove_nodes_from(list(nx.isolates(graph)))

        ## check connectivity
        all_graphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

        ## start tsp
        print("Start tsp...") if SHOW_TSP else None
        total_g=len(all_graphs)
        all_paths=[]
        count=1
        for subg in all_graphs:
            print("graph: ",count,"/",total_g," nodes: ",subg.number_of_nodes(), " edges: ", subg.number_of_edges()) if SHOW_TSP else None
            if subg.number_of_nodes() > max_nodes or subg.number_of_edges() > max_edges:
                print("skip... graph too large", subg.number_of_nodes(), subg.number_of_edges())
                shrink_flag = True
                continue
            draw_path = nx.approximation.traveling_salesman_problem(subg, cycle=False)
            all_paths.append(draw_path)
            count+=1
        print("End tsp...") if SHOW_TSP else None
        print("tsp time pass",time.time()-tsp_st) if SHOW_TSP else None
        ## if not shrink, then break
        if not shrink_flag:
            break

    strokes_split = []
    for i in range(len(all_paths)):
        stroke=[]
        for j in range(len(all_paths[i])):
            width = graph.nodes[all_paths[i][j]]['width']
            this_p = list(all_paths[i][j])
            this_p.append(width)
            stroke.append(this_p)
        strokes_split.append(np.array(stroke))
    strokes=strokes_split
    
    return strokes, image_thresh

def main():
    
    save_paths=False

    # Read image
    img_name='me_out'
    # image_path = Path("../imgs/"+img_name+".png")
    image_path = Path("../temp_data/img_out.jpg")
    image = cv2.imread(str(image_path))
    print("image shape: ",image.shape)

    # Get pixel paths
    st = time.time()
    strokes, image_thresh = travel_pixel_dots(image,resize_ratio=2,max_radias=10,min_radias=2,SHOW_TSP=True)
    print("time: ",time.time()-st)

    if save_paths:
        ## save to strokes to file
        Path('../path/pixel_path/'+img_name+'/').mkdir(parents=True, exist_ok=True)
        for i in range(len(strokes)):
            np.savetxt('../path/pixel_path/'+img_name+'/'+str(i)+'.csv', strokes[i], delimiter=',')
        ## save resized image
        cv2.imwrite('../imgs/'+img_name+'_resized.png', image_thresh)

if __name__ == "__main__":
    main()