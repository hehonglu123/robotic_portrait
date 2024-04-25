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

def travel_pixel_skeletons(image,resize_ratio=2,max_radias=10,min_radias=2,min_stroke_length=100,min_white_pixels=50,
                      face_mask=None,face_drawing_order=[],skip_background=False,SHOW_TSP=False):
    
    skeleton_st = time.time()
    # rename face_mask
    if face_mask is not None:
        face_seg_dict = {}
        for face_seg in face_drawing_order:
            if type(face_seg) == tuple:
                for face_seg_i in face_seg:
                    if face_seg_i not in face_seg_dict:
                        face_seg_dict[face_seg_i] = [face_seg]
                    else:
                        face_seg_dict[face_seg_i].append(face_seg)
            else:
                if face_seg not in face_seg_dict:
                    face_seg_dict[face_seg] = [face_seg]
                else:
                    face_seg_dict[face_seg].append(face_seg)
    #### image preprocessing
    # convert image to gray
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresholding
    # _, image = cv2.threshold(image_gray, 15, 255, cv2.THRESH_BINARY)
    _, image = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY)
    image = cv2.resize(image, (int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)), interpolation = cv2.INTER_NEAREST)
    if face_mask is not None:
        face_mask = cv2.resize(face_mask, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_NEAREST)
    print("image size: ",image.shape) if SHOW_TSP else None
    print("face mask size: ",face_mask.shape) if SHOW_TSP and face_mask is not None else None
    image_filled = deepcopy(image)
    image_vis = deepcopy(image)*0.9

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
            # if round(dist_transform[y,x])>=min_radias:
            if round(dist_transform[y,x])>=2:
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
        
        # visualize skeleton
        # image_skele_circle = deepcopy(image_skeleton)
        # for i in range(len(white_pixels[0])):
        #     x = white_pixels[1][i]
        #     y = white_pixels[0][i]
        #     image_skele_circle = cv2.circle(image_skele_circle, (x, y), 2, 255, -1)
        # plt.imshow(np.clip(image_vis+image_skele_circle,0,255).astype(int), cmap='gray')
        # plt.show()
        
        
        if len(white_pixels)==0:
            print("Number of white pixels is less than min_white_pixels") if SHOW_TSP else None
            break
        print(f"Number of white pixels: {len(white_pixels[0])}")
        if len(white_pixels[0])<min_white_pixels:
            print("Number of white pixels is less than min_white_pixels") if SHOW_TSP else None
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
                
        print(f"Number of white pixels with 2 white neighbors: {edge_count}") if SHOW_TSP else None
        
        remove_factor = 1

        ## find min max stroke width
        max_dist = max(dist_transform[white_pixels])
        min_dist = min(dist_transform[white_pixels])
        print(f"Max distance: {max_dist}, Min distance: {min_dist}") if SHOW_TSP else None

        ##### draw circle and find nodes and edges
        out_graph_pix = np.array([white_pixels[1],white_pixels[0]]).T
        start_draw = True
        while out_graph_pix.shape[0]>0:
            # move to in graph
            if len(in_graph_pix)==0:
            # if start_draw:
                start_draw = False
                # add first node
                graph.add_node(tuple(np.append(edges[0],min(max(dist_transform[edges[0][1]][edges[0][0]],1),max_radias))))
                in_graph_pix = np.array([edges[0]]) # add the in graph pixel liset
                out_graph_pix = np.delete(out_graph_pix,np.all(out_graph_pix==edges[0],axis=1),axis=0) # remove from outside graph list
            else:
                closest_i = np.argmin(np.linalg.norm(out_graph_pix-in_graph_pix[-1],axis=1)) # find the closest pixel to the "draw nodes"
                in_graph_pix = np.vstack((in_graph_pix,out_graph_pix[closest_i])) # add the in graph pixel liset
                out_graph_pix = np.delete(out_graph_pix,closest_i,axis=0) # remove from outside graph list
            
            ## get draw radius
            draw_radius = dist_transform[in_graph_pix[-1][1]][in_graph_pix[-1][0]]
            if draw_radius > max_radias:
                draw_radius = min(max_radias,draw_radius/1.)
            if draw_radius < min_radias:
                draw_radius = min_radias
            # draw_radius = min(max(dist_transform[in_graph_pix[-1][1]][in_graph_pix[-1][0]],1),max_radias) # drawing radius
            
            this_node_name = tuple(np.append(in_graph_pix[-1],draw_radius)) # node name
            in_graph_radius = np.append(in_graph_radius,draw_radius) # add the in graph radius list
            draw_graph_pos[this_node_name]=tuple(in_graph_pix[-1]) # add the drawing position
            # remove all the pixels that are too close to the in graph
            close_i = np.where(np.linalg.norm(out_graph_pix-in_graph_pix[-1],axis=1)<=in_graph_radius[-1]*remove_factor)[0]
            out_graph_pix = np.delete(out_graph_pix,close_i,axis=0)
            # draw the circle
            image_filled = cv2.circle(image_filled, in_graph_pix[-1], min(max(round(draw_radius),1),max_radias), 255, -1)
            image_vis = cv2.circle(image_vis, in_graph_pix[-1], min(max(round(draw_radius),1),max_radias), 120, -1)
            
            if len(in_graph_pix)==1:
                continue
            # draw edge if there's one
            edges_node_ids = np.where(np.linalg.norm(in_graph_pix[:-1]-in_graph_pix[-1],axis=1)<=in_graph_radius[:-1]+in_graph_radius[-1])[0]
            
            for eni in edges_node_ids:
                if face_mask is not None:
                    group1 = face_seg_dict[face_mask[in_graph_pix[eni][1],in_graph_pix[eni][0]]]
                    group2 = face_seg_dict[face_mask[in_graph_pix[-1][1],in_graph_pix[-1][0]]]
                    for g1 in group1:
                        for g2 in group2:
                            if g1 == g2:
                                # graph.add_edge(tuple(np.append(in_graph_pix[eni],in_graph_radius[eni])),this_node_name\
                                #     ,weight=np.linalg.norm(np.append(in_graph_pix[eni]-in_graph_pix[-1],in_graph_radius[eni]-in_graph_radius[-1])))
                                graph.add_edge(tuple(np.append(in_graph_pix[eni],in_graph_radius[eni])),this_node_name\
                                    ,weight=np.linalg.norm(in_graph_pix[eni]-in_graph_pix[-1]))
                else:
                    graph.add_edge(tuple(np.append(in_graph_pix[eni],in_graph_radius[eni])),this_node_name\
                        ,weight=np.linalg.norm(np.append(in_graph_pix[eni]-in_graph_pix[-1],in_graph_radius[eni]-in_graph_radius[-1])))
                    # graph.add_edge(tuple(np.append(in_graph_pix[eni],in_graph_radius[eni])),this_node_name\
                    #     ,weight=np.linalg.norm(in_graph_pix[eni]-in_graph_pix[-1]))
            
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
            
        # find strokes with deep first search, starting from the edge pixels
        # plt.imshow(image_filled, cmap='gray')
        # plt.show()
        # plt.imshow(image_vis, cmap='gray')
        # plt.show()
        
        count+=1

    print("skeleton time pass",time.time()-skeleton_st) if SHOW_TSP else None

    tsp_st=time.time()
    ## check isolation
    graph.remove_nodes_from(list(nx.isolates(graph)))
    print("Total nodes: ", graph.number_of_nodes(), "Total edges: ", graph.number_of_edges()) if SHOW_TSP else None
    options = {
    'node_color': 'black',
    'node_size': 10,
    }  
    # nx.draw(graph, draw_graph_pos, **options)
    # plt.show()

    # find subgraphs
    subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    total_g=len(subgraphs)

    ### check all subgraphs are small enough
    while True:
        subgraphs_split = []
        small_flag = True
        for subg in subgraphs:
            if subg.number_of_nodes() > 2000 or subg.number_of_edges() > 2000:
                print("Subgraph is too large") if SHOW_TSP else None
                small_flag = False

                node_x = [node[0] for node in subg.nodes]
                
                max_x = max(node_x)
                min_x = min(node_x)
                split_x = (max_x+min_x)/2
                print("Split x: ",split_x) if SHOW_TSP else None
                print('nodes: ',subg.number_of_nodes(),'edges: ',subg.number_of_edges()) if SHOW_TSP else None
                for edge in subg.edges:
                    if edge[0][0] > split_x and edge[1][0] <= split_x:
                        subg.remove_edge(*edge)
                    elif edge[1][0] > split_x and edge[0][0] <= split_x:
                        subg.remove_edge(*edge)
                subg.remove_nodes_from(list(nx.isolates(subg)))
                subsubgraphs = [subg.subgraph(c).copy() for c in nx.connected_components(subg)]
                subgraphs_split.extend(subsubgraphs)
            else:
                subgraphs_split.append(subg)
        subgraphs = subgraphs_split
        if small_flag:
            break
    
    print('Total subgraphs: ', total_g) if SHOW_TSP else None
    graph_count=1
    for subg in subgraphs:    
        print("graph: ",graph_count,"/",total_g," nodes: ",subg.number_of_nodes(), " edges: ", subg.number_of_edges()) if SHOW_TSP else None
        if subg.number_of_nodes()<2 or subg.number_of_edges()<2:
            print("Skip subgraph with less than 2 nodes") if SHOW_TSP else None
            continue

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
        # nx.draw(subg, draw_graph_pos, **options)
        # plt.show()
        
        graph_count+=1
    print("Total strokes: ", len(strokes_split)) if SHOW_TSP else None
    print("End tsp...") if SHOW_TSP else None
    print("tsp time pass",time.time()-tsp_st) if SHOW_TSP else None
    
    strokes = strokes_split
    
    return strokes, image

def travel_pixel_dots(image,resize_ratio=2,max_radias=10,min_radias=2,max_nodes=1500,max_edges=2500,max_pixel=250000,
                      blank_thres=1.5,leave_paper_weight=2,face_mask=None,face_drawing_order=[],skip_background=False,SHOW_TSP=False):
    
    # rename face_mask
    face_seg_dict = {}
    for face_seg in face_drawing_order:
        if type(face_seg) == tuple:
            for face_seg_i in face_seg:
                if face_seg_i not in face_seg_dict:
                    face_seg_dict[face_seg_i] = [face_seg]
                else:
                    face_seg_dict[face_seg_i].append(face_seg)
        else:
            if face_seg not in face_seg_dict:
                face_seg_dict[face_seg] = [face_seg]
            else:
                face_seg_dict[face_seg].append(face_seg)
    
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
        list_patches = []
        if face_mask is not None or len(face_drawing_order) > 0:
            for patch in face_drawing_order:
                nodes[patch] = []
                if type(patch) == tuple:
                    list_patches.append(tuple(patch))
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
                if face_mask is not None and len(face_drawing_order) > 0:
                    face_seg = face_mask[y,x]
                    for list_patch in list_patches:
                        if face_seg in list_patch:
                            face_seg = list_patch
                            break
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
    # image_path = Path("../temp_data/img_out.jpg")
    image_path = Path("../imgs/logos.png")
    image = cv2.imread(str(image_path))
    print("image shape: ",image.shape)

    # Get pixel paths
    st = time.time()
    # strokes, image_thresh = travel_pixel_dots(image,resize_ratio=2,max_radias=10,min_radias=2,SHOW_TSP=True)
    strokes, image_thresh = travel_pixel_skeletons(image,resize_ratio=0.5,max_radias=10,min_radias=2,SHOW_TSP=True)
    print("time: ",time.time()-st)
    
    image_out = np.ones_like(image_thresh)*255
    for stroke in strokes:
        for n in stroke:
            image_out = cv2.circle(image_out, (int(n[0]), int(n[1])), round(n[2]), 0, -1)
            image_out[int(n[1]),int(n[0])]=120
            cv2.imshow("Image", image_out)
            if cv2.waitKey(1) == ord('q'): 
                # press q to terminate the loop 
                cv2.destroyAllWindows() 
                break 
        input("Next stroke? (Press Enter)")

    if save_paths:
        ## save to strokes to file
        Path('../path/pixel_path/'+img_name+'/').mkdir(parents=True, exist_ok=True)
        for i in range(len(strokes)):
            np.savetxt('../path/pixel_path/'+img_name+'/'+str(i)+'.csv', strokes[i], delimiter=',')
        ## save resized image
        cv2.imwrite('../imgs/'+img_name+'_resized.png', image_thresh)

if __name__ == "__main__":
    main()