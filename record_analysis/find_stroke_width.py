import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
sys.path.append('../search_algorithm')
sys,sys.path.append('../toolbox')
from dfs import DFS
from robot_def import *
from lambda_calc import *
from utils import *
from scipy.interpolate import interp1d

data_pixel_w = []
data_force = []
data_lam_robot = []
data_lam_stroke = []
for data_dir in ['./record_0131_1/','./record_0131_2/']:
    # Load the image
    image_path = data_dir+'picture.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Perform skeletonization using cv2.ximgproc.thinning
    _, image_thresh = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    image_flip = np.bitwise_not(image_thresh)
    thinned_image = cv2.ximgproc.thinning(image_flip)

    # find the distance closest black pixel in image_thresh using distance transform
    dist_transform = cv2.distanceTransform(image_flip, cv2.DIST_L2, 5)

    # find width of the strokes
    skele_pixels = np.where(thinned_image == 255)
    edge_count = 0
    edges = []
    for i in range(len(skele_pixels[0])):
        x = skele_pixels[1][i]
        y = skele_pixels[0][i]
        white_n = np.sum(thinned_image[y-1:y+2, x-1:x+2] > 0)
        if white_n==2:
            edge_count+=1
            edges.append([x, y])
    edges = np.array(edges)
    # find three edges with smallest x value
    sorted_edges = edges[np.argsort(edges[:,0])]
    sorted_edges = sorted_edges[:3]
    edges = sorted_edges
    edge_count = len(edges)
    ## find strokes with deep first search, starting from the edge pixels
    dfs = DFS(thinned_image, edges)
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

    ## read force from ft_record_move
    # defined robot
    robot=robot_obj('ABB_1200_5_90','../config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='../config/heh6_pen.csv')

    for i in range(3):
        
        # read data
        ft_record_move = np.loadtxt(data_dir+'ft_record_move_'+str(i)+'.csv', delimiter=',')
        
        # force and joint trajectory
        path_f = -1*ft_record_move[:,1]
        path_q = np.radians(ft_record_move[:,2:])
        
        # moving average filter
        path_f = moving_average(path_f, n=31, padding=True)

        # get path length
        lam_robot = calc_lam_js(path_q,robot)
        lam_stroke = calc_lam_cs(strokes[i][:,:2])
        data_lam_robot.append(lam_robot[-1])
        data_lam_stroke.append(lam_stroke[-1])
        
        # interpolate force to the same length as the stroke
        path_f_interp = np.interp(lam_stroke/lam_stroke[-1],lam_robot/lam_robot[-1],path_f)
        
        data_pixel_w.extend(strokes[i][:,2])
        data_force.extend(path_f_interp)

    data_x = np.vstack((data_pixel_w,np.ones(len(data_pixel_w))))
    data_y = np.array(data_force)
    A = data_y@np.linalg.pinv(data_x)

## save parameters
np.savetxt('../config/pixel2force.csv',A,delimiter=',')
print(data_lam_robot)
print(data_lam_stroke)
np.savetxt('../config/pixel2mm.csv',[np.mean(np.divide(data_lam_robot,data_lam_stroke))])

plt.scatter(data_pixel_w,data_force,label='pixel width vs force')
plt.plot([min(data_pixel_w), max(data_pixel_w)], [A[0]*min(data_pixel_w)+A[1], A[0]*max(data_pixel_w)+A[1]], color='red',label='y='+str(A[0])+'x+'+str(A[1]))
plt.xlabel('Pixel width')
plt.title('Pixel width vs force')
plt.show()