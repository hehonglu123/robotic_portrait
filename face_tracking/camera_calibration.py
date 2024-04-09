
# Import required modules 
import cv2 
import numpy as np 
import os 
import glob 
import sys
from RobotRaconteur.Client import *     #import RR client library
import time

#RR client setup, connect to turtle service
url='rr+tcp://localhost:52222/?service=Face_tracking'
#take url from command line
if (len(sys.argv)>=2):
        url=sys.argv[1]

###2 modes available, choose either one		
########wire connection mode:
# obj=RRN.ConnectService(url)
# turtle_change=obj.turtle_change.Connect()


########subscription mode
def connect_failed(s, client_id, url, err):
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))

face_tracking_sub=RRN.SubscribeService(url)
obj = face_tracking_sub.GetDefaultClientWait(30)		#connect, timeout=30s
bbox_wire=face_tracking_sub.SubscribeWire("bbox")
image_wire=face_tracking_sub.SubscribeWire("frame_stream")

face_tracking_sub.ClientConnectFailed += connect_failed
  
# Define the dimensions of checkerboard 
CHECKERBOARD = (5, 7) 
  
  
# stop the iteration when specified 
# accuracy, epsilon, is reached or 
# specified number of iterations are completed. 
criteria = (cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
  
  
# Vector for 3D points 
threedpoints = [] 
  
# Vector for 2D points 
twodpoints = [] 
  
  
#  3D points real world coordinates 
objectp3d = np.zeros((1, CHECKERBOARD[0]  
                      * CHECKERBOARD[1],  
                      3), np.float32) 
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
                               0:CHECKERBOARD[1]].T.reshape(-1, 2) 
prev_img_shape = None
  
  
# Extracting path of individual image stored 
# in a given directory. Since no path is 
# specified, it will take current directory 
# jpg files alone 
images = glob.glob('*.jpg') 

count=0
while True:
    RR_image = image_wire.TryGetInValue()
    if RR_image[0]:
        image=RR_image[1]
        image=np.array(image.data,dtype=np.uint8).reshape((image.image_info.height,image.image_info.width,3))
    else:
        continue
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
    # Find the chess board corners 
    # If desired number of corners are 
    # found in the image then ret = true 
    ret, corners = cv2.findChessboardCorners( 
                    grayColor, CHECKERBOARD,  
                    cv2.CALIB_CB_ADAPTIVE_THRESH  
                    + cv2.CALIB_CB_FAST_CHECK + 
                    cv2.CALIB_CB_NORMALIZE_IMAGE) 
  
    # If desired number of corners can be detected then, 
    # refine the pixel coordinates and display 
    # them on the images of checker board 
    if ret == True: 
        threedpoints.append(objectp3d) 
  
        # Refining pixel coordinates 
        # for given 2d points. 
        corners2 = cv2.cornerSubPix( 
            grayColor, corners, (11, 11), (-1, -1), criteria) 
  
        twodpoints.append(corners2) 
  
        # Draw and display the corners 
        image = cv2.drawChessboardCorners(image,  
                                          CHECKERBOARD,  
                                          corners2, ret) 
        
        count+=1
        print(count)
  
    cv2.imshow('img', image) 
    cv2.waitKey(0) 

    time.sleep(0.1)

    if count>20:
        break
  
cv2.destroyAllWindows() 
  
h, w = image.shape[:2] 
  
  
# Perform camera calibration by 
# passing the value of above found out 3D points (threedpoints) 
# and its corresponding pixel coordinates of the 
# detected corners (twodpoints) 
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
    threedpoints, twodpoints, grayColor.shape[::-1], None, None) 
  
  
# Displaying required output 
print(" Camera matrix:") 
print(matrix) 
  
print("\n Distortion coefficient:") 
print(distortion) 
  
print("\n Rotation Vectors:") 
print(r_vecs) 
  
print("\n Translation Vectors:") 
print(t_vecs) 