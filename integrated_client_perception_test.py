from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys, cv2
import numpy as np
sys.path.append('toolbox')
from robot_def import *
from lambda_calc import *
from motion_toolbox import *
from portrait import *
sys.path.append('image_processing')
from ClusterImgs import *
sys.path.append('motion_planning')
from PathGenCartesian import *
sys.path.append('robot_motion')
from RobotMotionController import *

ROBOT_NAME='ABB_1200_5_90' # ABB_1200_5_90 or ur5

if ROBOT_NAME=='ABB_1200_5_90':
    #########################################################config parameters#########################################################
    robot_cam=robot_obj(ROBOT_NAME,'config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/camera.csv')
    # robot=robot_obj(ROBOT_NAME,'config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen.csv')
    robot=robot_obj(ROBOT_NAME,'config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/brush_pen.csv')
    radius=500 ###eef position to robot base distance w/o z height
    # angle_range=np.array([-3*np.pi/4,-np.pi/4]) ###angle range of joint 1 for robot to move
    angle_range=np.array([-np.pi/2,0]) ###angle range of joint 1 for robot to move
    height_range=np.array([500,1500]) ###height range for robot to move
    # p_start=np.array([0,-radius,700])	###initial position
    # R_start=np.array([	[0,1,0],
    #                     [0,0,-1],
    #                     [-1,0,0]])	###initial orientation
    p_tracking_start=np.array([ 107.2594, -196.3541,  859.7145])	###initial position
    R_tracking_start=np.array([[ 0.0326 , 0.8737 , 0.4854],
                            [ 0.0888,  0.4812, -0.8721],
                            [-0.9955 , 0.0715, -0.0619]])	###initial orientation
    q_seed=np.zeros(6)
    q_tracking_start=robot_cam.inv(p_tracking_start,R_tracking_start,q_seed)[0]	###initial joint position
    image_center=np.array([1080,1080])/2	###image center
elif ROBOT_NAME=='ur5':
    #########################################################UR config parameters#########################################################
    robot_cam=robot_obj(ROBOT_NAME,'config/ur5_robot_default_config.yml',tool_file_path='config/camera_ur.csv')
    robot=robot_obj(ROBOT_NAME,'config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')
    radius=500 ###eef position to robot base distance w/o z height
    angle_range=np.array([-np.pi/4,np.pi/4]) ###angle range of joint 1 for robot to move
    height_range=np.array([500,900]) ###height range for robot to move
    p_tracking_start=np.array([-radius,0,750])	###initial position
    R_tracking_start=np.array([	[0,0,-1],
                        [0,-1,0],
                        [-1,0,0]])	###initial orientation
    q_seed=np.radians([0,-54.8,110,-142,-90,0])
    q_tracking_start=robot.inv(p_tracking_start,R_tracking_start,q_seed)[0]	###initial joint position
    image_center=np.array([1080,1080])/2	###image center
else:
    assert False, "ROBOT_NAME is not valid"

#########################################################config parameters#########################################################
paper_size=np.loadtxt('config/paper_size.csv',delimiter=',') # size of the paper
pixel2mm=np.loadtxt('config/pixel2mm.csv',delimiter=',') # pixel to mm ratio
pixel2force=np.loadtxt('config/pixel2force.csv',delimiter=',') # pixel to force ratio
ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',') # ipad pose
H_pentip2ati=np.loadtxt('config/pentip2ati.csv',delimiter=',') # FT sensor info
p_button=np.array([131, -92, 0]) # button position
R_pencil=ipad_pose[:3,:3]@Ry(np.pi) # pencil orientation
target_size=[1200,800]

### Portrait NNs ###
faceseg = FaceSegmentation()
anime = AnimeGANv3('models/AnimeGANv3_PortraitSketch.onnx')

# print(robot.fwd(mctrl.read_position()))
# exit()

test_img_path = 'temp_data/img1.jpg'
#########################################################EXECUTION#########################################################
while True:
    start_time=time.time()
    
    ### simulating image capture
    print('IMAGE TAKEN')
    img = cv2.imread(test_img_path)
    
    img_st = time.time()
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    ############################################################

    ########################## portrait FaceSegmentation/GAN ##############################
    ## Face Segmentation
    gray_image_masked,image_mask,face_mask = faceseg.get_face_mask(img)
    anime_img = anime.forward(gray_image_masked)
    img_gray=cv2.cvtColor(anime_img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('img',anime_img)
    # cv2.waitKey(0)
    print("IMAGE PROCESSING TIME: ", time.time()-img_st)
    ####################################################################
    
    ####################################PLANNING#####################################################
    planning_st = time.time()
    ###Pixel Traversal
    print('TRAVERSING PIXELS')
    face_drawing_order=[10,1,6,7,8,9,2,3,4,5,0]
    resize_ratio=np.max(np.divide(target_size,anime_img.shape[:2]))
    
    pixel_paths, image_thresh = travel_pixel_dots(anime_img,resize_ratio=resize_ratio,max_radias=10,min_radias=2,face_mask=image_mask,face_drawing_order=face_drawing_order,SHOW_TSP=True)
    print("travel_pixel_dots time: ", time.time()-planning_st)
    print("Image size: ", image_thresh.shape)
    ###Project to IPAD
    project_st = time.time()
    print("PROJECTING TO IPAD")
    _,cartesian_paths_world,force_paths=image2plane(image_thresh,ipad_pose,pixel2mm,pixel_paths,pixel2force)
    print("image2plane time: ", time.time()-project_st)
    ###Solve Joint Trajectory
    ik_st = time.time()
    print("SOLVING JOINT TRAJECTORY")
    js_paths=[]
    for cartesian_path in cartesian_paths_world:
        curve_js=robot.find_curve_js(cartesian_path,[R_pencil]*len(cartesian_path),q_seed)
        js_paths.append(curve_js)
    print("find_curve_js time: ", time.time()-ik_st)
    print("PLANNING TIME: ", time.time()-planning_st)

    ####################################################################
    print('TOTAL TIME: ', time.time()-img_st)
    
    input("Next round? (Press Enter)")