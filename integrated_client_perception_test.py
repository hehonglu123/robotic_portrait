from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys, cv2
import numpy as np
import glob
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
# test_img_path = 'imgs/logo.png'
test_logo = 'logos_words'
#########################################################EXECUTION#########################################################
while True:
    # start_time=time.time()
    
    # ### simulating image capture
    # print('IMAGE TAKEN')
    # img = cv2.imread(test_img_path)
    
    # img_st = time.time()
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # ############################################################

    # ########################## portrait FaceSegmentation/GAN ##############################
    # ## Face Segmentation
    # gray_image_masked,face_parse_mask,face_mask,faces = faceseg.get_face_mask(img)
    # anime_img = anime.forward(gray_image_masked)

    # # print(gray_image_masked.shape)    
    # # print(anime_img.shape)
    # # anime_img_viz = facer.hwc2bchw(torch.from_numpy(anime_img)).to(device='cuda') 
    # # facer.show_bchw(facer.draw_bchw(anime_img_viz, faces))
    # plt.matshow(face_parse_mask)
    # plt.show()
    
    # img_gray=cv2.cvtColor(anime_img, cv2.COLOR_BGR2GRAY)

    # # cv2.imshow('img',anime_img)
    # # cv2.waitKey(0)
    # print("IMAGE PROCESSING TIME: ", time.time()-img_st)
    # ####################################################################
    
    # ####################################PLANNING#####################################################
    # planning_st = time.time()
    # ###Pixel Traversal
    # print('TRAVERSING PIXELS')
    
    # resize_ratio=np.max(np.divide(target_size,anime_img.shape[:2]))
    
    # # face_drawing_order=[10,1,6,(7,8,9),2,3,4,5,0] # hair, face, nose, upper lip, teeth, lower lip, left eyebrow, right eyebrow, left eye, right eye
    # # pixel_paths, image_thresh = travel_pixel_dots(anime_img,resize_ratio=resize_ratio,max_radias=10,min_radias=2,face_mask=face_parse_mask,face_drawing_order=face_drawing_order,SHOW_TSP=True)
    # face_drawing_order=[10,1,(6,1),(7,8,9),(2,1),(3,1),(4,1),(5,1),0] # hair, face, nose, upper lip, teeth, lower lip, left eyebrow, right eyebrow, left eye, right eye
    # pixel_paths, image_thresh = travel_pixel_skeletons(anime_img,resize_ratio=resize_ratio,max_radias=10,min_radias=2,face_mask=face_parse_mask,face_drawing_order=face_drawing_order,SHOW_TSP=True)
    
    image_thresh = np.zeros((1244,800))*255
    print(image_thresh.shape)
    # print("travel_pixel_dots time: ", time.time()-planning_st)
    
    # # # plot force profile
    # # path_idx = 0
    # # ave_dfdlam_std = []
    # # for pixel_path in pixel_paths:
    # #     lam = calc_lam_cs(pixel_path[:,:2])
    # #     dfdlam = np.gradient(pixel_path[:-1,2])/np.gradient(lam[:-1])
    # #     print("path_idx %d Mean dfdlam: %f, Std dfdlam: %f"%(path_idx, np.mean(dfdlam), np.std(dfdlam)))
    # #     ave_dfdlam_std.append(np.std(dfdlam))
    # #     # plt.plot(lam,pixel_path[:,2])
    # #     # plt.show()
    # # print("Average dfdlam std: %f"%(np.mean(ave_dfdlam_std)))
    
    # print("Image size: ", image_thresh.shape)
    # ###Project to IPAD
    # project_st = time.time()
    # print("PROJECTING TO IPAD")
    # _,cartesian_paths_world,force_paths=image2plane(image_thresh,ipad_pose,pixel2mm,pixel_paths,pixel2force)
    # print("image2plane time: ", time.time()-project_st)
    # ###Solve Joint Trajectory
    # ik_st = time.time()
    # print("SOLVING JOINT TRAJECTORY")
    # js_paths=[]
    # for cartesian_path in cartesian_paths_world:
    #     curve_js=robot.find_curve_js(cartesian_path,[R_pencil]*len(cartesian_path),q_seed)
    #     js_paths.append(curve_js)
    # print("find_curve_js time: ", time.time()-ik_st)
    # print("PLANNING TIME: ", time.time()-planning_st)

    # ####################################################################
    # print('TOTAL TIME: ', time.time()-img_st)
    
    # image_out = np.ones_like(image_thresh)*255
    # for stroke in pixel_paths:
    #     for n in stroke:
    #         image_out = cv2.circle(image_out, (int(n[0]), int(n[1])), round(n[2]), 0, -1)
    #         image_out[int(n[1]),int(n[0])]=120
    #         cv2.imshow("Image", image_out)
    #         if cv2.waitKey(1) == ord('q'): 
    #             # press q to terminate the loop 
    #             cv2.destroyAllWindows() 
    #             break 
    #     input("Next stroke? (Press Enter)")
    
    ####### write words
    num_segments=len(glob.glob('path/pixel_path/'+test_logo+'/*.csv'))
    img=cv2.imread('imgs/'+test_logo+'_resized.png')
    # replocate the image
    img_new = np.ones((image_thresh.shape[0]+2*img.shape[0],image_thresh.shape[1]+2*img.shape[1],3))*255
    pixel_offset = np.array([image_thresh.shape[1]+img.shape[1],0])
    img_new[pixel_offset[1]:img.shape[0]+pixel_offset[1],pixel_offset[0]:pixel_offset[0]+img.shape[1]]=img
    img = img_new
    # relocate paths
    pixel_paths=[]
    for i in range(num_segments):
        pixel_paths.append(np.loadtxt('path/pixel_path/'+test_logo+'/%i.csv'%i,delimiter=',').reshape((-1,3)))
        pixel_paths[-1][:,:2]+=pixel_offset
    
    print("PROJECTING TO IPAD")
    _,cartesian_paths_world,force_paths=image2plane(img,ipad_pose,pixel2mm,pixel_paths,pixel2force)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(cartesian_paths_world)):
        ###plot out the path in 3D
        ax.plot(cartesian_paths_world[i][:,0], cartesian_paths_world[i][:,1], cartesian_paths_world[i][:,2], 'b')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    
    print("SOLVING JOINT TRAJECTORY")
    js_paths=[]
    for cartesian_path in cartesian_paths_world:
        curve_js=robot.find_curve_js(cartesian_path,[R_pencil]*len(cartesian_path),q_seed)
        js_paths.append(curve_js)
        
    image_out = np.ones_like(img)*255
    for stroke in pixel_paths:
        for n in stroke:
            image_out = cv2.circle(image_out, (int(n[0]), int(n[1])), round(n[2]), 0, -1)
            image_out[int(n[1]),int(n[0])]=120
            cv2.imshow("Image", image_out)
            if cv2.waitKey(1) == ord('q'): 
                # press q to terminate the loop 
                cv2.destroyAllWindows() 
                break 
    
    input("Next round? (Press Enter)")