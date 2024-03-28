import time, traceback, sys, cv2
import numpy as np
from general_robotics_toolbox import *
from matplotlib import pyplot as plt
sys.path.append('../toolbox')
from robot_def import *

ROBOT_NAME='ABB_1200_5_90'

if ROBOT_NAME=='ABB_1200_5_90':
    #########################################################config parameters#########################################################
    robot_cam=robot_obj(ROBOT_NAME,'ABB_1200_5_90_robot_default_config.yml',tool_file_path='camera.csv')
    robot=robot_obj(ROBOT_NAME,'ABB_1200_5_90_robot_default_config.yml',tool_file_path='heh6_pen.csv')
    radius=500 ###eef position to robot base distance w/o z height
    angle_range=np.array([-np.pi/2,-np.pi/4]) ###angle range of joint 1 for robot to move
    height_range=np.array([500,1500]) ###height range for robot to move
    p_tracking_start=np.array([ 107.2594, -196.3541,  859.7145])	###initial position
    R_tracking_start=np.array([[ 0.0326 , 0.8737 , 0.4854],
                            [ 0.0888,  0.4812, -0.8721],
                            [-0.9955 , 0.0715, -0.0619]])	###initial orientation
    q_seed=np.zeros(6)
    q_tracking_start=robot_cam.inv(p_tracking_start,R_tracking_start,q_seed)[0]	###initial joint position
    image_center=np.array([1080,1080])/2	###image center
    TIMESTEP=0.004
elif ROBOT_NAME=='ur5':
    #########################################################UR config parameters#########################################################
    robot_cam=robot_obj(ROBOT_NAME,'ur5_robot_default_config.yml',tool_file_path='camera_ur.csv')
    robot=robot_obj(ROBOT_NAME,'ur5_robot_default_config.yml',tool_file_path='heh6_pen_ur.csv')
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
    TIMESTEP=0.01

paper_size=np.loadtxt('paper_size.csv',delimiter=',') # size of the paper
pixel2mm=np.loadtxt('pixel2mm.csv',delimiter=',') # pixel to mm ratio
pixel2force=np.loadtxt('pixel2force.csv',delimiter=',') # pixel to force ratio
ipad_pose=np.loadtxt('ipad_pose.csv',delimiter=',') # ipad pose
H_pentip2ati=np.loadtxt('pentip2ati.csv',delimiter=',') # FT sensor info
p_button=np.array([131, -92, 0]) # button position, in ipad frame
R_pencil=ipad_pose[:3,:3]@Ry(np.pi) # pencil orientation, world frame
hover_height=20 # height to hover above the paper

print("zero configuration tool pose:",robot.fwd(np.zeros(6)))

##### check the reachable pose of the tablet
z_lower=300 # lower bound of ipad pose z (mm)
z_upper=1000 # upper bound of ipad pose z (mm)
x_lower=300 # lower bound of ipad pose x (mm)
x_upper=1200 # upper bound of ipad pose x (mm)
angle_lower=-np.radians(60) # lower bound of ipad pose angle (rad)
angle_upper=0 # upper bound of ipad pose angle (rad)
paper_size=paper_size*1.1 # x y dimension of the paper
paper_h_lower=-10 # lower bound of the paper height (mm)
paper_h_upper=250 # upper bound of the paper height (mm)
paper_dim_low = np.array([-paper_size[0]/2,-paper_size[1]/2,paper_h_lower])
paper_dim_high = np.array([paper_size[0]/2,paper_size[1]/2,paper_h_upper])

box_points = []
disc_N=5
for x in np.linspace(paper_dim_low[0],paper_dim_high[0],disc_N):
    for y in np.linspace(paper_dim_low[1],paper_dim_high[1],disc_N):
        for z in np.linspace(paper_dim_low[2],paper_dim_high[2],disc_N):
            box_points.append(np.array([x,y,z]))

### feasible conditions
j1_lower = -np.pi/2
j1_upper = np.pi/2
j2_upper = np.pi/2
j3_lower = -np.pi/2
j5_lower = 0
j_svd = 0.05

## run through all possible poses of the paper
plt_cnt = 0

feasible_xz_all = []
smallest_svd_all = []
for angle in np.linspace(angle_lower,angle_upper,5):
    print("angle:",np.degrees(angle))
    feasible_xz = []
    smallest_svd = []
    for z in np.linspace(z_lower,z_upper,10):
        for x in np.linspace(x_lower,x_upper,10):
            ipad_pose = Transform(rot(np.array([0,1,0]),angle),np.array([x,0,z]))
            
            non_feasible = False
            smallest_svd_collect=[]
            for bp in box_points:
                tool_pose = Transform(ipad_pose.R@Ry(np.pi),ipad_pose.p+ipad_pose.R@bp)
                # q = robot.inv(tool_pose.p,tool_pose.R,q_seed)
                try:
                    q = robot.inv(tool_pose.p,tool_pose.R,q_seed)[0]
                except:
                    non_feasible = True
                    break
                if q[0]<j1_lower or q[0]>j1_upper or q[1]>j2_upper or q[2]<j3_lower or q[4]<j5_lower:
                    non_feasible = True
                    break
                u,s,v=np.linalg.svd(robot.jacobian(q))
                smallest_J_svd = np.min(s)
                if smallest_J_svd<j_svd:
                    non_feasible = True
                    break
                smallest_svd_collect.append(smallest_J_svd)
            if non_feasible:
                continue
            feasible_xz.append([x,z])
            smallest_svd.append(np.mean(smallest_svd_collect))
    feasible_xz=np.array(feasible_xz)
    smallest_svd=np.array(smallest_svd)
    feasible_xz_all.append(feasible_xz)
    smallest_svd_all.append(smallest_svd)

vmin=np.min([np.min(smallest_svd) for smallest_svd in smallest_svd_all])
vmax=np.max([np.max(smallest_svd) for smallest_svd in smallest_svd_all])
xmin=np.min([np.min(feasible_xz[:,0]) for feasible_xz in feasible_xz_all])-100
xmax=np.max([np.max(feasible_xz[:,0]) for feasible_xz in feasible_xz_all])+100
zmin=np.min([np.min(feasible_xz[:,1]) for feasible_xz in feasible_xz_all])-100
zmax=np.max([np.max(feasible_xz[:,1]) for feasible_xz in feasible_xz_all])+100

suptitle_fontsize=24
title_fontsize=16
xylabel_fontsize=14
fig,axs = plt.subplots(2,3)
for feasible_xz,smallest_svd,angle in zip(feasible_xz_all,smallest_svd_all,np.linspace(angle_lower,angle_upper,5)):
    feasible_xz = np.array(feasible_xz)
    smallest_svd = np.array(smallest_svd)
    sc = axs[int(plt_cnt/3),int(plt_cnt%3)].scatter(feasible_xz[:,0],feasible_xz[:,1],c=smallest_svd,cmap='coolwarm', vmin=vmin, vmax=vmax)
    axs[int(plt_cnt/3),int(plt_cnt%3)].set_xlim(xmin,xmax)
    axs[int(plt_cnt/3),int(plt_cnt%3)].set_ylim(zmin,zmax)
    axs[int(plt_cnt/3),int(plt_cnt%3)].set_title("Rotated Angle: "+str(np.fabs(np.round(np.degrees(angle))).astype(int)),fontsize=title_fontsize)
    axs[int(plt_cnt/3),int(plt_cnt%3)].set_xlabel("x - w.r.t the robot base (mm)",fontsize=xylabel_fontsize)
    axs[int(plt_cnt/3),int(plt_cnt%3)].set_ylabel("z - w.r.t the robot base (mm)",fontsize=xylabel_fontsize)
    axs[int(plt_cnt/3),int(plt_cnt%3)].grid()
    axs[int(plt_cnt/3),int(plt_cnt%3)].set_xticks(np.arange(xmin,xmax+1,100))
    axs[int(plt_cnt/3),int(plt_cnt%3)].set_yticks(np.arange(zmin,zmax+1,100))
    axs[int(plt_cnt/3),int(plt_cnt%3)].set_xticklabels(np.round(axs[int(plt_cnt/3),int(plt_cnt%3)].get_xticks()).astype(int),fontsize=xylabel_fontsize)
    axs[int(plt_cnt/3),int(plt_cnt%3)].set_yticklabels(np.round(axs[int(plt_cnt/3),int(plt_cnt%3)].get_yticks()).astype(int),fontsize=xylabel_fontsize)
    plt_cnt+=1
fig.colorbar(sc, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
fig.suptitle("Tablet Reachable Pose (Color: Smallest Singular Value of Jacobian)",fontsize=suptitle_fontsize)
plt.show()