import numpy as np
import matplotlib.pyplot as plt
import glob, sys, time
from scipy.interpolate import CubicSpline
from general_robotics_toolbox import *
from copy import deepcopy
sys.path.append('../toolbox')
from robot_def import *
from lambda_calc import *
from motion_toolbox import *

# img_name='wen_out'
# img_name='strokes_out'
img_name='eric_name_out'

ipad_pose=np.loadtxt('../config/ipad_pose.csv',delimiter=',')
num_segments=len(glob.glob('cartesian_path/'+img_name+'/*.csv'))
robot=robot_obj('ABB_1200_5_90','../config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='../config/heh6_pen.csv')
# robot=robot_obj('ur5','config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')
# print(robot.fwd(np.array([0,0,0,0,0,0])))
# exit()

######## FT sensor info
H_pentip2ati=np.loadtxt('../config/pentip2ati.csv',delimiter=',')

######## Controller parameters ###
controller_params = {
    "force_ctrl_damping": 40.0,
    "force_epsilon": 0.1, # Unit: N
    "moveL_speed_lin": 5.0, # Unit: mm/sec
    "moveL_acc_lin": 5, # Unit: mm/sec^2
    "moveL_speed_ang": np.radians(10), # Unit: rad/sec
    "trapzoid_slope": 1, # trapzoidal load profile. Unit: N/sec
    "load_speed": 10.0, # Unit mm/sec
    "unload_speed": 1.0, # Unit mm/sec
    'settling_time': 1 # Unit: sec
    }

for i in range(num_segments):
    print('Segment %i'%i)
    pixel_path=np.loadtxt('pixel_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
    cartesian_path=np.loadtxt('cartesian_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
    curve_js=np.loadtxt('js_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,6))
    force_path=np.loadtxt('force_path/'+img_name+'/%i.csv'%i,delimiter=',')
    
    # get path length
    lam = calc_lam_js(curve_js,robot)

    # find the time for each segment, with acceleration and deceleration
    time_bp = np.zeros_like(lam)
    acc = controller_params['moveL_acc_lin']
    vel = 0
    for i in range(0,len(lam)):
        if vel>=controller_params['moveL_speed_lin']:
            time_bp[i] = time_bp[i-1]+(lam[i]-lam[i-1])/controller_params['moveL_speed_lin']
        else:
            time_bp[i] = np.sqrt(2*lam[i]/acc)
            vel = acc*time_bp[i]
            print(vel)

    time_bp_half = []
    vel = 0
    for i in range(len(lam)-1,-1,-1):
        if vel>=controller_params['moveL_speed_lin'] or i<=len(lam)/2:
            break
        else:
            time_bp_half.append(np.sqrt(2*(lam[-1]-lam[i])/acc))
            vel = acc*time_bp_half[-1]
    time_bp_half = np.array(time_bp_half)[::-1]
    time_bp_half = time_bp_half*-1+time_bp_half[0]
    time_bp[-len(time_bp_half):] = time_bp[-len(time_bp_half)-1]+time_bp_half\
        +(lam[-len(time_bp_half)]-lam[-len(time_bp_half)-1])/controller_params['moveL_speed_lin']