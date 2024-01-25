import numpy as np
import matplotlib.pyplot as plt
import sys, time
from general_robotics_toolbox import *
from copy import deepcopy
sys.path.append('../toolbox')
from robot_def import *
from lambda_calc import *
from motion_toolbox import *

robot=robot_obj('ABB_1200_5_90','../config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='../config/heh6_pen.csv')

for i in range(3):
    load_ft = np.loadtxt('ft_record_load_'+str(i)+'.csv',delimiter=',')
    move_ft = np.loadtxt('ft_record_move_'+str(i)+'.csv',delimiter=',')
    
    #### plot load force ####
    # find the start touch point index
    start_id = np.where(load_ft[:,1]<=-0.5)[0][0]
    load_ft = load_ft[start_id-100:,:]
    # subtract the start time
    load_ft[:,0] = load_ft[:,0]-load_ft[0,0]
    # plot in positive direction
    load_ft[:,1] = -load_ft[:,1]
    # reference force
    ref_ft = np.zeros(len(load_ft))
    ref_ft[100:] = (load_ft[100:,0]-load_ft[100,0])*1
    ref_ft[100:] = np.clip(ref_ft[100:],-2,2)
    
    #### plot move force ####
    # subtract the start time
    move_ft[:,0] = move_ft[:,0]-move_ft[0,0]
    move_ft[:,0] = move_ft[:,0]+load_ft[-1,0]
    # plot in positive direction
    move_ft[:,1] = -move_ft[:,1]
    # reference force
    ref_ft_move = np.zeros(len(move_ft))
    ref_ft_move[:] = 2
    
    #### plot everything together ####
    total_ft = np.append(load_ft[:,1],move_ft[:,1],axis=0)
    total_t = np.append(load_ft[:,0],move_ft[:,0],axis=0)
    ref_ft = np.append(ref_ft,ref_ft_move,axis=0)
    # draw all force
    plt.plot(total_t,total_ft,label='force')
    plt.plot(total_t,ref_ft,label='ref')
    plt.vlines(move_ft[0,0],max(total_ft)+0.5,min(total_ft)-0.5,colors='r',linestyles='dashed',label='start move')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('time (s)',fontsize=20)
    plt.ylabel('force (N)',fontsize=20)
    plt.title('total force. Stroke '+str(i),fontsize=24)
    plt.show()
    
    