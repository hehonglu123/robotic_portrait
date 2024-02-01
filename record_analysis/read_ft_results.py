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

thin2thick = {'lam':{},'f_actual':{},'f_desired':{}}
thick2thin = {'lam':{},'f_actual':{},'f_desired':{}}
bump2bump = {'lam':{},'f_actual':{},'f_desired':{}}
for data_dir in ['record_0131_1/','record_0131_2/','record_0131_3/']:
    for i in range(3):
        # load recorded data
        load_ft = np.loadtxt(data_dir+'ft_record_load_'+str(i)+'.csv',delimiter=',')
        move_ft = np.loadtxt(data_dir+'ft_record_move_'+str(i)+'.csv',delimiter=',')
        # load planned path and force
        planned_q = np.loadtxt('../path/js_path/strokes_out/'+str(i)+'.csv',delimiter=',')
        planned_f = np.loadtxt(data_dir+'strokes_out/'+str(i)+'.csv',delimiter=',')
        # degrees to radians
        load_ft[:,2:] = np.radians(load_ft[:,2:])
        move_ft[:,2:] = np.radians(move_ft[:,2:])
        # change force direction
        load_ft[:,1] = -load_ft[:,1]
        move_ft[:,1] = -move_ft[:,1]
        # get path length
        lam = calc_lam_js(planned_q,robot)
        lam_exe = calc_lam_js(move_ft[:,2:],robot)
        
        # linear interpolation
        f_desired = np.interp(lam_exe, lam, planned_f)
        
        if planned_f[-1]-planned_f[0]>0.1:
            thin2thick['lam'][data_dir] = lam_exe
            thin2thick['f_actual'][data_dir] = move_ft[:,1]
            thin2thick['f_desired'][data_dir] = f_desired
        elif planned_f[0]-planned_f[-1]>0.1:
            thick2thin['lam'][data_dir] = lam_exe
            thick2thin['f_actual'][data_dir] = move_ft[:,1]
            thick2thin['f_desired'][data_dir] = f_desired
        else:
            bump2bump['lam'][data_dir] = lam_exe
            bump2bump['f_actual'][data_dir] = move_ft[:,1]
            bump2bump['f_desired'][data_dir] = f_desired

# plot thin2thick results on one image
for data_dir in ['record_0131_1/','record_0131_2/','record_0131_3/']:
    plt.plot(thin2thick['lam'][data_dir],thin2thick['f_actual'][data_dir],label='force actual, test '+data_dir[-2])
plt.plot(thin2thick['lam']['record_0131_1/'],thin2thick['f_desired']['record_0131_1/'],label='force reference')
plt.legend(fontsize=16)
plt.xlabel('path length (mm)',fontsize=16)
plt.ylabel('force (N)',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Thin to Thick',fontsize=16)
plt.show()

# plot thick2thin results on one image
for data_dir in ['record_0131_1/','record_0131_2/','record_0131_3/']:
    plt.plot(thick2thin['lam'][data_dir],thick2thin['f_actual'][data_dir],label='force actual, test '+data_dir[-2])
plt.plot(thick2thin['lam']['record_0131_1/'],thick2thin['f_desired']['record_0131_1/'],label='force reference')
plt.legend(fontsize=16)
plt.xlabel('path length (mm)',fontsize=16)
plt.ylabel('force (N)',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Thick to Thin',fontsize=16)
plt.show()

# plot bump2bump results on one image
for data_dir in ['record_0131_1/','record_0131_2/','record_0131_3/']:
    plt.plot(bump2bump['lam'][data_dir],bump2bump['f_actual'][data_dir],label='force actual, test '+data_dir[-2])
plt.plot(bump2bump['lam']['record_0131_1/'],bump2bump['f_desired']['record_0131_1/'],label='force reference')
plt.legend(fontsize=16)
plt.xlabel('path length (mm)',fontsize=16)
plt.ylabel('force (N)',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Bump to Bump',fontsize=16)
plt.show()
    