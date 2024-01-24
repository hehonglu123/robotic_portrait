import numpy as np
import sys
import time
from general_robotics_toolbox import *
sys.path.append('toolbox')
from robot_def import *

robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen.csv')

dt_arr = []
for i in range(10000):
    q = np.radians(np.random.rand(6)*40)
    T = robot.fwd(q)
    st=time.perf_counter()
    qik = robot.inv(T.p,T.R,q)[0]
    et=time.perf_counter()
    dt_arr.append(et-st)
dt_arr=np.array(dt_arr)
print("IK duration:",np.mean(dt_arr),", Max:",np.max(dt_arr),", >0.004:",len(dt_arr[dt_arr>0.004]))