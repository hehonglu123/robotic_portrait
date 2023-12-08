import sys
import numpy as np
from general_robotics_toolbox import * 
from general_robotics_toolbox import robotraconteur as rr_rox


q_seed=np.radians([0,-54.8,110,-142,-90,0])
R=np.array([[ 1.0000000e+00,  0.0000000e+00, -1.2246468e-16],
 [ 0.0000000e+00, -1.0000000e+00,  0.0000000e+00],
 [-1.2246468e-16,  0.0000000e+00, -1.0000000e+00]])
p=np.array([-525.55205452,  -52.06721068, -111.20878221])


with open('config/ur5_robot_default_config.yml', 'r') as f:
	robot = rr_rox.load_robot_info_yaml_to_robot(f)
tool_H=np.loadtxt('config/heh6_pen_ur.csv',delimiter=',')
robot.R_tool=tool_H[:3,:3]
robot.p_tool=tool_H[:3,-1]

q=np.radians([-8,-105.75,93.64,-163,-90,23.7])
print(fwdkin(robot,q))
# q=iterative_invkin(robot,Transform(R,p),q_seed)
# print(np.linalg.cond(robotjacobian(robot,list(q[1][0]))))
# print(fwdkin(robot,list(q[1][0])))