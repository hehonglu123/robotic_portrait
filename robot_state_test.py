from RobotRaconteur.Client import *
import numpy as np
import time, sys

sys.path.append('toolbox')
from robot_def import *
from utils import *

def force_prop(force,torque,T):###propagate force from sensor to the tip
	force_tip=force+np.cross(torque,T)
	return np.linalg.norm(force_tip)

def get_force_ur(robot,q,torque):
	gravity_torque=np.array([ 1.00890585e-16, -2.45729382e+01, -1.00239347e+01, -2.41221986e+00, -2.27689380e-02,  0.00000000e+00])
	torque_act=torque-gravity_torque

	tf=robot.jacobian(q)@torque_act

	return force_prop(tf[3:6],tf[0:3],robot.p_tool)

robot=robot_obj('ur5','config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')

RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')
RR_robot=RR_robot_sub.GetDefaultClientWait(1)
robot_state = RR_robot_sub.SubscribeWire("robot_state")

time.sleep(1)
while True:
    print(robot_state.InValue.joint_effort[4])
    # print(get_force_ur(robot,robot_state.InValue.joint_position,robot_state.InValue.joint_effort))