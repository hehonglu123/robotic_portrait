from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, time
from scipy.interpolate import CubicSpline
from general_robotics_toolbox import *
from copy import deepcopy
sys.path.append('toolbox')
from robot_def import *
from lambda_calc import *
from motion_toolbox import *

################ Robot Connection ################
##RR PARAMETERS
RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot')
# RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')
RR_robot=RR_robot_sub.GetDefaultClientWait(1)
robot_state = RR_robot_sub.SubscribeWire("robot_state")
robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
halt_mode = robot_const["RobotCommandMode"]["halt"]
position_mode = robot_const["RobotCommandMode"]["position_command"]
RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
command_seqno = 1
RR_robot.command_mode = halt_mode
time.sleep(0.1)

## connect to position command mode wire
RR_robot.command_mode = position_mode
cmd_w = RR_robot_sub.SubscribeWire("position_command")

dq_test=np.radians([3,3,3,3,3,3])
dqdot_test=np.radians([1,1,1,1,1,1])
dt_nominal = 0.004

######## test dt ########
test_seg = 1e5
for i in range(test_seg):
    q_now = robot_state.InValue.joint_position
    # Increment command_seqno
    command_seqno += 1
    # Create Fill the RobotJointCommand structure
    joint_cmd = RobotJointCommand()
    joint_cmd.seqno = command_seqno # Strictly increasing command_seqno
    joint_cmd.state_seqno = robot_state.InValue.seqno # Send current robot_state.seqno as failsafe

    # Set the joint command
    joint_cmd.command = q_now

    # Send the joint command to the robot
    cmd_w.SetOutValueAll(joint_cmd)