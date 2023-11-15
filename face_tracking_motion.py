from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys
import numpy as np
sys.path.append('toolbox')
from robot_def import *
from vel_emulate_sub import EmulatedVelocityControl
from lambda_calc import *
from motion_toolbox import *

#########################################################config parameters#########################################################
robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/camera.csv')
radius=500 ###eef position to robot base distance w/o z height
angle_range=np.array([-3*np.pi/4,-np.pi/4]) ###angle range for robot to move
height_range=np.array([500,900]) ###height range for robot to move
p_start=np.array([0,-radius,700])	###initial position
R_start=np.array([	[0,1,0],
					[0,0,-1],
					[-1,0,0]])	###initial orientation
q_start=robot.inv(p_start,R_start,np.zeros(6))	###initial joint position
image_center=np.array([1080,1080])/2	###image center

#########################################################RR PARAMETERS#########################################################
RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot')
RR_robot=RR_robot_sub.GetDefaultClientWait(1)
robot_state = RR_robot_sub.SubscribeWire("robot_state")
robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
halt_mode = robot_const["RobotCommandMode"]["halt"]
position_mode = robot_const["RobotCommandMode"]["position_command"]
trajectory_mode = robot_const["RobotCommandMode"]["trajectory"]
jog_mode = robot_const["RobotCommandMode"]["jog"]
RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
RR_robot.command_mode = halt_mode
time.sleep(0.1)
# RR_robot.reset_errors()
# time.sleep(0.1)

RR_robot.command_mode = position_mode
cmd_w = RR_robot_sub.SubscribeWire("position_command")

vel_ctrl = EmulatedVelocityControl(RR_robot,robot_state, cmd_w)
#enable velocity mode
vel_ctrl.enable_velocity_mode()

########subscription mode
def connect_failed(s, client_id, url, err):
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))

face_tracking_sub=RRN.SubscribeService('rr+tcp://localhost:52222/?service=Face_tracking')
obj = face_tracking_sub.GetDefaultClientWait(1)		#connect, timeout=30s
bbox_wire=face_tracking_sub.SubscribeWire("bbox")

face_tracking_sub.ClientConnectFailed += connect_failed


#jog to initial_position
jog_joint(q_start,vel_ctrl)
while True:
	wire_packet=bbox_wire.TryGetInValue()
	if wire_packet[0]:
		bbox=wire_packet[1]
		print(len(bbox))
		if len(bbox)==0: #if no face detected, jog to initial position
			diff=q_start-vel_ctrl.joint_position()
			if np.linalg.norm(diff)>0.1:
				qdot=diff/np.linalg.norm(diff)
			else:
				qdot=diff
			
			vel_ctrl.set_velocity_command(qdot)
			
		else:
			q_cur=vel_ctrl.joint_position()
			pose_cur=robot.fwd(q_cur)
			if q_cur[0]<angle_range[0] or q_cur[0]>angle_range[1] or pose_cur.p[2]<height_range[0] or pose_cur.p[2]>height_range[1]:
				vel_ctrl.set_velocity_command(np.zeros(6))
				continue
			#calculate size of bbox
			size=np.array([bbox[2]-bbox[0],bbox[3]-bbox[1]])
			#calculate center of bbox
			center=np.array([bbox[0]+size[0]/2,bbox[1]+size[1]/2])
			z_gain=-1
			x_gain=-1e-3
			zd=center[1]-image_center[1]
			xd=center[0]-image_center[0]
			
			try:
				q_temp=robot.inv(pose_cur.p+zd*np.array([0,0,z_gain]),pose_cur.R,q_cur)
			except:
				continue
			q_temp+=xd*np.array([x_gain,0,0,0,0,0])
			q_diff=q_temp-q_cur
			if np.linalg.norm(q_diff)>0.3:
				qdot=q_diff/np.linalg.norm(q_diff)
			else:
				qdot=q_diff
			vel_ctrl.set_velocity_command(qdot)