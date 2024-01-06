from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import glob, cv2, sys, time
from sklearn.decomposition import PCA

sys.path.append('toolbox')
from robot_def import *
from utils import *
from rpi_ati_net_ft import *

def position_cmd(q):
	global RobotJointCommand, cmd_w, command_seqno, robot_state

	# Increment command_seqno
	command_seqno += 1
	# Create Fill the RobotJointCommand structure
	joint_cmd = RobotJointCommand()
	joint_cmd.seqno = command_seqno # Strictly increasing command_seqno
	joint_cmd.state_seqno = robot_state.InValue.seqno # Send current robot_state.seqno as failsafe
	
	# Set the joint command
	joint_cmd.command = q

	try:
		# Send the joint command to the robot
		cmd_w.SetOutValueAll(joint_cmd)
	
	except:
		print(joint_cmd.command,q)
		raise Exception('command failed')

def jog_joint_position_cmd(q,v=0.4,wait_time=0):
	global robot_state

	q_start=robot_state.InValue.joint_position
	total_time=np.linalg.norm(q-q_start)/v

	start_time=time.time()
	while time.time()-start_time<total_time:
		# Set the joint command
		frac=(time.time()-start_time)/total_time
		position_cmd(frac*q+(1-frac)*q_start)
		
	###additional points for accuracy
	start_time=time.time()
	while time.time()-start_time<wait_time:
		position_cmd(q)


def get_force_ur(jacobian,torque):
	gravity_torque=np.array([0,0,0,0,0,0.1])
	torque_act=torque-gravity_torque
	return np.linalg.pinv(jacobian)@torque_act


####################################################FT Connection####################################################
ati_tf=NET_FT('192.168.60.100')
ati_tf.start_streaming()
H_pentip2ati=np.loadtxt('config/pentip2ati.csv',delimiter=',')

#########################################################RR PARAMETERS#########################################################
RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot')
# RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')

RR_robot=RR_robot_sub.GetDefaultClientWait(1)
robot_state = RR_robot_sub.SubscribeWire("robot_state")
robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
halt_mode = robot_const["RobotCommandMode"]["halt"]
position_mode = robot_const["RobotCommandMode"]["position_command"]
trajectory_mode = robot_const["RobotCommandMode"]["trajectory"]
jog_mode = robot_const["RobotCommandMode"]["jog"]
RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
command_seqno = 1
RR_robot.command_mode = halt_mode
time.sleep(0.1)
# RR_robot.reset_errors()
# time.sleep(0.1)

RR_robot.command_mode = position_mode
cmd_w = RR_robot_sub.SubscribeWire("position_command")


#########################################################Robot config parameters#########################################################
robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen.csv')
q_seed=np.zeros(6)

# robot=robot_obj('ur5','config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')
# q_seed=np.radians([0,-54.8,110,-142,-90,0])

ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',')
paper_size=np.loadtxt('config/paper_size.csv',delimiter=',')
R_pencil=ipad_pose[:3,:3]@Ry(np.pi)


corners_offset=np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]])*1.2*np.array([paper_size[0],paper_size[1],0])/2
corners=np.dot(ipad_pose[:3,:3],corners_offset.T).T+np.tile(ipad_pose[:3,-1],(4,1))

###loop four corners to get precise position base on force feedback
corners_adjusted=[]
f_d=10	#10N push down
for corner in corners:
	corner_top=corner+30*ipad_pose[:3,-2]
	q_corner_top=robot.inv(corner_top,R_pencil,q_seed)[0]	###initial joint position
	jog_joint_position_cmd(q_corner_top,v=0.2)

	ati_tf.set_tare_from_ft()	#clear bias

	qdot=np.linalg.pinv(robot.jacobian(q_corner_top))@np.hstack((np.zeros(3),-ipad_pose[:3,-2]))		#motion direction
	K=0.02
	q_cur=copy.deepcopy(q_corner_top)
	f_cur=0
	while f_cur<f_d:
		position_cmd(q_cur+K*(f_d-f_cur)*qdot)
		q_cur=q_cur+K*qdot
		res, tf, status = ati_tf.try_read_ft_streaming(.1)###get force feedback
		f_cur=force_prop(tf,H_pentip2ati[:-1,-1])
		print(f_cur)
	
	corners_adjusted.append(robot.fwd(q_cur).p)

	jog_joint_position_cmd(q_corner_top,v=0.2)


###UPDATE IPAD POSE based on new corners
p_all=np.array(corners_adjusted)
#identify the center point and the plane
center=np.mean(p_all,axis=0)
pca = PCA()
pca.fit(p_all)
R_temp = pca.components_.T		###decreasing variance order
if R_temp[:,0]@center<0:		###correct orientation
	R_temp[:,0]=-R_temp[:,0]
if R_temp[:,-1]@R_pencil[:,-1]>0:
	R_temp[:,-1]=-R_temp[:,-1]

R_temp[:,1]=np.cross(R_temp[:,2],R_temp[:,0])

np.savetxt('config/ipad_pose.csv', H_from_RT(R_temp,center), delimiter=',')
		