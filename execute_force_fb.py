from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import glob, cv2, sys, time
from scipy.interpolate import CubicSpline

sys.path.append('toolbox')
from robot_def import *
from vel_emulate_sub import EmulatedVelocityControl
from lambda_calc import *
from motion_toolbox import *
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

	# Send the joint command to the robot
	cmd_w.SetOutValueAll(joint_cmd)

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



def trajectory_position_cmd(q_all,v=0.4):
	global RobotJointCommand, cmd_w, command_seqno, robot_state


	lamq_bp=[0]
	for i in range(len(q_all)-1):
		lamq_bp.append(lamq_bp[-1]+np.linalg.norm(q_all[i+1]-q_all[i]))
	time_bp=np.array(lamq_bp)/v
	seg=1

	start_time=time.time()
	while time.time()-start_time<time_bp[-1]:

		#find current segment
		if time.time()-start_time>time_bp[seg]:
			seg+=1
		if seg==len(q_all):
			break
		frac=(time.time()-start_time-time_bp[seg-1])/(time_bp[seg]-time_bp[seg-1])
		position_cmd(frac*q_all[seg]+(1-frac)*q_all[seg-1])
		


def spline_js(cartesian_path,curve_js,vd,rate=250):
	lam=calc_lam_cs(cartesian_path)
	# polyfit=np.polyfit(lam,curve_js,deg=40)
	# lam=np.linspace(0,lam[-1],int(rate*lam[-1]/vd))
	# return np.vstack((np.poly1d(polyfit[:,0])(lam), np.poly1d(polyfit[:,1])(lam), np.poly1d(polyfit[:,2])(lam), np.poly1d(polyfit[:,3])(lam), np.poly1d(polyfit[:,4])(lam), np.poly1d(polyfit[:,5])(lam))).T
	polyfit=CubicSpline(lam,curve_js)
	lam=np.linspace(0,lam[-1],int(rate*lam[-1]/vd))
	return polyfit(lam)

def main():
	global  RobotJointCommand, cmd_w, command_seqno, robot_state
	img_name='wen_out'
	ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',')
	num_segments=len(glob.glob('path/cartesian_path/'+img_name+'/*.csv'))
	robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen.csv')
	# robot=robot_obj('ur5','config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')

	####################################################FT Connection####################################################
	ati_tf=NET_FT('192.168.60.100')
	ati_tf.start_streaming()
	H_pentip2ati=np.loadtxt('config/pentip2ati.csv',delimiter=',')
	F_MAX=0	#maximum pushing force 10N

	##RR PARAMETERS
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

	start=True
	for i in range(num_segments):
		pixel_path=np.loadtxt('path/pixel_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
		cartesian_path=np.loadtxt('path/cartesian_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
		curve_js=np.loadtxt('path/js_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,6))
		print(i)
		if len(curve_js)>1:

			pose_start=robot.fwd(curve_js[0])
			if start:
				#jog to starting point
				p_start=pose_start.p+30*ipad_pose[:3,-2]
				q_start=robot.inv(p_start,pose_start.R,curve_js[0])[0]
				jog_joint_position_cmd(q_start)
				jog_joint_position_cmd(curve_js[0],wait_time=1)
				ati_tf.set_tare_from_ft()	#clear bias
				start=False
			else:
				pose_cur=robot.fwd(robot_state.InValue.joint_position)
				p_mid=(pose_start.p+pose_cur.p)/2+20*ipad_pose[:3,-2]
				q_mid=robot.inv(p_mid,pose_start.R,curve_js[0])[0]
				#arc-like trajectory to next segment
				trajectory_position_cmd(np.vstack((robot_state.InValue.joint_position,q_mid,curve_js[0])),v=0.2)
				jog_joint_position_cmd(curve_js[0],wait_time=0.3)

			traversal_velocity=50	#mm/s
			###force feedback drawing trajectory interp
			
			# lamq_bp=[0]
			# for m in range(len(curve_js)-1):
			# 	lamq_bp.append(lamq_bp[-1]+np.linalg.norm(curve_js[m+1]-curve_js[m]))
			# time_bp=np.array(lamq_bp)/v
			# seg=1
			# start_time=time.time()
			# while time.time()-start_time<time_bp[-1]:

			# 	#find current segment, give command based on timestamped segments
			# 	if time.time()-start_time>time_bp[seg]:
			# 		seg+=1
			# 	if seg==len(curve_js):
			# 		break
			# 	frac=(time.time()-start_time-time_bp[seg-1])/(time_bp[seg]-time_bp[seg-1])
			# 	#interpolated js position
			# 	q_d=frac*curve_js[seg]+(1-frac)*curve_js[seg-1]
			# 	#interpolated force
			# 	f_d=F_MAX*(frac*pixel_path[seg,-1]+(1-frac)*pixel_path[seg-1,-1])
			# 	#force feedback
			# 	res, tf, status = ati_tf.try_read_ft_streaming(.1)###get force feedback
			# 	f_cur=force_prop(tf,H_pentip2ati[:-1,-1])
			# 	position_gain=0.05
			# 	normal_adjustment=position_gain*(f_d-f_cur)
			# 	q_cmd=q_d+normal_adjustment*np.linalg.pinv(robot.jacobian(q_d))@np.hstack((np.zeros(3),-ipad_pose[:3,-2]))

			# 	position_cmd(q_cmd)
			# 	print(normal_adjustment,f_cur)

			###force feedback, traj tracking QP
			pose_cur=robot.fwd(robot_state.InValue.joint_position)
			q_cmd=curve_js[0]
			ts=time.time()
			for m in range(1,len(curve_js)):
				# v_d=cartesian_path[m]-cartesian_path[m-1]
				# v_d=v_d/np.linalg.norm(v_d)
				f_d=F_MAX*pixel_path[m,-1]
				while np.linalg.norm(pose_cur.p[:2]-cartesian_path[m,:2])>2:	###loop to get to waypoint
					v_d=cartesian_path[m]-pose_cur.p
					if np.linalg.norm(v_d)>2.5:
						v_d=traversal_velocity*v_d/np.linalg.norm(v_d)
					# print('v_d: ',v_d)
					#force feedback
					res, tf, status = ati_tf.try_read_ft_streaming(.1)###get force feedback
					f_cur=force_prop(tf,H_pentip2ati[:-1,-1])
					print(f_cur)
					# position_gain=0.01
					position_gain=0.0
					v_d+=position_gain*(f_d-f_cur)*(-ipad_pose[:3,-2])
					p_inc=(time.time()-ts)*v_d	###discrete increments in dt
					# print('p_inc: ',p_inc)
					dq=np.linalg.pinv(robot.jacobian(robot_state.InValue.joint_position))@np.hstack((np.zeros(3),p_inc))
					q_cmd+=dq
					position_cmd(q_cmd)

					pose_cur=robot.fwd(robot_state.InValue.joint_position)
					ts=time.time()

				
			#jog to end point
			jog_joint_position_cmd(curve_js[-1],wait_time=0.3)
	
	#jog to end point
	pose_end=robot.fwd(curve_js[-1])
	p_end=pose_end.p+20*ipad_pose[:3,-2]
	q_end=robot.inv(p_end,pose_end.R,curve_js[-1])[0]
	jog_joint_position_cmd(q_end)

if __name__ == '__main__':
	main()