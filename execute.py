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


def spline_js(cartesian_path,curve_js,vd,rate=250):
	lam=calc_lam_cs(cartesian_path)
	# polyfit=np.polyfit(lam,curve_js,deg=40)
	# lam=np.linspace(0,lam[-1],int(rate*lam[-1]/vd))
	# return np.vstack((np.poly1d(polyfit[:,0])(lam), np.poly1d(polyfit[:,1])(lam), np.poly1d(polyfit[:,2])(lam), np.poly1d(polyfit[:,3])(lam), np.poly1d(polyfit[:,4])(lam), np.poly1d(polyfit[:,5])(lam))).T
	polyfit=CubicSpline(lam,curve_js)
	lam=np.linspace(0,lam[-1],int(rate*lam[-1]/vd))
	return polyfit(lam)

def main():
	global vel_ctrl
	rate=250
	img_name='me_out'
	ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',')
	num_segments=len(glob.glob('path/cartesian_path/'+img_name+'/*.csv'))
	robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen.csv')
	##RR PARAMETERS
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


	for i in range(num_segments):
		cartesian_path=np.loadtxt('path/cartesian_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
		curve_js=np.loadtxt('path/js_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,6))
		if len(curve_js)>1:
			###spline the path
			curve_js_spline=spline_js(cartesian_path,curve_js,vd=10)

			#jog to starting point
			pose_start=robot.fwd(curve_js[0])
			p_start=pose_start.p+20*ipad_pose[:3,-2]
			q_start=robot.inv(p_start,pose_start.R,curve_js[0])        
			jog_joint(q_start,vel_ctrl)

			#execute path
			# qdot_d=rate*(curve_js_spline[1]-curve_js_spline[0])
			# gain=10
			# for j in range(len(curve_js_spline)):
			#     qdot=qdot_d+gain*(curve_js_spline[j]-vel_ctrl.joint_position())
			#     vel_ctrl.set_velocity_command(qdot)
			#     try:
			#         qdot_d=rate*(curve_js_spline[j+1]-curve_js_spline[j])
			#     except:
			#         break
				
			#     time.sleep(1/rate)

			for j in range(len(curve_js)):
				while np.linalg.norm(vel_ctrl.joint_position()-curve_js[j])>0.01:
					qdot_d=curve_js[j]-vel_ctrl.joint_position()
					vel_ctrl.set_velocity_command(0.1*qdot_d/np.linalg.norm(qdot_d))
			
			#jog to end point
			pose_end=robot.fwd(curve_js[-1])
			p_end=pose_end.p+20*ipad_pose[:3,-2]
			q_end=robot.inv(p_end,pose_end.R,curve_js[-1])
			jog_joint(q_end,vel_ctrl)
			
	vel_ctrl.disable_velocity_mode()
if __name__ == '__main__':
	main()