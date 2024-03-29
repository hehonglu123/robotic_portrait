from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys, cv2
import numpy as np
sys.path.append('toolbox')
sys.path.append('motion_planning')
from robot_def import *
from vel_emulate_sub import EmulatedVelocityControl
from lambda_calc import *
from motion_toolbox import *
from portrait import *
from traversal_force import *
from traj_gen_cartesian import *


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
		

### parameters and variables
paper_size=np.loadtxt('config/paper_size.csv',delimiter=',')
pen_radius=np.loadtxt('config/pen_radius.csv',delimiter=',')
ipad_pose=np.loadtxt('config/ipad_pose_ur.csv',delimiter=',')
anime = AnimeGANv3('models/AnimeGANv3_PortraitSketch.onnx')
faceseg = FaceSegmentation()

#########################################################config parameters#########################################################
# robot_cam=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/camera.csv')
# robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen2.csv')
# radius=500 ###eef position to robot base distance w/o z height
# angle_range=np.array([-3*np.pi/4,-np.pi/4]) ###angle range for robot to move
# height_range=np.array([500,900]) ###height range for robot to move
# p_start=np.array([0,-radius,700])	###initial position
# R_start=np.array([	[0,1,0],
# 					[0,0,-1],
# 					[-1,0,0]])	###initial orientation
# q_start=robot_cam.inv(p_start,R_start,np.zeros(6))[0]	###initial joint position
# image_center=np.array([1080,1080])/2	###image center

#########################################################UR config parameters#########################################################
robot_cam=robot_obj('ur5','config/ur5_robot_default_config.yml',tool_file_path='config/camera_ur.csv')
robot=robot_obj('ur5','config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')
radius=500 ###eef position to robot base distance w/o z height
angle_range=np.array([-np.pi/4,np.pi/4]) ###angle range for robot to move
height_range=np.array([500,900]) ###height range for robot to move
p_tracking_start=np.array([-radius,0,750])	###initial position
R_tracking_start=np.array([	[0,0,-1],
					[0,-1,0],
					[-1,0,0]])	###initial orientation
q_seed=np.radians([0,-54.8,110,-142,-90,0])
image_center=np.array([1080,1080])/2	###image center
###initial joint position
q_tracking_start=robot.inv(p_tracking_start,R_tracking_start,q_seed)[0]	
# Orientation of pencil
R_pencil=ipad_pose[:3,:3]@Ry(np.pi)
# flipping pose
p_flip=np.array([-577.69418679,   59.06883188, -117])
q_flip=robot.inv(p_flip,R_pencil,q_seed)[0]
pose=robot.fwd(q_flip)
q_flip_top=robot.inv(pose.p+30*ipad_pose[:3,-2],pose.R,q_flip)[0]
print('q_flip_top',np.degrees(q_flip_top))
print('q_flip',np.degrees(q_flip))

#########################################################RR PARAMETERS#########################################################
RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')
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


########subscription mode
def connect_failed(s, client_id, url, err):
	print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))

face_tracking_sub=RRN.SubscribeService('rr+tcp://localhost:52222/?service=Face_tracking')
obj = face_tracking_sub.GetDefaultClientWait(1)		#connect, timeout=30s
bbox_wire=face_tracking_sub.SubscribeWire("bbox")
image_wire=face_tracking_sub.SubscribeWire("frame_stream")

face_tracking_sub.ClientConnectFailed += connect_failed


#########################################################EXECUTION#########################################################
while True:
	start_time=time.time()
	#jog to initial_position
	jog_joint_position_cmd(q_tracking_start,v=0.2,wait_time=0.5)

	q_cmd_prev=q_tracking_start
	while True:
		loop_start_time=time.time()
		wire_packet=bbox_wire.TryGetInValue()
		q_cur=robot_state.InValue.joint_position

		if wire_packet[0]:
			bbox=wire_packet[1]
			if len(bbox)==0: #if no face detected, jog to initial position
				diff=q_tracking_start-q_cur
				if np.linalg.norm(diff)>0.1:
					qdot=diff/np.linalg.norm(diff)
				else:
					qdot=diff
				
				q_cmd=q_cmd_prev+qdot*(time.time()-loop_start_time)
				position_cmd(q_cmd)
				q_cmd_prev=copy.deepcopy(q_cmd)
				
			else:	#if face detected
				pose_cur=robot_cam.fwd(q_cur)
				if q_cur[0]<angle_range[0] or q_cur[0]>angle_range[1] or pose_cur.p[2]<height_range[0] or pose_cur.p[2]>height_range[1]:
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
					q_temp=robot_cam.inv(pose_cur.p+zd*np.array([0,0,z_gain]),pose_cur.R,q_cur)[0]
				except:
					continue
				q_temp+=xd*np.array([x_gain,0,0,0,0,0])
				q_diff=q_temp-q_cur
				if np.linalg.norm(q_diff)>0.8:
					qdot=0.8*q_diff/np.linalg.norm(q_diff)
				else:
					qdot=q_diff

				q_cmd=q_cmd_prev+qdot*(time.time()-loop_start_time)
				position_cmd(q_cmd)
				q_cmd_prev=copy.deepcopy(q_cmd)
				
				if np.linalg.norm(qdot)<0.1:
					# print(time.time()-start_time)
					if time.time()-start_time>3:
						break
				else:
					start_time=time.time()



	RR_image=image_wire.TryGetInValue()
	if RR_image[0]:
		img=RR_image[1]
		img=np.array(img.data,dtype=np.uint8).reshape((img.image_info.height,img.image_info.width,3))
		#get the image within the bounding box, a bit larger than the bbox
		img=img[int(bbox[1]-size[1]/5):int(bbox[3]+size[1]/9),int(bbox[0]-size[0]/9):int(bbox[2]+size[0]/9),:]
	
	# move to prepare position
	jog_joint_position_cmd(q_flip_top,v=0.1)

	print('IMAGE TAKEN')
	cv2.imwrite('img.jpg',img)
	# cv2.imshow("img", img)
	# cv2.waitKey(0)
	# cv2. destroyAllWindows() 
	###############################################################################PLANNING########################################################################################

	## face segmentation
	image_mask = faceseg.forward_faceonly(img)
	#convert dark pixels to bright pixels
	# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray_image = img
	gray_image_masked = cv2.bitwise_and(gray_image, gray_image, mask = image_mask)
	# get second masked value (background) mask must be inverted
	background = np.full(gray_image.shape, 255, dtype=np.uint8)
	bk = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(image_mask))
	gray_image_masked = cv2.add(gray_image_masked, bk)

	###portrait GAN
	anime_img = anime.forward(gray_image_masked)
	img_gray=cv2.cvtColor(anime_img, cv2.COLOR_BGR2GRAY)
	pixel2mm=min(paper_size/img_gray.shape)

	cv2.imwrite('img_out.jpg',anime_img)
	# cv2.imshow("img", anime_img)
	# cv2.waitKey(0)
	# cv2. destroyAllWindows() 

	###Pixel Traversal
	print('TRAVERSING PIXELS')
	pixel_paths=image_traversal(anime_img,paper_size,pen_radius)


	###Project to IPAD
	cartesian_paths=image2plane(anime_img, ipad_pose, pixel2mm,pixel_paths)

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# for cartesian_path in cartesian_paths:
	# 	###plot out the path in 3D
	# 	ax.plot(cartesian_path[:,0], cartesian_path[:,1], cartesian_path[:,2], 'b')
	# ax.set_xlabel('X Label')
	# ax.set_ylabel('Y Label')
	# ax.set_zlabel('Z Label')
	# plt.show()

	###Solve Joint Trajectory
	print("SOLVING JOINT TRAJECTORY")
	js_paths=[]
	for cartesian_path in cartesian_paths:
		curve_js=robot.find_curve_js(cartesian_path,[R_pencil]*len(cartesian_path),q_seed)
		js_paths.append(curve_js)



	print("PAGE FLIPPING")
	jog_joint_position_cmd(q_flip_top,v=0.1)
	jog_joint_position_cmd(q_flip,v=0.1,wait_time=0.2)
	jog_joint_position_cmd(q_flip_top,v=0.1)



	print('START DRAWING')
	###Execute
	start=True
	for i in range(len(js_paths)):
		cartesian_path=cartesian_paths[i]
		curve_js=js_paths[i]
		if len(curve_js)>1:
			pose_start=robot.fwd(curve_js[0])
			if start:
				#jog to starting point
				p_start=pose_start.p+30*ipad_pose[:3,-2]
				q_start=robot.inv(p_start,pose_start.R,curve_js[0])[0]
				jog_joint_position_cmd(q_start,v=0.2)
				jog_joint_position_cmd(curve_js[0],wait_time=1)
				start=False
			else:
				pose_cur=robot.fwd(robot_state.InValue.joint_position)
				p_mid1=pose_cur.p+20*ipad_pose[:3,-2]
				q_mid1=robot.inv(p_mid1,pose_start.R,curve_js[0])[0]
				p_mid2=pose_start.p+20*ipad_pose[:3,-2]
				q_mid2=robot.inv(p_mid2,pose_start.R,curve_js[0])[0]
				#arc-like trajectory to next segment
				trajectory_position_cmd(np.vstack((robot_state.InValue.joint_position,q_mid1,q_mid2,curve_js[0])),v=0.2)
				jog_joint_position_cmd(curve_js[0],wait_time=0.1)

			#drawing trajectory
			trajectory_position_cmd(curve_js,v=0.15)
			#jog to end point in case
			jog_joint_position_cmd(curve_js[-1],wait_time=0.3)

	#jog to end point
	pose_end=robot.fwd(curve_js[-1])
	p_end=pose_end.p+30*ipad_pose[:3,-2]
	q_end=robot.inv(p_end,pose_end.R,curve_js[-1])[0]
	jog_joint_position_cmd(q_end)

	print('FINISHED DRAWING')