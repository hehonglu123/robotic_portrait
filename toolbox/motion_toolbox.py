import numpy as np
import time

def jog_joint(q,vel_ctrl,v=0.5):

	while np.linalg.norm(q-vel_ctrl.joint_position())>0.01:
		diff=q-vel_ctrl.joint_position()
		if np.linalg.norm(diff)>0.5:
			qdot=v*diff/np.linalg.norm(diff)
		else:
			qdot=v*diff

		vel_ctrl.set_velocity_command(qdot)

	vel_ctrl.set_velocity_command(np.zeros((6,)))



# def jog_joint_position_cmd(q,RobotJointCommand, cmd_w, command_seqno, robot_state,v=0.5,wait_time=0):

# 	total_time=np.linalg.norm(q-robot_state.InValue.joint_position)/v

# 	start_time=time.time()
# 	while time.time()-start_time<total_time:
# 		# Increment command_seqno
# 		command_seqno += 1
# 		# Create Fill the RobotJointCommand structure
# 		joint_cmd = RobotJointCommand()
# 		joint_cmd.seqno = command_seqno # Strictly increasing command_seqno
# 		joint_cmd.state_seqno = robot_state.InValue.seqno # Send current robot_state.seqno as failsafe
		
# 		# Set the joint command
# 		frac=(time.time()-start_time)/total_time
# 		joint_cmd.command = frac*q+(1-frac)*robot_state.InValue.joint_position
	
# 		# Send the joint command to the robot
# 		cmd_w.SetOutValueAll(joint_cmd)
	
# 	###additional points for accuracy
# 	start_time=time.time()
# 	while time.time()-start_time<wait_time:
# 		# Increment command_seqno
# 		command_seqno += 1
# 		# Create Fill the RobotJointCommand structure
# 		joint_cmd = RobotJointCommand()
# 		joint_cmd.seqno = command_seqno # Strictly increasing command_seqno
# 		joint_cmd.state_seqno = robot_state.InValue.seqno # Send current robot_state.seqno as failsafe
		
# 		# Set the joint command
# 		joint_cmd.command = q

# 		# Send the joint command to the robot
# 		cmd_w.SetOutValueAll(joint_cmd)
	
# 	return command_seqno


# def trajectory_position_cmd(q_all,RobotJointCommand, cmd_w, command_seqno, robot_state,v=0.5):

# 	lamq_bp=[0]
# 	for i in range(len(q_all)-1):
# 		lamq_bp.append(lamq_bp[-1]+np.linalg.norm(q_all[i+1]-q_all[i]))
# 	time_bp=np.array(lamq_bp)/v
# 	seg=1

# 	start_time=time.time()
# 	while time.time()-start_time<time_bp[-1]:
# 		# now=time.time()
# 		# Increment command_seqno
# 		command_seqno += 1
# 		# Create Fill the RobotJointCommand structure
# 		joint_cmd = RobotJointCommand()
# 		joint_cmd.seqno = command_seqno # Strictly increasing command_seqno
# 		joint_cmd.state_seqno = robot_state.InValue.seqno # Send current robot_state.seqno as failsafe
		

# 		#find current segment
# 		if time.time()-start_time>time_bp[seg]:
# 			seg+=1
# 		if seg==len(q_all):
# 			break
# 		frac=(time.time()-start_time-time_bp[seg-1])/(time_bp[seg]-time_bp[seg-1])

# 		# Set the joint command
# 		joint_cmd.command = frac*q_all[seg]+(1-frac)*q_all[seg-1]

# 		# Send the joint command to the robot
# 		cmd_w.SetOutValueAll(joint_cmd)

# 		# print(time.time()-now)
	
# 	return command_seqno


