import numpy as np

def jog_joint(q,vel_ctrl,v=0.5):

	while np.linalg.norm(q-vel_ctrl.joint_position())>0.01:
		diff=q-vel_ctrl.joint_position()
		if np.linalg.norm(diff)>0.5:
			qdot=v*diff/np.linalg.norm(diff)
		else:
			qdot=v*diff

		vel_ctrl.set_velocity_command(qdot)

	vel_ctrl.set_velocity_command(np.zeros((6,)))

