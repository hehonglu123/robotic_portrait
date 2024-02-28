from RobotRaconteur.Client import *
import numpy as np
import time

##RR PARAMETERS
RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot')
# RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')
RR_robot=RR_robot_sub.GetDefaultClientWait(1)
robot_state = RR_robot_sub.SubscribeWire("robot_state")
time.sleep(0.1)

robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
halt_mode = robot_const["RobotCommandMode"]["halt"]
position_mode = robot_const["RobotCommandMode"]["position_command"]
RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
RR_robot.command_mode = halt_mode
time.sleep(0.1)
## connect to position command mode wire
RR_robot.command_mode = position_mode
cmd_w = RR_robot_sub.SubscribeWire("position_command")

# test motion
command_seqno = 0
min_delta = -np.radians(10)
max_delta = np.radians(10)
joint_vel = np.radians(5)
dt = 0.004
initial_q = robot_state.InValue.joint_position
dt_actual = []
print("Start testing")
while True:
    try:
        st = time.time()
        q_now = robot_state.InValue.joint_position
        q_next = q_now + np.array([0,0,0,0,0,joint_vel*dt])
        if q_next[5] >= initial_q[5]+max_delta or q_next[5] <= initial_q[5]+min_delta:
            joint_vel = -joint_vel
        
        command_seqno+=1
        cmd = RobotJointCommand()
        cmd.seqno = command_seqno
        cmd.state_seqno = robot_state.InValue.seqno
        cmd.command = q_next
        cmd_w.OutValue.SetOutValueAll(cmd)
        dt_actual.append(time.time()-st)
    except KeyboardInterrupt:
        break
print("Average time: ", np.mean(dt_actual))
print("Aveerage of 10 Longest time:", np.mean(np.sort(dt_actual)[-10:]))
print("Aveerage of 10 Shortest time:", np.mean(np.sort(dt_actual)[:10]))