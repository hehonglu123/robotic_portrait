from RobotRaconteur.Client import *
import numpy as np
import time
from matplotlib import pyplot as plt

##RR PARAMETERS
# RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot')
RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')
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
joint_num=0
command_seqno = 0
min_delta = -np.radians(10)
max_delta = np.radians(10)
joint_vel = np.radians(8)
dt = 0.01
initial_q = robot_state.InValue.joint_position
dt_actual = []
joint_actual = []
stamps=[]
print("Start testing")
start_time = time.time()
while True:
    try:
        st = time.time()
        stamps.append(st-start_time)
        time.sleep(dt)
        q_now = robot_state.InValue.joint_position
        diff_q = np.zeros(6)
        diff_q[joint_num] = joint_vel*dt
        q_next = q_now + diff_q
        if q_next[joint_num] >= initial_q[joint_num]+max_delta or q_next[joint_num] <= initial_q[joint_num]+min_delta:
            print("change direction")
            joint_vel = -joint_vel
        
        command_seqno+=1
        cmd = RobotJointCommand()
        cmd.seqno = command_seqno
        cmd.state_seqno = robot_state.InValue.seqno
        cmd.command = q_next
        cmd_w.SetOutValueAll(cmd)
        dt_actual.append(time.time()-st)
        joint_actual.append(q_now[joint_num])
    except KeyboardInterrupt:
        break
print("Average time: ", np.mean(dt_actual))
print("Aveerage of 10 Longest time:", np.mean(np.sort(dt_actual)[-10:]))
print("Aveerage of 10 Shortest time:", np.mean(np.sort(dt_actual)[:10]))
joint_vel_exec = np.diff(np.degrees(joint_actual))/dt_actual[1:]
plt.plot(joint_vel_exec)
plt.show()
last_stamp=0
for i in range(1000,2000):
    print("Joint actual:",i,stamps[i], joint_actual[i])
    if joint_actual[i]-joint_actual[i-1] != 0:
        print("Joint value change. dt:",stamps[i]-last_stamp)
        last_stamp = stamps[i]