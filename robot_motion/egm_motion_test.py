from abb_robot_client.egm import EGM
import numpy as np
import time

egm = EGM()

# test motion
min_delta = -np.radians(10)
max_delta = np.radians(10)
joint_vel = np.radians(5)
dt = 0.004
res, state = egm.receive_from_robot(timeout=0.1)
if not res:
    raise Exception("Robot communication lost")
initial_q = state.joint_angles
print(initial_q)
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