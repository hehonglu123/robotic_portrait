from abb_robot_client.egm import EGM
import numpy as np
import time
from matplotlib import pyplot as plt

egm = EGM()

# test motion
min_delta = -np.radians(10)
max_delta = np.radians(10)
joint_vel = -1*np.radians(5)
lookahead_time = 0.132
dt = 0.004
res, state = egm.receive_from_robot(timeout=0.1)
if not res:
    raise Exception("Robot communication lost")
initial_q = np.radians(state.joint_angles)
# generate trajectory
j5_traj=[]
this_q=initial_q[5]
for i in range(1,2001):
    this_q=this_q+joint_vel*dt
    if (joint_vel>0 and this_q >= initial_q[5]+max_delta) or (joint_vel<0 and this_q <= initial_q[5]+min_delta):
        joint_vel = -joint_vel
        print("change direction")
    j5_traj.append(this_q)

dt_actual = []
joint_actual=[]
print("Start testing")
command_seqno = 0
while True:
    try:
        st = time.time()
        res, state = egm.receive_from_robot(timeout=0.1)
        if not res:
            raise Exception("Robot communication lost")
        q_now = np.radians(state.joint_angles)
        q_next = q_now + np.array([0,0,0,0,0,joint_vel*(dt+lookahead_time)])
        if q_next[5] >= initial_q[5]+max_delta or q_next[5] <= initial_q[5]+min_delta:
            joint_vel = -joint_vel
            print("change direction")
        # q_next = np.array([0,0,0,0,0,j5_traj[command_seqno]])
        # command_seqno+=1
        # if command_seqno>=2000:
        #     command_seqno=0
        
        egm.send_to_robot(np.degrees(q_next))
        dt_actual.append(time.time()-st)
        joint_actual.append(q_now[5])
    except KeyboardInterrupt:
        break
print("Average time: ", np.mean(dt_actual))
print("Aveerage of 10 Longest time:", np.mean(np.sort(dt_actual)[-10:]))
print("Aveerage of 10 Shortest time:", np.mean(np.sort(dt_actual)[:10]))
joint_vel_exec = np.diff(np.degrees(joint_actual))/dt_actual[1:]
plt.plot(joint_vel_exec)
plt.show()
plt.plot(joint_actual)
plt.plot(j5_traj)
plt.show()
for i in range(1000,2000):
    print("Joint actual:",i, joint_actual[i])