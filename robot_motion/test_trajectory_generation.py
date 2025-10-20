import numpy as np
from matplotlib import pyplot as plt

# velocity and acceleration
lin_vel = 10
lin_acc = 10
# Calculate the path length
lam = np.arange(0,50,0.1)

time_for_accelration = lin_vel/lin_acc
distance_for_accelration = 0.5*lin_acc*time_for_accelration**2
if lam[-1]<2*distance_for_accelration:
    half_travel_time = np.sqrt((lam[-1])/lin_acc)
    half_lambda = 0.5*lin_acc*(np.arange(0,half_travel_time,0.004)**2)
    lam_stamped = np.hstack((half_lambda,lam[-1]-half_lambda[::-1]))
    acc_time = half_travel_time
else:
    constant_velocity_time = (lam[-1]-2*distance_for_accelration)/lin_vel
    time_points = np.arange(0,2*time_for_accelration+constant_velocity_time,0.004)
    lam_stamped = np.piecewise(time_points, 
                        [time_points < time_for_accelration,
                            (time_points >= time_for_accelration) & (time_points <= time_for_accelration + constant_velocity_time),
                            time_points > time_for_accelration + constant_velocity_time],
                        [lambda t: 0.5*lin_acc*t**2,
                            lambda t: distance_for_accelration + lin_vel*(t - time_for_accelration),
                            lambda t: lam[-1] - 0.5*lin_acc*(2*time_for_accelration + constant_velocity_time - t)**2])
    acc_time = time_for_accelration

timestamp_lam_stamped = np.arange(0,len(lam_stamped))*0.004
plt.plot(timestamp_lam_stamped, lam_stamped)
plt.vlines(acc_time,0,lam[-1],colors='r',linestyles='dashed',label='End of Acceleration Phase')
plt.xlabel('Time (s)')
plt.ylabel('Path Length (units)')
plt.title('Path Length vs Time for Trapezoidal Velocity Profile')
plt.grid()
plt.show()

# create a fake Nx6 and Nx2 curve
curve_js = np.random.rand(len(lam),6)
curve_xy = np.random.rand(len(lam),2)
force_path = np.random.rand(len(lam))

traj_j = []
for j in range(len(curve_js[0])):
    traj_j.append(np.interp(lam_stamped, lam, curve_js[:,j]))
traj_q = np.array(traj_j).T
traj_xy = []
for i in range(len(curve_xy[0])):
    traj_xy.append(np.interp(lam_stamped, lam, curve_xy[:,i]))
traj_xy = np.array(traj_xy).T
traj_fz = np.interp(lam_stamped, lam, force_path)

plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.plot(timestamp_lam_stamped, traj_q)
plt.title('Joint Trajectories')
plt.xlabel('Time (s)')
plt.ylabel('Joint Angles (rad)')
plt.grid()
plt.subplot(3,1,2)
plt.plot(timestamp_lam_stamped, traj_xy)
plt.title('End-Effector Trajectories')
plt.xlabel('Time (s)')
plt.ylabel('Position (units)')
plt.grid()
plt.subplot(3,1,3)
plt.plot(timestamp_lam_stamped, traj_fz)
plt.title('Force Trajectory')
plt.xlabel('Time (s)')
plt.ylabel('Force (units)')
plt.grid()
plt.tight_layout()
plt.show()