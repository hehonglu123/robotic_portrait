import numpy as np
from matplotlib import pyplot as plt

CODE_PATH = '../'
# data_dir=CODE_PATH+'record/'
traj_xyf=np.loadtxt(data_dir+'traj_xyf.csv',delimiter=',')
traj_input_iter=[]
traj_exe_iter=[]
traj_G_iter=[]
iter_n=0
while True:
    try:
        traj_input_iter.append(np.loadtxt(data_dir+'iter_%i_traj_xyf_input.csv'%iter_n,delimiter=','))
        traj_exe_iter.append(np.loadtxt(data_dir+'iter_%i_xyz_exe.csv'%iter_n,delimiter=','))
        traj_G_iter.append(np.loadtxt(data_dir+'iter_%i_G_error.csv'%iter_n,delimiter=','))
    except:
        break
    iter_n+=1
    # if iter_n>5:
    #     break

# Plotting all iterations of x, y, and f in one plot using subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

time_t = np.arange(traj_xyf.shape[0])*0.004
# # Plotting all iterations of x
# axs[0].set_title('Trajectory - X')
# axs[0].plot(time_t,traj_xyf[:, 0], label='Reference')
# for i in range(iter_n):
#     axs[0].plot(time_t,traj_exe_iter[i][:, 0], label='Iteration %i' % i)
# axs[0].set_xlabel('Time')
# axs[0].set_ylabel('X')
# axs[0].legend()
# # Plotting all iteration of G x
# axs[1].set_title('Gradient of X')
# for i in range(iter_n):
#     axs[1].plot(time_t,traj_G_iter[i][:, 0], label='Iteration %i' % i)
# axs[1].set_xlabel('Time')
# axs[1].set_ylabel('Gradient of X')
# axs[1].legend()

# # Plotting all iterations of y
# axs[2].set_title('Trajectory - Y')
# axs[2].plot(time_t,traj_xyf[:, 1], label='Reference')
# for i in range(iter_n):
#     axs[2].plot(time_t,traj_exe_iter[i][:, 1], label='Iteration %i' % i)
# axs[2].set_xlabel('Time')
# axs[2].set_ylabel('Y')
# axs[2].legend()
# # Plotting all iteration of G y
# axs[3].set_title('Gradient of Y')
# for i in range(iter_n):
#     axs[3].plot(time_t,traj_G_iter[i][:, 1], label='Iteration %i' % i)
# axs[3].set_xlabel('Time')
# axs[3].set_ylabel('Gradient of Y')
# axs[3].legend()

# Plotting all iterations of f
axs[0].set_title('Trajectory Input - F')
axs[0].plot(time_t,traj_xyf[:, 2], label='Reference')
for i in range(iter_n):
    axs[0].plot(time_t,traj_exe_iter[i][:, 2], label='Iteration %i' % i)
axs[0].set_xlabel('Time')
axs[0].set_ylabel('F')
axs[0].legend()
# Plotting all iteration of G f
axs[1].set_title('Gradient of F')
for i in range(iter_n):
    axs[1].plot(time_t,traj_G_iter[i][:, 2], label='Iteration %i' % i)
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Gradient of F')
axs[1].legend()

plt.tight_layout()
plt.show()
