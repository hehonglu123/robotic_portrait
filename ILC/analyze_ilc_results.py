import numpy as np
from matplotlib import pyplot as plt

folder_dir='results/'
show_N_only = 3
title_fontsize=20
label_fontsize=15
legend_fontsize=15
tick_fontsize=15

data_sets = ['record_0305_gain60/', 'record_0305_gain30/', 'record_0305_gain60_fonly/', 'record_0305_path2_gain60/']
data_sets_names = ['Damping 1/60', 'Damping 1/30', 'Damping 1/60 Force only', 'Path 2 Damping 1/60']
show_datasets = ['record_0305_gain60/', 'record_0305_path2_gain60/']

error_x_norm_datasets = []
error_y_norms_datasets = []
error_f_norms_datasets = []
for data in data_sets:

    data_dir = folder_dir+data
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
    
    # get error norm
    error_x_norm=[]
    error_y_norm=[]
    error_f_norm=[]
    all_error_f = []
    for iter_i in range(iter_n):
        error_x = traj_exe_iter[iter_i][:,0]-traj_xyf[:,0]
        error_y = traj_exe_iter[iter_i][:,1]-traj_xyf[:,1]
        error_f = traj_exe_iter[iter_i][:,2]-traj_xyf[:,2]
        error_x_norm.append(np.linalg.norm(error_x))
        error_y_norm.append(np.linalg.norm(error_y))
        error_f_norm.append(np.linalg.norm(error_f))
        all_error_f.append(error_f)
    error_x_norm_datasets.append(error_x_norm)
    error_y_norms_datasets.append(error_y_norm)
    error_f_norms_datasets.append(error_f_norm)
    
    if data in show_datasets:
        # plot f error in xy plane
        # Plotting the last all_error_f heatmap on traj_exe_iter x y plane
        for i in [0,-1]:
            fig, ax = plt.subplots(figsize=(8, 6))
            if i==0:
                ax.set_title('Error Heatmap - X vs Y (First Iteration)', fontsize=title_fontsize)
            else:
                ax.set_title('Error Heatmap - X vs Y (Last Iteration)', fontsize=title_fontsize)
            ax.set_xlabel('X (mm)', fontsize=label_fontsize)
            ax.set_ylabel('Y (mm)', fontsize=label_fontsize)
            ax.tick_params(labelsize=tick_fontsize)
            ax.scatter(traj_exe_iter[i][:, 1]*-1, traj_exe_iter[i][:, 0], c=np.fabs(all_error_f[i]), cmap='hot', s=10)
            cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
            ax.set_aspect('equal')  # Fix x-y ratio
            plt.show()
        
        jump_n = int(iter_n/show_N_only)
        draw_n = np.arange(0,iter_n,jump_n)
        if iter_n-1 not in draw_n:
            draw_n = np.append(draw_n, iter_n-1)

        time_t = np.arange(traj_xyf.shape[0])*0.004

        # Plotting all iterations of x
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        axs[0].set_title('Trajectory Ouput - X', fontsize=title_fontsize)
        axs[0].plot(time_t,traj_xyf[:, 0], label='Reference', color='black')
        for i in draw_n:
            axs[0].plot(time_t,traj_exe_iter[i][:, 0], label='Iteration %i' % i)
        axs[0].set_xlabel('Time (sec)', fontsize=label_fontsize)
        axs[0].set_ylabel('mm', fontsize=label_fontsize)
        axs[0].xaxis.set_tick_params(labelsize=tick_fontsize)
        axs[0].yaxis.set_tick_params(labelsize=tick_fontsize)
        axs[0].legend(fontsize=legend_fontsize)
        axs[1].set_title('Input - X', fontsize=title_fontsize)
        axs[1].plot(time_t,traj_xyf[:, 0], label='Reference', color='black')
        for i in draw_n:
            axs[1].plot(time_t,traj_input_iter[i][:, 0], label='Iteration %i' % i)
        axs[1].set_xlabel('Time (sec)', fontsize=label_fontsize)
        axs[1].set_ylabel('mm', fontsize=label_fontsize)
        axs[1].xaxis.set_tick_params(labelsize=tick_fontsize)
        axs[1].yaxis.set_tick_params(labelsize=tick_fontsize)
        axs[1].legend(fontsize=legend_fontsize)
        # Plotting all iteration of G x
        axs[2].set_title('Gradient of X', fontsize=title_fontsize)
        for i in draw_n:
            axs[2].plot(time_t,traj_G_iter[i][:, 0], label='Iteration %i' % i)
        axs[2].set_xlabel('Time (sec)', fontsize=label_fontsize)
        axs[2].set_ylabel('Gradient of X', fontsize=label_fontsize)
        axs[2].xaxis.set_tick_params(labelsize=tick_fontsize)
        axs[2].yaxis.set_tick_params(labelsize=tick_fontsize)
        axs[2].legend(fontsize=legend_fontsize)
        plt.tight_layout()
        plt.show()

        # Plotting all iterations of y
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        axs[0].set_title('Trajectory - Y', fontsize=title_fontsize)
        axs[0].plot(time_t,traj_xyf[:, 1], label='Reference', color='black')
        for i in draw_n:
            axs[0].plot(time_t,traj_exe_iter[i][:, 1], label='Iteration %i' % i)
        axs[0].set_xlabel('Time (sec)', fontsize=label_fontsize)
        axs[0].set_ylabel('mm', fontsize=label_fontsize)
        axs[0].xaxis.set_tick_params(labelsize=tick_fontsize)
        axs[0].yaxis.set_tick_params(labelsize=tick_fontsize)
        axs[0].legend(fontsize=legend_fontsize)
        axs[1].set_title('Input - Y', fontsize=title_fontsize)
        axs[1].plot(time_t,traj_xyf[:, 1], label='Reference', color='black')
        for i in draw_n:
            axs[1].plot(time_t,traj_input_iter[i][:, 1], label='Iteration %i' % i)
        axs[1].set_xlabel('Time (sec)', fontsize=label_fontsize)
        axs[1].set_ylabel('mm', fontsize=label_fontsize)
        axs[1].xaxis.set_tick_params(labelsize=tick_fontsize)
        axs[1].yaxis.set_tick_params(labelsize=tick_fontsize)
        axs[1].legend(fontsize=legend_fontsize)
        # Plotting all iteration of G y
        axs[2].set_title('Gradient of Y', fontsize=title_fontsize)
        for i in draw_n:
            axs[2].plot(time_t,traj_G_iter[i][:, 1], label='Iteration %i' % i)
        axs[2].set_xlabel('Time (sec)', fontsize=label_fontsize)
        axs[2].set_ylabel('Gradient of Y', fontsize=label_fontsize)
        axs[2].xaxis.set_tick_params(labelsize=tick_fontsize)
        axs[2].yaxis.set_tick_params(labelsize=tick_fontsize)
        axs[2].legend(fontsize=legend_fontsize)
        plt.tight_layout()
        plt.show()

        # Plotting all iterations of f
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        axs[0].set_title('Trajectory - F', fontsize=title_fontsize)
        axs[0].plot(time_t,traj_xyf[:, 2], label='Reference', color='black')
        for i in draw_n:
            axs[0].plot(time_t,traj_exe_iter[i][:, 2], label='Iteration %i' % i)
        axs[0].set_xlabel('Time (sec)', fontsize=label_fontsize)
        axs[0].set_ylabel('N', fontsize=label_fontsize)
        axs[0].xaxis.set_tick_params(labelsize=tick_fontsize)
        axs[0].yaxis.set_tick_params(labelsize=tick_fontsize)
        axs[0].legend(fontsize=legend_fontsize)
        axs[1].set_title('Input - F', fontsize=title_fontsize)
        axs[1].plot(time_t,traj_xyf[:, 2], label='Reference', color='black')
        for i in draw_n:
            axs[1].plot(time_t,traj_input_iter[i][:, 2], label='Iteration %i' % i)
        axs[1].set_xlabel('Time (sec)', fontsize=label_fontsize)
        axs[1].set_ylabel('N', fontsize=label_fontsize)
        axs[1].xaxis.set_tick_params(labelsize=tick_fontsize)
        axs[1].yaxis.set_tick_params(labelsize=tick_fontsize)
        axs[1].legend(fontsize=legend_fontsize)
        # Plotting all iteration of G f
        axs[2].set_title('Gradient of F', fontsize=title_fontsize)
        for i in draw_n:
            axs[2].plot(time_t,traj_G_iter[i][:, 2], label='Iteration %i' % i)
        axs[2].set_xlabel('Time (sec)', fontsize=label_fontsize)
        axs[2].set_ylabel('N', fontsize=label_fontsize)
        axs[2].xaxis.set_tick_params(labelsize=tick_fontsize)
        axs[2].yaxis.set_tick_params(labelsize=tick_fontsize)
        axs[2].legend(fontsize=legend_fontsize)
        plt.tight_layout()
        plt.show()

# Plotting error norm datasets
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
axs[0].set_title('Error Norm - X', fontsize=title_fontsize)
for i, error_x_norm in enumerate(error_x_norm_datasets):
    axs[0].plot(range(len(error_x_norm)), error_x_norm, label=data_sets_names[i], marker='o')
axs[0].set_xlabel('Iteration', fontsize=label_fontsize)
axs[0].set_ylabel('Error Norm', fontsize=label_fontsize)
axs[0].legend(fontsize=legend_fontsize)

axs[1].set_title('Error Norm - Y', fontsize=title_fontsize)
for i, error_y_norm in enumerate(error_y_norms_datasets):
    axs[1].plot(range(len(error_y_norm)), error_y_norm, label=data_sets_names[i], marker='o')
axs[1].set_xlabel('Iteration', fontsize=label_fontsize)
axs[1].set_ylabel('Error Norm', fontsize=label_fontsize)
axs[1].legend(fontsize=legend_fontsize)

axs[2].set_title('Error Norm - F', fontsize=title_fontsize)
for i, error_f_norm in enumerate(error_f_norms_datasets):
    axs[2].plot(range(len(error_f_norm)), error_f_norm, label=data_sets_names[i], marker='o')
axs[2].set_xlabel('Iteration', fontsize=label_fontsize)
axs[2].set_ylabel('Error Norm', fontsize=label_fontsize)
axs[2].legend(fontsize=legend_fontsize)

plt.tight_layout()
plt.show()