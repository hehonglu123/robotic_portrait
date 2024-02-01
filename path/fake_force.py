import numpy as np

start_force = 0.2  # Starting force. Units: N
start_length = 2  # Starting length
end_force = 1  # Ending force. Units: N
end_length = 2  # Ending length

### linear up #######
seg_i=2
# read path from file
path = np.loadtxt("cartesian_path/strokes_out/"+str(seg_i)+".csv", delimiter=",")
path_length = len(path)
# generate force profile
force_profile = np.linspace(start_force, end_force, path_length-start_length-end_length)
force_profile = np.append(np.ones(start_length)*start_force, force_profile)
force_profile = np.append(force_profile, np.ones(end_length)*end_force)
# save path with force
np.savetxt("force_path/strokes_out/"+str(seg_i)+".csv", force_profile, delimiter=",")

### linear down #######
seg_i=0
# read path from file
path = np.loadtxt("cartesian_path/strokes_out/"+str(seg_i)+".csv", delimiter=",")
path_length = len(path)
# generate force profile
force_profile = np.linspace(end_force, start_force, path_length-start_length-end_length)
force_profile = np.append(np.ones(start_length)*end_force, force_profile)
force_profile = np.append(force_profile, np.ones(end_length)*start_force)
# save path with force
np.savetxt("force_path/strokes_out/"+str(seg_i)+".csv", force_profile, delimiter=",")

### linear up and down, 2 segments #######
seg_i=1
# read path from file
path = np.loadtxt("cartesian_path/strokes_out/"+str(seg_i)+".csv", delimiter=",")
path_length = len(path)
# parameters
up_length = 10
down_length = 10
# generate force profile
force_profile = np.linspace(start_force, end_force, up_length)
force_profile = np.append(force_profile,np.linspace(end_force, start_force, down_length))
force_profile = np.append(force_profile,np.linspace(start_force, end_force, up_length))
force_profile = np.append(force_profile,np.linspace(end_force, start_force, path_length-len(force_profile)))
# save path with force
np.savetxt("force_path/strokes_out/"+str(seg_i)+".csv", force_profile, delimiter=",") 

