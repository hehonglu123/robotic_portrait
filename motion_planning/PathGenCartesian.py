import numpy as np
import matplotlib.pyplot as plt
import glob, cv2
from pathlib import Path

CODE_PATH = '../'

def image2plane(img,ipad_pose,pixel2mm,pixel_paths,pixel2force):
    ###convert image pixel path to cartesian path
    #img: single channel gray scale image
    #ipad_pose: 4x4 pose matrix of ipad
    #pixel2mm: ratio of pixel to mm
    #pixel_paths: list of pixel paths
    #force_gain: gain adjust position command according to width ratio, gain=mm/ratio, 0 means push down, 1 means 1mm up when ratio=0
    
    ###find a ratio to fit image in paper
    image_center=np.array([img.shape[1]/2,img.shape[0]/2])
    cartesian_paths=[]
    cartesian_paths_world=[]
    force_paths=[]
    for pixel_path in pixel_paths:
        cartesian_path=(pixel_path[:,:2]-image_center)*pixel2mm
        #convert pixel frame to world frame first
        cartesian_path=np.dot(np.array([[0,-1],[-1,0]]),cartesian_path.T).T
        #append 0 in z
        cartesian_path=np.hstack((cartesian_path,np.zeros((len(cartesian_path),1))))
        #convert to world frame
        cartesian_path_world=np.dot(ipad_pose[:3,:3],cartesian_path.T).T+np.tile(ipad_pose[:3,-1],(len(cartesian_path),1))

        ##convert to force
        force_path=pixel2force[0]*pixel_path[:,2]+pixel2force[1]

        cartesian_paths.append(cartesian_path)
        cartesian_paths_world.append(cartesian_path_world)
        force_paths.append(force_path)
    
    return cartesian_paths, cartesian_paths_world, force_paths

def main():
    # img_name='eric_name_out'
    # img_name='wen_name_out'
    img_name='logos_words'
    # img_name='me_out'
    # img_name='new_year_out'
    # img_name='ilc_path1'
    num_segments=len(glob.glob(CODE_PATH+'path/pixel_path/'+img_name+'/*.csv'))
    pixel_paths=[]
    for i in range(num_segments):
        pixel_paths.append(np.loadtxt(CODE_PATH+'path/pixel_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    img=cv2.imread(CODE_PATH+'imgs/'+img_name+'_resized.png')
    ipad_pose=np.loadtxt(CODE_PATH+'config/ipad_pose.csv',delimiter=',')
    paper_size=np.loadtxt(CODE_PATH+'config/paper_size.csv',delimiter=',')
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## load pixel to mm parameters
    pixel2mm=np.loadtxt(CODE_PATH+'config/pixel2mm.csv',delimiter=',')
    ## load width to force parameters
    pixel2force=np.loadtxt(CODE_PATH+'config/pixel2force.csv',delimiter=',')
    
    print(paper_size,img_gray.shape,pixel2mm)
    
    cartesian_paths,cartesian_paths_world,force_paths=image2plane(img,ipad_pose,pixel2mm,pixel_paths,pixel2force)
    for i in range(len(cartesian_paths_world)):

        ###plot out the path in 3D
        ax.plot(cartesian_paths_world[i][:,0], cartesian_paths_world[i][:,1], cartesian_paths_world[i][:,2], 'b')
        ###save path
        Path(CODE_PATH+'path/cartesian_path/'+img_name).mkdir(parents=True, exist_ok=True)
        Path(CODE_PATH+'path/force_path/'+img_name).mkdir(parents=True, exist_ok=True)
        np.savetxt(CODE_PATH+'path/cartesian_path/'+img_name+'/%i.csv'%i,cartesian_paths_world[i],delimiter=',')
        np.savetxt(CODE_PATH+'path/force_path/'+img_name+'/%i.csv'%i,force_paths[i],delimiter=',')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
        


if __name__ == '__main__':
    main()