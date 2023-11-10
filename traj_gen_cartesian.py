import numpy as np
import matplotlib.pyplot as plt
import glob, cv2

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',')
    pixel2mm=0.3
    img_name='me_out'
    image=cv2.imread('imgs/'+img_name+'.png')
    image_center=np.array([image.shape[1]/2,image.shape[0]/2])
    num_segments=len(glob.glob('path/pixel_path/'+img_name+'/*.csv'))
    for i in range(num_segments):
        pixel_path=np.loadtxt('path/pixel_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,2))
        cartesian_path=(pixel_path-image_center)*pixel2mm
        print(cartesian_path)
        cartesian_path=np.hstack((cartesian_path,np.zeros((len(cartesian_path),1))))
        cartesian_path=np.dot(ipad_pose[:3,:3],cartesian_path.T).T+np.tile(ipad_pose[:3,-1],(len(cartesian_path),1))
        np.savetxt('path/cartesian_path/'+img_name+'/%i.csv'%i, cartesian_path, delimiter=',')

        ###plot out the path in 3D
        ax.plot(cartesian_path[:,0], cartesian_path[:,1], cartesian_path[:,2], 'b.')

    plt.show()
        


if __name__ == '__main__':
    main()