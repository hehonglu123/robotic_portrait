import numpy as np
import matplotlib.pyplot as plt
import glob, cv2, sys
sys.path.append('toolbox')
from robot_def import *

def main():
    pixel2mm=0.5
    img_name='me_out'
    image=cv2.imread('imgs/'+img_name+'.png')
    image_center=np.array([image.shape[1]/2,image.shape[0]/2])
    num_segments=len(glob.glob('path/cartesian_path/'+img_name+'/*.csv'))

    robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen.csv')

    R_pencil=np.array([ [-1,0,0],
                        [0,1,0],
                        [0,0,-1]])
    for i in range(num_segments):
        cartesian_path=np.loadtxt('path/cartesian_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
        curve_js=robot.find_curve_js(cartesian_path,[R_pencil]*len(cartesian_path),np.zeros(6))

        np.savetxt('path/js_path/'+img_name+'/%i.csv'%i, curve_js, delimiter=',')
        
if __name__ == '__main__':
    main()