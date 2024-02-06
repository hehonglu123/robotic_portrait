import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob, cv2, sys
sys.path.append('toolbox')
from robot_def import *

def main():
    # img_name='wen_out'
    # img_name='strokes_out'
    img_name='eric_name_out'
    image=cv2.imread('imgs/'+img_name+'.png')
    image_center=np.array([image.shape[1]/2,image.shape[0]/2])
    num_segments=len(glob.glob('path/cartesian_path/'+img_name+'/*.csv'))
    ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',')


    robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen.csv')

    # robot=robot_obj('ur5','config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')

    R_pencil=ipad_pose[:3,:3]@Ry(np.pi)
    print(R_pencil)
    for i in range(num_segments):
        cartesian_path=np.loadtxt('path/cartesian_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))

        q_seed=np.radians([0,-54.8,110,-142,-90,0]) if 'ur' in robot.robot_name.lower() else np.zeros(6)
        curve_js=robot.find_curve_js(cartesian_path,[R_pencil]*len(cartesian_path),q_seed)

        Path('path/js_path/'+img_name).mkdir(parents=True, exist_ok=True)
        np.savetxt('path/js_path/'+img_name+'/%i.csv'%i, curve_js, delimiter=',')

        print(cartesian_path[0],robot.fwd(curve_js[0]).p)
        
if __name__ == '__main__':
    main()