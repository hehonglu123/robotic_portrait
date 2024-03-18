import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob, cv2, sys

CODE_PATH='../'
sys.path.append(CODE_PATH+'toolbox')
from robot_def import *

def main():
    # img_name='wen_out'
    # img_name='strokes_out'
    # img_name='wen_name_out'
    # img_name='me_out'
    # img_name='new_year_out'
    img_name='ilc_path1'
    image=cv2.imread(CODE_PATH+'imgs/'+img_name+'.png')
    image_center=np.array([image.shape[1]/2,image.shape[0]/2])
    num_segments=len(glob.glob(CODE_PATH+'path/cartesian_path/'+img_name+'/*.csv'))
    ipad_pose=np.loadtxt(CODE_PATH+'config/ipad_pose.csv',delimiter=',')


    robot=robot_obj('ABB_1200_5_90',CODE_PATH+'config/ABB_1200_5_90_robot_default_config.yml',tool_file_path=CODE_PATH+'config/heh6_pen.csv')

    # robot=robot_obj('ur5','config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')

    R_pencil=ipad_pose[:3,:3]@Ry(np.pi)
    print(R_pencil)
    for i in range(num_segments):
        cartesian_path=np.loadtxt(CODE_PATH+'path/cartesian_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))

        q_seed=np.radians([0,-54.8,110,-142,-90,0]) if 'ur' in robot.robot_name.lower() else np.zeros(6)
        curve_js=robot.find_curve_js(cartesian_path,[R_pencil]*len(cartesian_path),q_seed)

        Path(CODE_PATH+'path/js_path/'+img_name).mkdir(parents=True, exist_ok=True)
        np.savetxt(CODE_PATH+'path/js_path/'+img_name+'/%i.csv'%i, curve_js, delimiter=',')

        print(cartesian_path[0],robot.fwd(curve_js[0]).p)
        
if __name__ == '__main__':
    main()