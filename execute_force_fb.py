from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, time
from scipy.interpolate import CubicSpline
from general_robotics_toolbox import *
from copy import deepcopy
sys.path.append('toolbox')
from robot_def import *
from lambda_calc import *
from motion_toolbox import *

FORCE_PROTECTION = 5  # Force protection. Units: N

def spline_js(cartesian_path,curve_js,vd,rate=250):
    lam=calc_lam_cs(cartesian_path)
    # polyfit=np.polyfit(lam,curve_js,deg=40)
    # lam=np.linspace(0,lam[-1],int(rate*lam[-1]/vd))
    # return np.vstack((np.poly1d(polyfit[:,0])(lam), np.poly1d(polyfit[:,1])(lam), np.poly1d(polyfit[:,2])(lam), np.poly1d(polyfit[:,3])(lam), np.poly1d(polyfit[:,4])(lam), np.poly1d(polyfit[:,5])(lam))).T
    polyfit=CubicSpline(lam,curve_js)
    lam=np.linspace(0,lam[-1],int(rate*lam[-1]/vd))
    return polyfit(lam)

def adjoint_map(T):
    R=T.R
    p=T.p
    return np.vstack((np.hstack((R,np.zeros((3,3)))),np.hstack((hat(p)@R,R))))

class MotionController(object):
    def __init__(self,robot,ipad_pose,H_pentip2ati,controller_param):
        
        # define robots
        self.robot=robot
        # pose relationship
        self.ipad_pose=ipad_pose
        self.ipad_pose_T = Transform(self.ipad_pose[:3,:3],self.ipad_pose[:3,-1])
        self.H_pentip2ati=H_pentip2ati
        self.H_ati2pentip=np.linalg.inv(H_pentip2ati)
        self.ad_ati2pentip=adjoint_map(Transform(self.H_ati2pentip[:3,:3],self.H_ati2pentip[:3,-1]))
        self.ad_ati2pentip_T=self.ad_ati2pentip.T
        # controller parameters
        self.params=controller_param

        ####################################################FT Connection####################################################
        RR_ati_cli=RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor')
        #Connect a wire connection
        wrench_wire = RR_ati_cli.wrench_sensor_value.Connect()
        #Add callback for when the wire value change
        wrench_wire.WireValueChanged += self.wrench_wire_cb
        
        ##RR PARAMETERS
        self.RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot')
        # RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')
        self.RR_robot=self.RR_robot_sub.GetDefaultClientWait(1)
        self.robot_state = self.RR_robot_sub.SubscribeWire("robot_state")
        robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", self.RR_robot)
        self.halt_mode = robot_const["RobotCommandMode"]["halt"]
        self.position_mode = robot_const["RobotCommandMode"]["position_command"]
        self.RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",self.RR_robot)
        self.command_seqno = 1
        self.RR_robot.command_mode = self.halt_mode
        time.sleep(0.1)
  
        self.TIMESTEP = 0.004 # egm timestep 4 ms
        
    def connect_position_mode(self):
     
        ## connect to position command mode wire
        self.RR_robot.command_mode = self.position_mode
        self.cmd_w = self.RR_robot_sub.SubscribeWire("position_command")
    
    def wrench_wire_cb(self,w,value,time):

        self.ft_reading = np.array([value['torque']['x'],value['torque']['y'],value['torque']['z'],value['force']['x'],value['force']['y'],value['force']['z']])

    def force_impedence_ctrl(self,f_err):
     
        return self.params["force_ctrl_damping"]*f_err

    def force_load_z(self,fz_des):

        # transform to tip desired
        fz_des = fz_des*(-1)

        touch_t = None
        this_st=None
        ft_record=[]
        while True:
            # force reading
            ft_tip = self.ad_ati2pentip_T@self.ft_reading
            fz_now = float(ft_tip[-1])
            # tool pose reading
            q_now = self.robot_state.InValue.joint_position
            tip_now = self.robot.fwd(q_now)
            tip_now_ipad = self.ipad_pose_T.inv()*tip_now
            # time force joint angle record
            ft_record.append(np.append(np.array([time.time(),fz_now]),np.degrees(q_now)))
            # force control
            tip_next_ipad = deepcopy(tip_now_ipad)
            if np.abs(fz_now) < 0.5 and touch_t is None: # if not touch ipad
                tip_next_ipad.p[2] = tip_now_ipad.p[2] + -1*self.params['load_speed'] * self.TIMESTEP # move in -z direction
            else: # when touch ipad
                if touch_t is None:
                    touch_t=time.time()
                # track a trapziodal force profile
                this_fz_des = min(fz_des,(time.time()-touch_t)*self.params["trapzoid_slope"])
                f_err = this_fz_des-fz_now # feedback error
                v_des = self.force_impedence_ctrl(f_err) # force control
                tip_next_ipad.p[2] = tip_now_ipad.p[2] + v_des * self.TIMESTEP # force impedence control 
            
            # check if force achieved
            if np.fabs(fz_des-fz_now)<self.params['force_epsilon']:
                if (time.time()-set_time)>self.params['settling_time']:
                    break
            else:
                set_time = time.time()

            # get joint angles using ik
            tip_next = self.ipad_pose_T*tip_next_ipad
            # print(tip_next.p)
            q_des = robot.inv(tip_next.p,tip_next.R,q_now)[0]
            
            # send robot position command
            # if this_st is not None:
            # 	while time.time()-this_st<self.TIMESTEP-0.00001:
            # 		pass
            # this_st=time.time()
            self.position_cmd(q_des)

            if np.linalg.norm(ft_tip[3:])>FORCE_PROTECTION:
                print("force: ",ft_tip[3:])
                print("force too large")
                break
        
        return ft_record
   
    def trajectory_force_control(self,q_all,fz_des):

        # transform to tip desired
        fz_des = fz_des*(-1)
        # check if fz_des is a number or list
        if isinstance(fz_des,(int,float)):
            fz_des = np.ones(len(q_all))*fz_des
    
        # get path length
        lam = calc_lam_js(q_all,self.robot)

        # find the time for each segment
        time_bp=np.array(lam)/self.params['moveL_speed_lin']
        seg=1

        start_time=time.time()
        dt=0.004
        current_time=None
        ft_record=[]
        while time.time()-start_time<time_bp[-1]:
            ### get dt
            if current_time is not None:
                dt = time.time()-current_time
            current_time=time.time()
            
            ### find current segment
            if current_time-start_time>time_bp[seg]:
                seg+=1
            if seg==len(q_all):
                break
            frac=(current_time-start_time-time_bp[seg-1])/(time_bp[seg]-time_bp[seg-1])
            
            ### force feedback control
            # read force
            ft_tip = self.ad_ati2pentip_T@self.ft_reading
            fz_now = float(ft_tip[-1])
            # get current desired force
            fz_des_now = frac*fz_des[seg]+(1-frac)*fz_des[seg-1]
            f_err = fz_des_now-fz_now # feedback error
            v_des_z = self.force_impedence_ctrl(f_err) # force control
            if np.linalg.norm(ft_tip[3:])>FORCE_PROTECTION: # force protection break
                print("force: ",ft_tip[3:])
                print("force too large")
                break
            
            ### positoin control
            # joint reading
            q_now = self.robot_state.InValue.joint_position
            tip_now = self.robot.fwd(q_now)
            tip_now_ipad = self.ipad_pose_T.inv()*tip_now
            # get current desired position
            q_planned=frac*q_all[seg]+(1-frac)*q_all[seg-1]
            T_planned=self.robot.fwd(q_planned)
            T_planned_ipad=self.ipad_pose_T.inv()*T_planned
            # add force control
            T_planned_ipad.p[2] = tip_now_ipad.p[2] + v_des_z * dt # force impedence control
            T_planned=self.ipad_pose_T*T_planned_ipad
            q_des = self.robot.inv(T_planned.p,T_planned.R,q_now)[0]

            ### send robot position command
            self.position_cmd(q_des)
            
            # time, force, joint angle, record
            ft_record.append(np.append(np.array([time.time(),fz_now]),np.degrees(q_now)))
            
        return ft_record
    
    def jog_joint_position_cmd(self,q,v=0.4,wait_time=0):

        q_start=self.robot_state.InValue.joint_position
        total_time=np.linalg.norm(q-q_start)/v

        start_time=time.time()
        while time.time()-start_time<total_time:
            # Set the joint command
            frac=(time.time()-start_time)/total_time
            self.position_cmd(frac*q+(1-frac)*q_start)
            
        ###additional points for accuracy
        start_time=time.time()
        while time.time()-start_time<wait_time:
            self.position_cmd(q)
    
    def trajectory_position_cmd(self,q_all,v=0.4):

        lamq_bp=[0]
        for i in range(len(q_all)-1):
            lamq_bp.append(lamq_bp[-1]+np.linalg.norm(q_all[i+1]-q_all[i]))
        time_bp=np.array(lamq_bp)/v
        seg=1

        start_time=time.time()
        while time.time()-start_time<time_bp[-1]:

            #find current segment
            if time.time()-start_time>time_bp[seg]:
                seg+=1
            if seg==len(q_all):
                break
            frac=(time.time()-start_time-time_bp[seg-1])/(time_bp[seg]-time_bp[seg-1])
            self.position_cmd(frac*q_all[seg]+(1-frac)*q_all[seg-1])

    def position_cmd(self,q):

        # Increment command_seqno
        self.command_seqno += 1
        # Create Fill the RobotJointCommand structure
        joint_cmd = self.RobotJointCommand()
        joint_cmd.seqno = self.command_seqno # Strictly increasing command_seqno
        joint_cmd.state_seqno = self.robot_state.InValue.seqno # Send current robot_state.seqno as failsafe
        
        # Set the joint command
        joint_cmd.command = q

        # Send the joint command to the robot
        self.cmd_w.SetOutValueAll(joint_cmd)

def main():
    global  RobotJointCommand, cmd_w, command_seqno, robot_state, robot
    # img_name='wen_out'
    img_name='strokes_out'

    ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',')
    num_segments=len(glob.glob('path/cartesian_path/'+img_name+'/*.csv'))
    robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen.csv')
    # robot=robot_obj('ur5','config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')
    # print(robot.fwd(np.array([0,0,0,0,0,0])))
    # exit()

    ######## FT sensor info
    H_pentip2ati=np.loadtxt('config/pentip2ati.csv',delimiter=',')
 
    ######## Controller parameters ###
    controller_params = {
        "force_ctrl_damping": 40.0,
        "force_epsilon": 0.1, # Unit: N
        "moveL_speed_lin": 10.0, # Unit: mm/sec
        "moveL_speed_ang": np.radians(10), # Unit: rad/sec
        "trapzoid_slope": 1, # trapzoidal load profile. Unit: N/sec
        "load_speed": 50.0, # Unit mm/sec
        "unload_speed": 1.0, # Unit mm/sec
        'settling_time': 1 # Unit: sec
        }
    
    ######## Motion Controller ###
    mctrl = MotionController(robot,ipad_pose,H_pentip2ati,controller_params)
    mctrl.connect_position_mode()

    F_MAX=0	#maximum pushing force 10N
    F_des = 2 # desired force unit: N

    start=True
    ft_record_load=[]
    ft_record_move=[]
    for i in range(num_segments):
        print('Segment %i'%i)
        pixel_path=np.loadtxt('path/pixel_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
        cartesian_path=np.loadtxt('path/cartesian_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
        curve_js=np.loadtxt('path/js_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,6))
        force_path=np.loadtxt('force_path/'+img_name+'/%i.csv'%i,delimiter=',')
        if len(curve_js)>1:

            pose_start=robot.fwd(curve_js[0])
            if start:
                h_offset = 10
                #jog to starting point
                p_start=pose_start.p+h_offset*ipad_pose[:3,-2]
                q_start=robot.inv(p_start,pose_start.R,curve_js[0])[0]
                mctrl.jog_joint_position_cmd(q_start,wait_time=0.5)
                ft_record = mctrl.force_load_z(F_des)
                # clear bias
                start=False
            else:
                h_offset = 10
                #arc-like trajectory to next segment
                p_start=pose_start.p+h_offset*ipad_pose[:3,-2]
                q_start=robot.inv(p_start,pose_start.R,curve_js[0])[0]
                pose_cur=robot.fwd(mctrl.robot_state.InValue.joint_position)
                p_mid=(pose_start.p+pose_cur.p)/2+h_offset*ipad_pose[:3,-2]
                q_mid=robot.inv(p_mid,pose_start.R,curve_js[0])[0]
                
                mctrl.trajectory_position_cmd(np.vstack((mctrl.robot_state.InValue.joint_position,q_mid,q_start)),v=0.2)
                mctrl.jog_joint_position_cmd(q_start,wait_time=0.3)
                ft_record=mctrl.force_load_z(F_des)
            
            ft_record_load.append(ft_record)

            traversal_velocity=50

            #drawing trajectory
            print("Travel trajectory")
            # constant force profile
            # ft_record=mctrl.trajectory_force_control(curve_js,F_des)
            # with path force profile
            ft_record=mctrl.trajectory_force_control(curve_js,force_path)
            ft_record_move.append(ft_record)

            mctrl.jog_joint_position_cmd(q_start,wait_time=0.5)
    
    #jog to end point
    pose_end=robot.fwd(curve_js[-1])
    p_end=pose_end.p+20*ipad_pose[:3,-2]
    q_end=robot.inv(p_end,pose_end.R,curve_js[-1])[0]
    mctrl.jog_joint_position_cmd(q_end)

    for i in range(len(ft_record_load)):
        ft_record_load[i]=np.array(ft_record_load[i])
        ft_record_move[i]=np.array(ft_record_move[i])
        np.savetxt('record/ft_record_load_%i.csv'%i,ft_record_load[i],delimiter=',')
        np.savetxt('record/ft_record_move_%i.csv'%i,ft_record_move[i],delimiter=',')

if __name__ == '__main__':
    main()