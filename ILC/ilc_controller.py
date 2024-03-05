from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, time
from scipy.interpolate import CubicSpline
from general_robotics_toolbox import *
from abb_robot_client.egm import EGM
from copy import deepcopy
from pathlib import Path

CODE_PATH = '../'
sys.path.append(CODE_PATH+'toolbox')
from robot_def import *
from lambda_calc import *
from motion_toolbox import *
from utils import *

FORCE_PROTECTION = 5  # Force protection. Units: N
SHOW_STATUS = False
PROGRESS_VIZ = False

USE_RR_ROBOT = False

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

class MotionILController(object):
    def __init__(self,robot,ipad_pose,H_pentip2ati,controller_param):
        
        # define robots
        self.robot=robot
        # pose relationship
        self.ipad_pose=ipad_pose
        self.ipad_pose_T = Transform(self.ipad_pose[:3,:3],self.ipad_pose[:3,-1])
        self.ipad_pose_inv_T = self.ipad_pose_T.inv()
        self.ad_ipad2base = adjoint_map(self.ipad_pose_T.inv())
        self.H_pentip2ati=H_pentip2ati
        self.H_ati2pentip=np.linalg.inv(H_pentip2ati)
        self.ad_ati2pentip=adjoint_map(Transform(self.H_ati2pentip[:3,:3],self.H_ati2pentip[:3,-1]))
        self.ad_ati2pentip_T=self.ad_ati2pentip.T
        # controller parameters
        self.params=controller_param

        #################### FT Connection ####################
        self.RR_ati_cli=RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor')
        #Connect a wire connection
        wrench_wire = self.RR_ati_cli.wrench_sensor_value.Connect()
        #Add callback for when the wire value change
        wrench_wire.WireValueChanged += self.wrench_wire_cb
        
        ################ Robot Connection ################
        self.command_seqno = 1
        if USE_RR_ROBOT:
            ##RR PARAMETERS
            self.RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot')
            # RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')
            self.RR_robot=self.RR_robot_sub.GetDefaultClientWait(1)
            self.robot_state = self.RR_robot_sub.SubscribeWire("robot_state")
            time.sleep(0.1)

            robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", self.RR_robot)
            self.halt_mode = robot_const["RobotCommandMode"]["halt"]
            self.position_mode = robot_const["RobotCommandMode"]["position_command"]
            self.RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",self.RR_robot)
            self.RR_robot.command_mode = self.halt_mode
            time.sleep(0.1)
            self.connect_position_mode()
        else: # use EGM else
            self.egm = EGM()
  
        self.TIMESTEP = 0.004 # egm timestep 4 ms
        self.params['lookahead_index'] = int(self.params['lookahead_time']/self.TIMESTEP)
        
    def connect_position_mode(self):
     
        ## connect to position command mode wire
        self.RR_robot.command_mode = self.position_mode
        self.cmd_w = self.RR_robot_sub.SubscribeWire("position_command")
    
    def position_cmd(self,q):

        # Increment command_seqno
        self.command_seqno += 1
        
        if USE_RR_ROBOT:
            # Create Fill the RobotJointCommand structure
            joint_cmd = self.RobotJointCommand()
            joint_cmd.seqno = self.command_seqno # Strictly increasing command_seqno
            joint_cmd.state_seqno = self.robot_state.InValue.seqno # Send current robot_state.seqno as failsafe
            # Set the joint command
            joint_cmd.command = q
            # Send the joint command to the robot
            self.cmd_w.SetOutValueAll(joint_cmd)
        else:
            self.egm.send_to_robot(np.degrees(q))
    
    def read_position(self):
        if USE_RR_ROBOT:
            return self.robot_state.InValue.joint_position
        else:
            res, state = self.egm.receive_from_robot(timeout=0.1)
            if not res:
                raise Exception("Robot communication lost")
            return np.radians(state.joint_angles)
    
    def drain_egm(self, read_N):

        for i in range(read_N):
            self.read_position()
    
    def wrench_wire_cb(self,w,value,time):

        self.ft_reading = np.array([value['torque']['x'],value['torque']['y'],value['torque']['z'],value['force']['x'],value['force']['y'],value['force']['z']])

    def force_impedence_ctrl(self,f_err):
     
        return self.params["force_ctrl_damping"]*f_err

    def force_load_z(self,fz_des):

        # # transform to tip desired
        # fz_des = fz_des*(-1)

        touch_t = None
        this_st=None
        set_time=None
        ft_record=[]
        while True:
            # tool pose reading
            q_now = self.read_position()
            tip_now = self.robot.fwd(q_now)
            tip_now_ipad = self.ipad_pose_T.inv()*tip_now
            # force reading
            ft_tip = self.ad_ati2pentip_T@self.ft_reading
            fz_now = float(ft_tip[-1])
            # force protection
            if np.linalg.norm(ft_tip[3:])>FORCE_PROTECTION:
                print("force: ",ft_tip[3:])
                print("force too large")
                break
            # time force joint angle record
            ft_record.append(np.append(np.array([time.time(),fz_now]),np.degrees(q_now)))
            # force control
            tip_next_ipad = deepcopy(tip_now_ipad)
            if np.abs(fz_now) < self.params['force_epsilon'] and touch_t is None: # if not touch ipad
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
                if set_time is None:
                    set_time = time.time()
                if (time.time()-set_time)>self.params['settling_time']:
                    break
            else:
                set_time = time.time()

            # get joint angles using ik
            tip_next = self.ipad_pose_T*tip_next_ipad
            # print(tip_next.p)
            q_des = self.robot.inv(tip_next.p,tip_next.R,q_now)[0]
            
            # send robot position command
            # if this_st is not None:
            # 	while time.time()-this_st<self.TIMESTEP-0.00001:
            # 		pass
            # this_st=time.time()
            self.position_cmd(q_des)
        
        return ft_record
   
    def motion_start_procedure(self, js_start, f_start, h_offset, h_offset_low, lin_vel=2):
        
        # jog to start point
        pose_start=self.robot.fwd(js_start)
        #arc-like trajectory to next segment
        p_start=pose_start.p+h_offset*self.ipad_pose[:3,-2]
        q_start=self.robot.inv(p_start,pose_start.R,js_start)[0]
        pose_cur=self.robot.fwd(self.read_position())
        p_mid=(pose_start.p+pose_cur.p)/2+h_offset*self.ipad_pose[:3,-2]
        q_mid=self.robot.inv(p_mid,pose_start.R,js_start)[0]
        self.trajectory_position_cmd(np.vstack((self.read_position(),q_mid,q_start)),v=lin_vel)
        p_start=pose_start.p+h_offset_low*self.ipad_pose[:3,-2]
        q_start=self.robot.inv(p_start,pose_start.R,js_start)[0]
        self.jog_joint_position_cmd(q_start,v=lin_vel,wait_time=0.5)
        # set tare before load force
        self.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
        if SHOW_STATUS:
            input("Set tare. Start load force?")
        print("Start load force")
        # load force
        ft_record=self.force_load_z(f_start)
        print("Load force done")
        
        return ft_record
    
    def motion_end_procedure(self, js_end, h_offset, lin_vel=2):
        # jog to end point z+hoffset
        pose_end=self.robot.fwd(js_end)
        p_end=pose_end.p+h_offset*self.ipad_pose[:3,-2]
        q_end=self.robot.inv(p_end,pose_end.R,js_end)[0]
        self.jog_joint_position_cmd(q_end,v=lin_vel,wait_time=0.5)
    
    def trajectory_generate(self,curve_js,curve_xy,force_path,lin_vel=None,lin_acc=None):
    
        # velocity and acceleration
        if lin_vel is None:
            lin_vel = self.params['moveL_speed_lin']
        if lin_acc is None:
            lin_acc = self.params['moveL_acc_lin']
        # Calculate the path length
        lam = calc_lam_js(curve_js, self.robot)
        
        # find the time for each segment, with acceleration and deceleration
        if len(lam)>2 and lin_acc>0:
            time_bp = np.zeros_like(lam)
            acc = lin_acc
            vel = 0
            for i in range(0,len(lam)):
                if vel>=lin_vel:
                    time_bp[i] = time_bp[i-1]+(lam[i]-lam[i-1])/lin_vel
                else:
                    time_bp[i] = np.sqrt(2*lam[i]/acc)
                    vel = acc*time_bp[i]
            time_bp_half = []
            vel = 0
            for i in range(len(lam)-1,-1,-1):
                if vel>=lin_vel or i<=len(lam)/2:
                    break
                else:
                    time_bp_half.append(np.sqrt(2*(lam[-1]-lam[i])/acc))
                    vel = acc*time_bp_half[-1]
            time_bp_half = np.array(time_bp_half)[::-1]
            time_bp_half = time_bp_half*-1+time_bp_half[0]
            time_bp[-len(time_bp_half):] = time_bp[-len(time_bp_half)-1]+time_bp_half\
                +(lam[-len(time_bp_half)]-lam[-len(time_bp_half)-1])/lin_vel
            if SHOW_STATUS:
                plt.plot(time_bp,lam)
                plt.show()
            ###
        else:
            time_bp = lam/lin_vel
        
        # Calculate the number of steps for the trajectory
        num_steps = int(time_bp[-1] / self.TIMESTEP)

        # Initialize the trajectory list
        traj_q = []
        traj_xy = []
        traj_fz = []

        # Generate the trajectory
        for step in range(num_steps):
            # Calculate the current time
            current_time = step * self.TIMESTEP
            
            # Find the current segment
            for i in range(len(time_bp)-1):
                if current_time >= time_bp[i] and current_time < time_bp[i+1]:
                    seg = i
                    break
            
            # Calculate the fraction of time within the current segment
            frac = (current_time - time_bp[seg]) / (time_bp[seg+1] - time_bp[seg])
            
            # Calculate the desired joint position for the current step
            q_des = frac * curve_js[seg+1] + (1 - frac) * curve_js[seg]
            # Calculate the desired position for the current step
            xy_des = frac * curve_xy[seg+1] + (1 - frac) * curve_xy[seg]
            # Calculate the desired force for the current step
            fz_des = frac * force_path[seg+1] + (1 - frac) * force_path[seg]
            
            # Append the desired position to the trajectory
            traj_q.append(q_des)
            traj_xy.append(xy_des)
            traj_fz.append(fz_des)
        
        return np.array(traj_q), np.array(traj_xy), np.array(traj_fz), np.array(time_bp)
   
    def trajectory_force_PIDcontrol(self,traj_xy,traj_js,traj_fz,force_lookahead=False):
        
        assert len(traj_xy)==len(traj_fz), "trajectory length mismatch"
        assert len(traj_xy)==len(traj_js), "trajectory length mismatch"
        
        ### trajectory force control
        start_time=time.time()
        joint_force_exe=[]
        cart_force_exe=[]
        for i in range(len(traj_xy)):
            # joint reading
            q_now = self.read_position()
            tip_now = self.robot.fwd(q_now)
            tip_now_ipad = self.ipad_pose_inv_T*tip_now
            # read force
            ft_tip = self.ad_ati2pentip_T@self.ft_reading
            fz_now = float(ft_tip[-1])
            if np.linalg.norm(ft_tip[3:])>FORCE_PROTECTION: # force protection break
                print("force: ",ft_tip[3:])
                print("force too large")
                break
            
            ### force feedback control
            # get current desired force
            if force_lookahead:
                fz_des = traj_fz[min(i+self.params['lookahead_index'],len(traj_fz)-1)]
            else:
                fz_des = traj_fz[i]
            f_err = fz_des-fz_now # feedback error lookahead
            v_des_z = self.force_impedence_ctrl(f_err) # force control
            # get xyz (in ipad frame) next
            next_T = self.ipad_pose_inv_T*self.robot.fwd(traj_js[i])
            next_T.p = np.append(traj_xy[i],tip_now_ipad.p[2]+v_des_z*self.TIMESTEP)
            next_T_world = self.ipad_pose_T*next_T
            # get desired joint angles using ik
            q_des = self.robot.inv(next_T_world.p,next_T_world.R,q_now)[0]
            
            # v_des_xy = (traj_xy[i]-tip_now_ipad.p[:2])/self.TIMESTEP # desired xy velocity
            # v_des = np.append(v_des_xy, v_des_z)
            # nu_des = np.append(0,v_des)
            # nu_des_base = self.ad_ipad2base@nu_des
            # ### get Jacobian 
            # J = self.robot.jacobian(q_now)
            # # Joint position control
            # qd_des = np.linalg.pinv(J)@nu_des_base
            # q_des = q_now + qd_des*self.TIMESTEP
            
            ### position control
            # get current desired position
            # q_planned=traj_q[i]
            # T_planned=self.robot.fwd(q_planned)
            # T_planned_ipad=self.ipad_pose_T.inv()*T_planned
            # # add force control
            # v_des = (T_planned_ipad.p-tip_now_ipad.p)/self.TIMESTEP + np.array([0,0,v_des_z]) # force impedence control
            # # Cartesian position control
            # T_planned_ipad.p = tip_now_ipad.p + v_des * self.TIMESTEP
            # T_planned=self.ipad_pose_T*T_planned_ipad
            # q_des = self.robot.inv(T_planned.p,T_planned.R,q_now)[0]

            ### send robot position command
            self.position_cmd(q_des)
            
            # time, force, joint angle, record
            joint_force_exe.append(np.append(np.array([time.time()-start_time,fz_now]),q_now))
            # time, force, xyz, record
            cart_force_exe.append(np.append(np.array([time.time()-start_time,fz_now]),tip_now_ipad.p))
            
        return np.array(joint_force_exe), np.array(cart_force_exe)
    
    def jog_joint_position_cmd(self,q,v=0.4,wait_time=0):

        q_start=self.read_position()
        total_time=np.linalg.norm(q-q_start)/v
        q_all = np.vstack((q_start,q))
        traj_q, traj_xy, traj_fz, time_bp=self.trajectory_generate(q_all,np.zeros_like(q_all),np.zeros_like(q_all),lin_vel=v)

        for i in range(len(traj_q)):
            self.read_position()
            self.position_cmd(traj_q[i])
            if USE_RR_ROBOT:
                time.sleep(self.TIMESTEP)

        # start_time=time.time()
        # while time.time()-start_time<total_time:
        #     # Set the joint command
        #     frac=(time.time()-start_time)/total_time
        #     self.position_cmd(frac*q+(1-frac)*q_start)
            
        ###additional points for accuracy
        start_time=time.time()
        while time.time()-start_time<wait_time:
            self.read_position()
            self.position_cmd(q)
    
    def trajectory_position_cmd(self,q_all,v=0.4):


        traj_q, traj_xy, traj_fz, time_bp=self.trajectory_generate(q_all,np.zeros_like(q_all),np.zeros_like(q_all),lin_vel=v,lin_acc=0)

        for i in range(len(traj_q)):
            self.read_position()
            self.position_cmd(traj_q[i])
            if USE_RR_ROBOT:
                time.sleep(self.TIMESTEP)

def main():
    # img_name='wen_out'
    # img_name='strokes_out'
    # img_name='strokes_out_3'
    # img_name='wen_name_out'
    # img_name='me_out'
    # img_name='new_year_out'
    img_name='ilc_path2'

    visualize=False

    print("Drawing %s"%img_name)

    ipad_pose=np.loadtxt(CODE_PATH+'config/ipad_pose.csv',delimiter=',')
    ipad_pose_inv = np.linalg.inv(ipad_pose)
    num_segments=len(glob.glob(CODE_PATH+'path/cartesian_path/'+img_name+'/*.csv'))
    robot=robot_obj('ABB_1200_5_90',CODE_PATH+'config/ABB_1200_5_90_robot_default_config.yml',tool_file_path=CODE_PATH+'config/heh6_pen.csv')
    # robot=robot_obj('ur5','config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')
    # print(robot.fwd(np.array([0,0,0,0,0,0])))
    # exit()

    ######## FT sensor info
    H_pentip2ati=np.loadtxt(CODE_PATH+'config/pentip2ati.csv',delimiter=',')
 
    ######## Controller parameters ###
    controller_params = {
        "force_ctrl_damping": 60.0, # 180, 90, 60
        "force_epsilon": 0.1, # Unit: N
        "moveL_speed_lin": 6.0, # Unit: mm/sec
        "moveL_acc_lin": 1.0, # Unit: mm/sec^2
        "moveL_speed_ang": np.radians(10), # Unit: rad/sec
        "trapzoid_slope": 1, # trapzoidal load profile. Unit: N/sec
        "load_speed": 10.0, # Unit mm/sec
        "unload_speed": 1.0, # Unit mm/sec
        'settling_time': 1, # Unit: sec
        "lookahead_time": 0.132 # Unit: sec
        }
    
    ######## Motion Controller ###
    mctrl = MotionILController(robot,ipad_pose,H_pentip2ati,controller_params)
    # time.sleep(1)
    mctrl.drain_egm(1000)

    # parameters
    h_offset = 20
    h_offset_low = 1
    joging_speed = 15
    force_smooth_n = 11
    motion_smooth_n = 1
    
    iteration_N = 100
    alpha = 0.25
    use_lookahead = False
    motion_ilc = True
    force_ilc = True

    # record data
    Path(CODE_PATH+'record').mkdir(parents=True, exist_ok=True)
    ft_record_load=[]
    ft_record_move=[]

    for i in range(0,num_segments):
        print('Segment %i'%i)
        # pixel_path=np.loadtxt('path/pixel_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
        cartesian_path_world=np.loadtxt(CODE_PATH+'path/cartesian_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
        curve_xyz = np.dot(ipad_pose_inv[:3,:3],cartesian_path_world.T).T+np.tile(ipad_pose_inv[:3,-1],(len(cartesian_path_world),1))
        curve_js=np.loadtxt(CODE_PATH+'path/js_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,6))
        force_path=np.loadtxt(CODE_PATH+'path/force_path/'+img_name+'/%i.csv'%i,delimiter=',')
        
        # constant force
        # force_path = np.ones(len(curve_js))*F_des
        if len(curve_js)<2:
            continue
        
        ######## ILC #########################
        # transform to tip desired
        fz_des = force_path*(-1)
        # check if fz_des is a number or list
        if isinstance(fz_des,(int,float)):
            fz_des = np.ones(len(curve_js))*fz_des
        # get xy curve
        curve_xy = curve_xyz[:,:2]
        # get path length
        lam = calc_lam_js(curve_js,mctrl.robot)
        # get trajectory and time_bp
        traj_q, traj_xy, traj_fz, time_bp = mctrl.trajectory_generate(curve_js,curve_xy,fz_des)
        
        traj_xy_input = deepcopy(traj_xy)
        traj_fz_input = deepcopy(traj_fz)
        for iter_n in range(iteration_N):
            if SHOW_STATUS:
                input("Start iteration %i?"%iter_n)
            print("Iteration %i"%iter_n)
            ## first half
            print("First half")
            mctrl.motion_start_procedure(traj_q[0],traj_fz_input[0],h_offset,h_offset_low,lin_vel=joging_speed)
            joint_force_exe, cart_force_exe = mctrl.trajectory_force_PIDcontrol(traj_xy_input,traj_q,traj_fz_input,force_lookahead=use_lookahead)
            mctrl.motion_end_procedure(traj_q[-1],h_offset, lin_vel=joging_speed)
            
            curve_xy_exe = cart_force_exe[:,-3:-1]
            curve_f_exe = cart_force_exe[:,1]
            # moving average to smooth executed force and xy
            curve_xy_exe[:,0] = moving_average(curve_xy_exe[:,0], n=motion_smooth_n, padding=True)
            curve_xy_exe[:,1] = moving_average(curve_xy_exe[:,1], n=motion_smooth_n, padding=True)
            curve_f_exe = moving_average(curve_f_exe, n=force_smooth_n, padding=True)
            ## get error
            error_exe = np.hstack((traj_xy-curve_xy_exe,traj_fz.reshape(-1,1)-curve_f_exe.reshape(-1,1)))

            print("Iteration %i, motion xy Error: %f"%(iter_n, np.linalg.norm(error_exe[:,:-1])))
            print("Iteration %i, force Error: %f"%(iter_n, np.linalg.norm(error_exe[:,-1])))
            
            ## second half
            print("Second half")
            error_exe_flip = np.flip(error_exe,axis=0)
            traj_xy_auginput = traj_xy_input - error_exe_flip[:,:-1]
            traj_fz_auginput = traj_fz_input - error_exe_flip[:,-1]
            mctrl.motion_start_procedure(traj_q[0],traj_fz_input[0],h_offset,h_offset_low,lin_vel=joging_speed)
            joint_force_exe, cart_force_exe = mctrl.trajectory_force_PIDcontrol(traj_xy_auginput,traj_q,traj_fz_auginput,force_lookahead=use_lookahead)
            mctrl.motion_end_procedure(traj_q[-1],h_offset,lin_vel=joging_speed)
            
            # get gradient
            delta_xy = cart_force_exe[:,-3:-1]-curve_xy_exe
            delta_fz = cart_force_exe[:,1]-curve_f_exe
            G_error_flip = np.hstack((delta_xy,delta_fz.reshape(-1,1)))
            G_error = np.flip(G_error_flip,axis=0)
            if motion_ilc:
                traj_xy_input = traj_xy_input - alpha*G_error[:,:-1]
            if force_ilc:
                traj_fz_input = traj_fz_input - alpha*G_error[:,-1]
            
            # save data
            np.savetxt(CODE_PATH+'record/traj_xyf.csv',np.hstack((traj_xy,traj_fz.reshape(-1,1))),delimiter=',')
            np.savetxt(CODE_PATH+'record/iter_%i_traj_xyf_input.csv'%iter_n,np.hstack((traj_xy_input,traj_fz_input.reshape(-1,1))),delimiter=',')
            np.savetxt(CODE_PATH+'record/iter_%i_xyz_exe.csv'%iter_n,np.hstack((curve_xy_exe,curve_f_exe.reshape(-1,1))),delimiter=',')
            np.savetxt(CODE_PATH+'record/iter_%i_G_error.csv'%iter_n,G_error,delimiter=',')

            if PROGRESS_VIZ:
                # show gradient
                time_t = np.arange(0,len(G_error))*mctrl.TIMESTEP
                # subplots plotting x y force gradient
                fig, axs = plt.subplots(6)
                axs[0].plot(time_t,traj_xy[:,0],label='desired x')
                axs[0].plot(time_t,curve_xy_exe[:,0],label='executed x')
                axs[0].set_title('x')
                axs[0].legend()
                axs[1].plot(time_t,G_error[:,0],label='gradient x')
                axs[1].set_title('gradient x')
                axs[1].legend()
                axs[2].plot(time_t,traj_xy[:,1],label='desired y')
                axs[2].plot(time_t,curve_xy_exe[:,1],label='executed y')
                axs[2].set_title('y')
                axs[2].legend()
                axs[3].plot(time_t,G_error[:,1],label='gradient y')
                axs[3].set_title('gradient y')
                axs[3].legend()
                axs[4].plot(time_t,traj_fz,label='desired force')
                axs[4].plot(time_t,curve_f_exe,label='executed force')
                axs[4].set_title('force')
                axs[4].legend()
                axs[5].plot(time_t,G_error[:,2],label='gradient force')
                axs[5].set_title('gradient force')
                axs[5].legend()
                plt.show()
            
            print("=================================")
        ####################################
    
    #jog to end point
    mctrl.motion_end_procedure(traj_q[-1],150,lin_vel=joging_speed)
    
    ## direct visualization
    if visualize:
        for i in range(len(ft_record_load)):
            lam = calc_lam_js(curve_js,robot)
            lam_exe = calc_lam_js(np.radians(ft_record_move[i][:,2:]),robot)
            # linear interpolation
            f_desired = np.interp(lam_exe, lam, force_path)
            plt.plot(lam_exe,-1*ft_record_move[i][:,1],label='executed force, test '+str(i))
            plt.plot(lam_exe,f_desired,label='desired force')
            plt.legend()
            plt.show()

if __name__ == '__main__':
    main()