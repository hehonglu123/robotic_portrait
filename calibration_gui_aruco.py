#!/usr/bin/python3
from RobotRaconteur.Client import *
import sys, os, time, argparse, traceback
from tkinter import *
from tkinter import messagebox
from qpsolvers import solve_qp
import numpy as np
from importlib import import_module
sys.path.append('toolbox')
from vel_emulate_sub import EmulatedVelocityControl
from robot_def import *
from general_robotics_toolbox import *    
from sklearn.decomposition import PCA




def Rx(theta):
    return np.array(([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]]))
def Ry(theta):
    return np.array(([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]))
def Rz(theta):
    return np.array(([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]))

global vel_ctrl, robot_kin, p_all
p_all=[]

#Accept the names of the robots from command line
parser = argparse.ArgumentParser(description="RR plug and play client")
parser.add_argument("--tool-file",type=str,required=True)
parser.add_argument("--robot-name",type=str,required=True)
parser.add_argument("--save-file",type=str,default='ipad_pose')
args, _ = parser.parse_known_args()

name_dict={'ABB':'ABB_1200_5_90','ur':'ur5'}
#load robot class
robot_kin=robot_obj(name_dict[args.robot_name],'config/'+name_dict[args.robot_name]+'_robot_default_config.yml',tool_file_path='config/'+args.tool_file+'.csv')
    

#auto discovery for robot service
time.sleep(2)
res=RRN.FindServiceByType("com.robotraconteur.robotics.robot.Robot",
["rr+local","rr+tcp","rrs+tcp"])
url=None
for serviceinfo2 in res:
    if args.robot_name in serviceinfo2.NodeName:
        url=serviceinfo2.ConnectionURL
        break
if url==None:
    print('service not found')
    sys.exit()

#auto discovery for robot tool service
res=RRN.FindServiceByType("com.robotraconteur.robotics.tool.Tool",
["rr+local","rr+tcp","rrs+tcp"])
url_gripper=None
for serviceinfo2 in res:
    if args.robot_name in serviceinfo2.NodeName:
        url_gripper=serviceinfo2.ConnectionURL
        break
if url_gripper==None:
    print('gripper service not found')
else:
    #connect
    try:
        tool=RRN.ConnectService(url_gripper)
    except:
        traceback.print_exc()

#connect robot services
robot_sub=RRN.SubscribeService(url)
robot=robot_sub.GetDefaultClientWait(1)
state_w = robot_sub.SubscribeWire("robot_state")
#get params of robots
num_joints=len(robot.robot_info.joint_info)
P=np.array(robot.robot_info.chains[0].P.tolist())
length=np.linalg.norm(P[1])+np.linalg.norm(P[2])+np.linalg.norm(P[3])
H=np.transpose(np.array(robot.robot_info.chains[0].H.tolist()))


##########Initialize robot constants
robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", robot)
state_flags_enum = robot_const['RobotStateFlags']
halt_mode = robot_const["RobotCommandMode"]["halt"]
position_mode = robot_const["RobotCommandMode"]["position_command"]
robot.command_mode = halt_mode
time.sleep(0.1)
robot.command_mode = position_mode
cmd_w = robot_sub.SubscribeWire("position_command")

vel_ctrl = EmulatedVelocityControl(robot,state_w, cmd_w)
#enable velocity mode
vel_ctrl.enable_velocity_mode()


###### for aruco tag calibration ####
robot_cam = robot_obj(name_dict[args.robot_name],'config/'+name_dict[args.robot_name]+'_robot_default_config.yml',tool_file_path='config/camera.csv')
aruco_ser_url='rr+tcp://localhost:2356?service=aruco_detector'
marker_id = 'marker0'
aruco_cli = None
aruco_pipe = None


top=Tk()
top.title(args.robot_name)
jobid = None
def gripper_ctrl(tool):

    if gripper.config('relief')[-1] == 'sunken':
        tool.open()
        gripper.config(relief="raised")
        gripper.configure(bg='red')
        gripper.configure(text='gripper off')

    else:
        tool.close()
        gripper.config(relief="sunken")
        gripper.configure(bg='green')
        gripper.configure(text='gripper on')
    return

def movej(qdot):
    global jobid

    if np.max(np.abs(speed.get()*qdot))>1.0:
        qdot=np.zeros(6)
        print('too fast')

    vel_ctrl.set_velocity_command(speed.get()*qdot)
    jobid = top.after(10, lambda: movej(qdot))
    return

def move(vd, ER):
    global jobid, vel_ctrl, robot_kin
    try:
        w=1.
        Kq=.01*np.eye(6)    #small value to make sure positive definite
        KR=np.eye(3)        #gains for position and orientation error

        q_cur=vel_ctrl.joint_position()
        J=robot_kin.jacobian(q_cur)       #calculate current Jacobian
        Jp=J[3:,:]
        JR=J[:3,:] 
        H=np.dot(np.transpose(Jp),Jp)+Kq+w*np.dot(np.transpose(JR),JR)

        H=(H+np.transpose(H))/2


        k,theta = R2rot(ER)
        k=np.array(k)
        s=np.sin(theta/2)*k         #eR2
        wd=-np.dot(KR,s)  
        f=-np.dot(np.transpose(Jp),vd)-w*np.dot(np.transpose(JR),wd)
        ###Don't put bound here, will affect cartesian motion outcome
        qdot=speed.get()*solve_qp(H, f)
        ###For safty, make sure robot not moving too fast
        if np.max(np.abs(qdot))>1.0:
            qdot=np.zeros(6)
            print('too fast')
        vel_ctrl.set_velocity_command(qdot)

        jobid = top.after(10, lambda: move(vd, ER))
    except:
        traceback.print_exc()
    return

def stop():
    global jobid,vel_ctrl
    top.after_cancel(jobid)
    vel_ctrl.set_velocity_command(np.zeros((6,)))
    return

def save_p(filename):
    global vel_ctrl, robot_kin, p_all
    p_all.append(robot_kin.fwd(vel_ctrl.joint_position()).p)
    if len(p_all)==4:
        p_all=np.array(p_all)
        #identify the center point and the plane
        center=np.mean(p_all,axis=0)
        pca = PCA()
        pca.fit(p_all)
        R_temp = pca.components_.T		###decreasing variance order
        if R_temp[:,0]@center<0:		###correct orientation
            R_temp[:,0]=-R_temp[:,0]
        if R_temp[:,-1]@robot_kin.fwd(vel_ctrl.joint_position()).R[:,2]>0:
            R_temp[:,-1]=-R_temp[:,-1]
        
        R_temp[:,1]=np.cross(R_temp[:,2],R_temp[:,0])


        np.savetxt('config/'+filename+'.csv', H_from_RT(R_temp,center), delimiter=',')

        messagebox.showinfo('Message', 'pose saved')
  
def save_p_aruco(filename):
    global robot_cam, aruco_ser_url, aruco_cli, aruco_pipe
    
    if aruco_cli is None:
        aruco_cli = RRN.ConnectService(aruco_ser_url)
        aruco_pipe = aruco_cli.fiducials_sensor_data.Connect(-1)
    
    euler_angle = []
    points = []
    st = time.time()
    while time.time() - st < 1:
        data = aruco_pipe.ReceivePacketWait()
        for i in range(len(data.fiducials.recognized_fiducials)):
            if data.fiducials.recognized_fiducials[i].fiducial_marker != marker_id:
                continue
            p = np.array([data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']['x'],
                            data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']['y'],
                            data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']['z']])
            R = np.array([data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']['w'],
                            data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']['x'],
                            data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']['y'],
                            data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']['z']])
            R = R2rpy(q2R(R))
            euler_angle.append(R)
            points.append(p)
    points = np.array(points)
    euler_angle = np.array(euler_angle)
    center = np.mean(points, axis=0)
    euler_angle = np.mean(euler_angle, axis=0)
    R = rpy2R(euler_angle)
    T_ipad_cam = Transform(R, center)
    T_cam_robot = robot_cam.fwd(vel_ctrl.joint_position())
    T_ipad_robot = T_cam_robot*T_ipad_cam
    R = T_ipad_robot.R
    R = R@Rz(np.pi)
    center = T_ipad_robot.p
    print(R, center)
    np.savetxt('config/'+filename+'.csv', H_from_RT(R,center), delimiter=',')
    messagebox.showinfo('Message', 'pose saved')

def clear_p():
    global p_all
    p_all=[]
    return


##RR part
def update_label():
    global p_all
    robot_state=state_w.TryGetInValue()
    flags_text = "Robot State Flags:\n\n"
    if robot_state[0]:
        for flag_name, flag_code in state_flags_enum.items():
            if flag_code & robot_state[1].robot_state_flags != 0:
                flags_text += flag_name + "\n"
    else:
        flags_text += 'service not running'
        
    joint_text = "Robot Joint Positions:\n\n"
    for j in robot_state[1].joint_position:
        joint_text += "%.2f\n" % np.rad2deg(j)

    point_text = "Robot Saved Position:\n\n"
    for p in p_all:
        point_text += "%.2f,%.2f,%.2f\n" % (p[0],p[1],p[2])
    label.config(text = flags_text + "\n\n" + joint_text + "\n\n" + point_text)

    label.after(250, update_label)

top.title = "Robot State"

###speed control
speed= Scale(orient='vertical', label='speed control',from_=1., length=500,resolution=0.1, to=10.)
speed.pack(side=RIGHT)


label = Label(top, fg = "black", justify=LEFT)
label.pack()
label.after(250,update_label)



save=Button(top,text='save')
clear=Button(top,text='clear')
use_aruco=Button(top,text='use aruco')
left=Button(top,text='left')
right=Button(top,text='right')
forward=Button(top,text='forward')
backward=Button(top,text='backward')
up=Button(top,text='up')
down=Button(top,text='down')

Rx_n=Button(top,text='Rx_n')
Rx_p=Button(top,text='Rx_p')
Ry_n=Button(top,text='Ry_n')
Ry_p=Button(top,text='Ry_p')
Rz_n=Button(top,text='Rz_n')
Rz_p=Button(top,text='Rz_p')

j1_n=Button(top,text='j1_n')
j1_p=Button(top,text='j1_p')
j2_n=Button(top,text='j2_n')
j2_p=Button(top,text='j2_p')
j3_n=Button(top,text='j3_n')
j3_p=Button(top,text='j3_p')
j4_n=Button(top,text='j4_n')
j4_p=Button(top,text='j4_p')
j5_n=Button(top,text='j5_n')
j5_p=Button(top,text='j5_p')
j6_n=Button(top,text='j6_n')
j6_p=Button(top,text='j6_p')



gripper=Button(top,text='gripper off',command=lambda: gripper_ctrl(tool),bg='red')

save.bind('<ButtonPress-1>', lambda event: save_p(args.save_file))
clear.bind('<ButtonPress-1>', lambda event: clear_p())
use_aruco.bind('<ButtonPress-1>', lambda event: save_p_aruco('test_ipad_pose'))
left.bind('<ButtonPress-1>', lambda event: move([0,20,0],np.eye(3)))
right.bind('<ButtonPress-1>', lambda event: move([0,-20,0],np.eye(3)))
forward.bind('<ButtonPress-1>', lambda event: move([20,0,0],np.eye(3)))
backward.bind('<ButtonPress-1>', lambda event: move([-20,0,0],np.eye(3)))
up.bind('<ButtonPress-1>', lambda event: move([0,0,20],np.eye(3)))
down.bind('<ButtonPress-1>', lambda event: move([0,0,-20],np.eye(3)))

Rx_n.bind('<ButtonPress-1>', lambda event: move([0.,0.,0.],Rx(+0.1)))
Rx_p.bind('<ButtonPress-1>', lambda event: move([0.,0.,0.],Rx(-0.1)))
Ry_n.bind('<ButtonPress-1>', lambda event: move([0.,0.,0.],Ry(+0.1)))
Ry_p.bind('<ButtonPress-1>', lambda event: move([0.,0.,0.],Ry(-0.1)))
Rz_n.bind('<ButtonPress-1>', lambda event: move([0.,0.,0.],Rz(+0.1)))
Rz_p.bind('<ButtonPress-1>', lambda event: move([0.,0.,0.],Rz(-0.1)))

j1_n.bind('<ButtonPress-1>', lambda event: movej(np.array([-0.1,0.,0.,0.,0.,0.])))
j1_p.bind('<ButtonPress-1>', lambda event: movej(np.array([+0.1,0.,0.,0.,0.,0.])))
j2_n.bind('<ButtonPress-1>', lambda event: movej(np.array([0.,-0.1,0.,0.,0.,0.])))
j2_p.bind('<ButtonPress-1>', lambda event: movej(np.array([0.,+0.1,0.,0.,0.,0.])))
j3_n.bind('<ButtonPress-1>', lambda event: movej(np.array([0.,0.,-0.1,0.,0.,0.])))
j3_p.bind('<ButtonPress-1>', lambda event: movej(np.array([0.,0.,+0.1,0.,0.,0.])))
j4_n.bind('<ButtonPress-1>', lambda event: movej(np.array([0.,0.,0.,-0.1,0.,0.])))
j4_p.bind('<ButtonPress-1>', lambda event: movej(np.array([0.,0.,0.,+0.1,0.,0.])))
j5_n.bind('<ButtonPress-1>', lambda event: movej(np.array([0.,0.,0.,0.,-0.1,0.])))
j5_p.bind('<ButtonPress-1>', lambda event: movej(np.array([0.,0.,0.,0.,+0.1,0.])))
j6_n.bind('<ButtonPress-1>', lambda event: movej(np.array([0.,0.,0.,0.,0.,-0.1])))
j6_p.bind('<ButtonPress-1>', lambda event: movej(np.array([0.,0.,0.,0.,0.,+0.1])))


left.bind('<ButtonRelease-1>', lambda event: stop())
right.bind('<ButtonRelease-1>', lambda event: stop())
forward.bind('<ButtonRelease-1>', lambda event: stop())
backward.bind('<ButtonRelease-1>', lambda event: stop())
up.bind('<ButtonRelease-1>', lambda event: stop())
down.bind('<ButtonRelease-1>', lambda event: stop())

Rx_n.bind('<ButtonRelease-1>', lambda event: stop())
Rx_p.bind('<ButtonRelease-1>', lambda event: stop())
Ry_n.bind('<ButtonRelease-1>', lambda event: stop())
Ry_p.bind('<ButtonRelease-1>', lambda event: stop())
Rz_n.bind('<ButtonRelease-1>', lambda event: stop())
Rz_p.bind('<ButtonRelease-1>', lambda event: stop())

j1_n.bind('<ButtonRelease-1>', lambda event: stop())
j1_p.bind('<ButtonRelease-1>', lambda event: stop())
j2_n.bind('<ButtonRelease-1>', lambda event: stop())
j2_p.bind('<ButtonRelease-1>', lambda event: stop())
j3_n.bind('<ButtonRelease-1>', lambda event: stop())
j3_p.bind('<ButtonRelease-1>', lambda event: stop())
j4_n.bind('<ButtonRelease-1>', lambda event: stop())
j4_p.bind('<ButtonRelease-1>', lambda event: stop())
j5_n.bind('<ButtonRelease-1>', lambda event: stop())
j5_p.bind('<ButtonRelease-1>', lambda event: stop())
j6_n.bind('<ButtonRelease-1>', lambda event: stop())
j6_p.bind('<ButtonRelease-1>', lambda event: stop())



gripper.pack()
save.pack()
clear.pack()
use_aruco.pack()
left.pack(in_=top, side=LEFT)
right.pack(in_=top, side=RIGHT)
forward.pack(in_=top, side=LEFT)
backward.pack(in_=top, side=RIGHT)
up.pack(in_=top, side=LEFT)
down.pack(in_=top, side=RIGHT)

Rx_n.pack()
Rx_p.pack()
Ry_n.pack()
Ry_p.pack()
Rz_n.pack()
Rz_p.pack()

j1_n.pack(in_=top, side=LEFT)
j1_p.pack(in_=top, side=LEFT)
j2_n.pack(in_=top, side=LEFT)
j2_p.pack(in_=top, side=LEFT)
j3_n.pack(in_=top, side=LEFT)
j3_p.pack(in_=top, side=LEFT)
j4_n.pack(in_=top, side=LEFT)
j4_p.pack(in_=top, side=LEFT)
j5_n.pack(in_=top, side=LEFT)
j5_p.pack(in_=top, side=LEFT)
j6_n.pack(in_=top, side=LEFT)
j6_p.pack(in_=top, side=LEFT)

top.mainloop()
vel_ctrl.disable_velocity_mode()