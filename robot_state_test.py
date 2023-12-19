from RobotRaconteur.Client import *
import numpy as np
import time

RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')
RR_robot=RR_robot_sub.GetDefaultClientWait(1)
robot_state = RR_robot_sub.SubscribeWire("robot_state")

time.sleep(1)
while True:
    print(robot_state.InValue.joint_effort)