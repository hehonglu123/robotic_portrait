from RobotRaconteur.Client import *     #import RR client library
import cv2
import numpy as np





face_tracking_sub=RRN.SubscribeService('rr+tcp://localhost:52222/?service=Face_tracking')
obj = face_tracking_sub.GetDefaultClientWait(1)		#connect, timeout=30s
bbox_wire=face_tracking_sub.SubscribeWire("bbox")
image_wire=face_tracking_sub.SubscribeWire("frame_stream")




while True:
    RR_image=image_wire.TryGetInValue()
    print(RR_image[0])
    if RR_image[0]:
        img=RR_image[1]
        img=np.array(img.data,dtype=np.uint8).reshape((img.image_info.height,img.image_info.width,3))