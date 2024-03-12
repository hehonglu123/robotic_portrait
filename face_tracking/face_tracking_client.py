from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys
import numpy as np
import cv2
#RR client setup, connect to turtle service
url='rr+tcp://localhost:52222/?service=Face_tracking'
#take url from command line
if (len(sys.argv)>=2):
        url=sys.argv[1]

###2 modes available, choose either one		
########wire connection mode:
# obj=RRN.ConnectService(url)
# turtle_change=obj.turtle_change.Connect()


########subscription mode
def connect_failed(s, client_id, url, err):
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))

face_tracking_sub=RRN.SubscribeService(url)
obj = face_tracking_sub.GetDefaultClientWait(30)		#connect, timeout=30s
bbox_wire=face_tracking_sub.SubscribeWire("bbox")
image_wire=face_tracking_sub.SubscribeWire("frame_stream")

face_tracking_sub.ClientConnectFailed += connect_failed

while True:
    try:
        time.sleep(0.1)
        wire_packet=bbox_wire.TryGetInValue()
        if wire_packet[0]:
            capture_time=float(wire_packet[2].seconds+wire_packet[2].nanoseconds*1e-9)
            now=RRN.NowTimeSpec()
            now=float(now.seconds+now.nanoseconds*1e-9)
            print(wire_packet[1],now-capture_time)
    except KeyboardInterrupt:
        break
RR_image=image_wire.TryGetInValue()
bbox=wire_packet[1]
size=np.array([bbox[2]-bbox[0],bbox[3]-bbox[1]])
if RR_image[0]:
    img=RR_image[1]
    img=np.array(img.data,dtype=np.uint8).reshape((img.image_info.height,img.image_info.width,3))
    #get the image within the bounding box, a bit larger than the bbox
    img=img[int(bbox[1]-size[1]/5):int(bbox[3]+size[1]/9),int(bbox[0]-size[0]/9):int(bbox[2]+size[0]/9),:]

print('IMAGE TAKEN')
cv2.imwrite('temp_data/img.jpg',img)