from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys
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


face_tracking_sub.ClientConnectFailed += connect_failed

while True:
	time.sleep(0.1)
	wire_packet=bbox_wire.TryGetInValue()
	if wire_packet[0]:
		capture_time=float(wire_packet[2].seconds+wire_packet[2].nanoseconds*1e-9)
		now=RRN.NowTimeSpec()
		now=float(now.seconds+now.nanoseconds*1e-9)
		print(wire_packet[1],now-capture_time)
		