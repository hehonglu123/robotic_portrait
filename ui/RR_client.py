from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys

#Function to call when "Bump" event occurs
def response_received():
    print("New Response!!")

url='rr+tcp://localhost:52222/?service=Create'
#take url from command line
if (len(sys.argv)>=2):
    url=sys.argv[1]

#Startup, connect, and pull out the camera from the objref    
c=RRN.ConnectService(url)

c.response_received+=response_received


input("press enter to quit")
	