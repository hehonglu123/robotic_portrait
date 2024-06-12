# Robotic Portrait

## Dependencies
### Robot Raconteur Robot Drivers
* [ABB IRC5](https://github.com/robotraconteur-contrib/ABBRobotRaconteurDriver)
* [UR cb2] (https://github.com/robotraconteur-contrib/URRobotRaconteurDriver)


### Python Libraries
`python -m pip install -r requirements.txt`


## Instruction
* Start Robot Driver: `cd ~/Desktop/abb_driver_csharp-2024-01-02/ && source run_driver.sh `
* Start ATI Driver: `cd ~/Desktop/ati_robotraconteur_driver/ && source run_sensor.sh`
* Start Camera Driver: `cd ~/Desktop/robotic_portrait/face_tracking && python face_tracking_service.py `


## Carlibration
* Start Aruco Detection: `cd ~/Desktop/robotic_portrait/face_tracking/ && python aruco_service.py`
* Start Aruco Calibration: `cd ~/Desktop/robotic_portrai/ &&$ source calibration.sh`
Jog the robot such that the camera can see the aruco tag, and click `use aruco`. Then quit the Aruco Detection Service and Calibration script.

* Force-based Calibration: `cd ~/Desktop/robotic_portrait/ && python recalibration_force.py`

## Draw the Portrait
Keep the ATI and Camera driver opened, run `python integrated_client_fb.py`.

<!-- ### Running the Robot
`.\ABBRobotRaconteurDriver.exe --robot-info-file=abb_1200_5_90_robot_default_config.yml`

### Config Files
* Tool Transformation: `heh6_pen.csv`
* Tool to ATI Transformation: `pentip2ati.csv`
* Camera Transformation: `camera.csv`
* Tablet Size (mm): `paper_size.csv`
* Maximum Drawing Radius (mm): `pen_radius.csv`
* Tablet Transformation: `ipad_pose.csv`

### Face Tracking
Connect the Oak1 camera with computer using USB3 cables. 
`python face_tracking_service.py` and `python face_tracking_motion.py`


### Calibration
Jog the robot to tap 4 corners of the tablet, no need to be accurate.
`python calibration_gui.py --tool-file=heh6_pen --robot-name=ABB` or `./calibration.sh`

Then rerun the force-based calibration again for accurate calibration `python recalibration_force.py` to update `ipad_pose.csv`.

### Image to Binary 
`python portrait.py`

### Pixel Traversal (Tool Path Planning)
`python traversal_force.py`

### Convert to Cartesian Space
`python traj_gen_cartesian.py`

### Convert to Joint Space
`python traj_gen_js.py`

### Drawing
`python execute.py` -->
