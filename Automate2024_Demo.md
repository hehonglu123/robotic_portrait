# Automate 2024 Demo

## Portrait Demo

### Launching Camera
1. Launch ATI ft sensor RR service. Open a terminal.
```
cd ~/Desktop/ati
source run_sensor.sh
```

2. Launch camera and face tracking RR service. Open a terminal.
```
cd ~/Desktop/robot_portrait/face_tracking
python3 face_tracking_service.py
```

3. Launch the portraiting script. Open a terminal
```
cd ~/Desktop/robot_portrait
python3 integrated_client_fb.py
```

There will be a ``pop up window`` and the robot will start operating. The robot will continuously writing ``RPI x ABB`` before doing the next step.

Make sure there's to ABB RR service running before launching the script.

4. (Optional) Test if the ATI ft sensor is working. Open a terminal
```
cd ~/Desktop/ati
python3 test_ati_client.py
```

### When a visitor comes
1. Ask the visitor to stand on the preparation postion and face the robot. 
2. Click ``continue`` on the ``pop up window`` on PC. 
3. The robot will finish the current stroke, going to the standby position, and start face tracking.
4. When the robot goes down, this means the image is taken and the visitor can leave the standby position.

## Trouble shoot

1. When drawing, the robot touch the tablet and leave without drawing

**Solution**: This is because the ft sensor RR service is disconnected, and the safety mecghanism is activated. You need to shutdown both ``ft sensor RR service`` and ``the portrait script`` and restart them.

2. There's error in the ft sensor terminal.

**Solution**: This is the same issue as issue 1. Refer to the item.

## Calibration

### Rough Calibration using Aruco Tag
1. Launch camera (and face tracking, they are in the same script) RR service. Open a terminal.
```
cd ~/Desktop/robot_portrait/face_tracking
python3 face_tracking_service.py
```

2. Launch the Aruco tag RR service. Open a terminal.
```
cd ~/Desktop/robot_portrait/face_tracking
python3 aruco_service.py
```

3. Launch the ABB robot RR service. Open a terminal
```
cd ~/Desktop/ABB
source run_robot.sh
```

4. Launch the calibration script. Open a terminal
```
cd ~/Desktop/robot_portrait
source calibration.sh
```

5. Turn on the tablet and open the notebook with the aruco tag.

6. Jog the robot with the gui such that the aruco tag is about in the center of field of view of the camera with not too much skew.

7. Click ``Calibrate with tag`` on the panel. A pop up window will show and click ``Continue``.

### Refined Calibration using Force Feedback

1. Leave everything on in the previous part. You can turn off Aruco tag RR service and camera RR service if you want to.

2. Launch ATI ft sensor RR service. Open a terminal.
```
cd ~/Desktop/ati
source run_sensor.sh
```

3. Launch the refined calibration script. Open a terminal
```
python3 recalibration_force.py
```

4. Press ``Enter`` and the robot will jog to the corner. Press ``Enter`` again for the robot to press on the corner for calibration refinement.

5. **Important**: Always look out the ATI service to see if it's still on. If the ATI service throws an error, **E-Stop the robot immediately**.

6. The robot will finish the refinement once all 4 corners are tabbed.

7. It's always a good idea and safe to supervise the robot motion when the robot is pressing toward the tablet. **E-Stop** the robot when ever an unexpected motion occurs.