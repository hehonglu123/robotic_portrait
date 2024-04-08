import cv2
import numpy as np
import time
import yaml

# Load the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

# Create the ArUco parameters
aruco_params = cv2.aruco.DetectorParameters()

camera_param = yaml.safe_load(open("../config/camera_parameters.yaml"))
f_x = camera_param['camera_matrix']['fx']
f_y = camera_param['camera_matrix']['fy']
c_x = camera_param['camera_matrix']['cx']
c_y = camera_param['camera_matrix']['cy']
k1 = camera_param['distortion_coefficients']['k1']
k2 = camera_param['distortion_coefficients']['k2']
p1 = camera_param['distortion_coefficients']['p1']
p2 = camera_param['distortion_coefficients']['p2']
k3 = camera_param['distortion_coefficients']['k3']
# camera matrix
camera_matrix = np.array([[f_x, 0, c_x],
                          [0, f_y, c_y],
                          [0,  0,  1]], dtype=np.float32)
# distortion coefficients
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

cap = cv2.VideoCapture(0)
while KeyboardInterrupt:
    # Load the input image
    # image = cv2.imread("path/to/your/image.jpg")
    # Read image from webcam
    ret, frame = cap.read()
    image = frame.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Draw the detected markers on the image
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    # Estimate the pose of the detected markers
    
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

    # Draw the pose axes on the image
    if rvecs is not None and tvecs is not None:
        for rvec, tvec, id_num in zip(rvecs, tvecs, ids):
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            print("ids: ", id_num)
            print("rvec: ", cv2.Rodrigues(rvec)[0])
            print("tvec: ", tvec)

    # Display the image
    cv2.imshow("Image", image)
    cv2.waitKey(1)
    time.sleep(0.1)
cap.release()