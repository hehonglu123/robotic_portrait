import numpy as np

#  f = ( image.height / 2.0 ) / tan( (M_PI * FOV/180.0 )/2.0 ) 

FOV_X = 69
fx = (4056/2.0) / np.tan( (np.pi * FOV_X/180.0)/2.0 )
FOX_Y = 55
fy = (3040/2.0) / np.tan( (np.pi * FOX_Y/180.0)/2.0 )

print('fx:', fx)
print('fy:', fy)