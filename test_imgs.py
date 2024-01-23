import numpy as np
import cv2
from matplotlib import pyplot as plt

img_name='wen_out'

img = cv2.imread('imgs/'+img_name+'.png')

print(img)

plt.imshow(img)
plt.show()