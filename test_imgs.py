import numpy as np
import cv2
from matplotlib import pyplot as plt

img_name='wen_out'

img = cv2.imread('imgs/'+img_name+'.png')

# plt.imshow(img)
# plt.show()

print(img.shape)
print(img.dtype)
img = np.ones(img.shape, dtype="uint8")*255

thickness=4
stroke_len=180
stroke_start=40
stroke_1=90
stroke_2=190
stroke_3=290
img[stroke_1:stroke_1+thickness,stroke_start:stroke_start+stroke_len,:]=0
img[stroke_2:stroke_2+thickness,stroke_start:stroke_start+stroke_len,:]=0
img[stroke_3:stroke_3+thickness,stroke_start:stroke_start+stroke_len,:]=0

plt.imshow(img)
plt.show()

cv2.imwrite('imgs/strokes_out.png', img)