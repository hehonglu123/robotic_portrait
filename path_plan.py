import cv2
import numpy as np



img=cv2.imread('imgs/me_out.png')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#threshold image to all 255 and 0
ret,img_gray=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
#run connected components of only 0's
thresholded_img=(img_gray==255).astype(np.uint8)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=4)


img2 = 255*np.ones(img_gray.shape, dtype="uint8")
#show each connected component, from blank to full
for i in range(0, nb_components):
    #ignore small components
    if stats[i][4] < 100 or img_gray[output == i][0]==0:
        continue
    img2[output == i] = 0
    cv2.imshow("img", img2)
    cv2.waitKey(0)
    img_temp= 255*np.ones(img_gray.shape, dtype="uint8")
    img_temp[output == i] = 0
    cv2.imwrite('imgs/connected_components/'+str(i)+'.png', img_temp)