import cv2, sys
import numpy as np
sys.setrecursionlimit(10**6)

from matplotlib import pyplot as plt

def pixels_in_radius(x, y, pen_radius_pixel, image_shape):
    max_x, max_y = image_shape[1], image_shape[0]  # Get image dimensions

    pixels_list = []
    for i in range(int(x - pen_radius_pixel), int(x + pen_radius_pixel + 1)):
        for j in range(int(y - pen_radius_pixel), int(y + pen_radius_pixel + 1)):
            # Check if the pixel (i, j) is within the circle and within the image boundaries
            if (0 <= i < max_x) and (0 <= j < max_y) and (np.sqrt((x - i) ** 2 + (y - j) ** 2) <= pen_radius_pixel):
                pixels_list.append((i, j))
    return pixels_list


###DFS to traverse connected component
img_name='me_out'
img_gray=cv2.imread('imgs/'+img_name+'.png',cv2.IMREAD_GRAYSCALE)
img_gray=cv2.bitwise_not(img_gray)

img_copy = img_gray.copy()
paper_size=np.loadtxt('config/paper_size.csv',delimiter=',')
pen_radius=np.loadtxt('config/pen_radius.csv',delimiter=',')
#find a ratio to fit image in paper
pixel2mm=min(paper_size/img_gray.shape)
pen_radius_pixel=pen_radius/pixel2mm

# convolution resize
shrunken_image = cv2.resize(img_copy, (int(img_copy.shape[1]/(2*pen_radius_pixel)), int(img_copy.shape[0]/(2*pen_radius_pixel))), interpolation = cv2.INTER_LINEAR )
binary_image=shrunken_image.copy()
ret,binary_image=cv2.threshold(binary_image,20,255,cv2.THRESH_BINARY)
plt.imshow(cv2.bitwise_not(binary_image), cmap='gray')
plt.show()

img_display=np.zeros(img_gray.shape, dtype="uint8")
path_all=[]
force_all=[]
while (binary_image==255).any():
    
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    path=[]
    force=[]
    def dfs(x, y):
        # Check if the current pixel is within the image and belongs to the connected component
        if 0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]:
            if binary_image[y][x] == 255:
                # cv2.imshow("img", binary_image)
                # cv2.waitKey(0)
                # Mark the current pixel as visited (e.g., change its value to 0)
                binary_image[y][x] = 0
                path.append([x,y])
                force.append(shrunken_image[y][x]/255.)
                # Recursively visit the 8-connected neighbors
                for dx, dy in directions:
                    dfs(x + dx, y + dy)

    #plot image
    # plt.imshow(cv2.bitwise_not(binary_image), cmap='gray')
    # plt.show()
    dfs(np.where(binary_image==255)[1][0],np.where(binary_image==255)[0][0])

    path_original_size=(np.array(path)*2*pen_radius_pixel).astype(int)
    ###display path in image
    img_temp = np.zeros(img_gray.shape, dtype="uint8")
    for i in range(len(path_original_size)):
        img_temp[path_original_size[i][1],path_original_size[i][0]]=0
        pixels=pixels_in_radius(path_original_size[i][0],path_original_size[i][1],force[i]*pen_radius_pixel*np.sqrt(2),img_gray.shape)
        for p in pixels:
            img_temp[p[1],p[0]]=255
        
        img_display=cv2.bitwise_or(img_display,img_temp)
        cv2.imshow("img", cv2.bitwise_not(img_display))
        cv2.waitKey(0)
    

    img_copy=img_copy-img_temp
    ret,img_copy=cv2.threshold(img_copy,126,255,cv2.THRESH_BINARY)

    cv2.imshow("img", cv2.bitwise_not(img_copy))
    cv2.waitKey(0)

    path_all.append(path_original_size)
    force_all.append(force)

#save path
count=0
for m in range(len(path_all)):
    indices=[]
    for i in range(len(path_all[m])-1):
        if np.linalg.norm(path_all[m][i]-path_all[m][i+1])>3*pen_radius_pixel:
            indices.append(i+1)
    #split path
    path_split=np.split(path_all[m],indices)
    force_split=np.split(force_all[m],indices)
    for i in range(len(path_split)):
        print(force_split[i].shape,path_split[i].shape)
        np.savetxt('path/pixel_path/'+img_name+'/%i.csv'%count, np.hstack((path_split[i],force_split[i].reshape(-1,1))), delimiter=',')
        count+=1
