import cv2, sys
import numpy as np
sys.setrecursionlimit(10**6)

from matplotlib import pyplot as plt

def pixels_in_radius(x, y, pen_radius, image_shape):
    max_x, max_y = image_shape[1], image_shape[0]  # Get image dimensions

    pixels_list = []
    for i in range(int(x - pen_radius), int(x + pen_radius + 1)):
        for j in range(int(y - pen_radius), int(y + pen_radius + 1)):
            # Check if the pixel (i, j) is within the circle and within the image boundaries
            if (0 <= i < max_x) and (0 <= j < max_y) and (np.sqrt((x - i) ** 2 + (y - j) ** 2) <= pen_radius):
                pixels_list.append((i, j))
    return pixels_list


###DFS to traverse connected component
img_name='me_out'
img=cv2.imread('imgs/'+img_name+'.png')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gray=cv2.bitwise_not(img_gray)

img_copy = img_gray.copy()
pen_radius=4
# Create a kernel for erosion
kernel = np.ones((2*pen_radius, 2*pen_radius), np.uint8)
shrunken_image = cv2.resize(img_copy, (int(img_copy.shape[1]/(2*pen_radius)), int(img_copy.shape[0]/(2*pen_radius))), interpolation = cv2.INTER_NEAREST )

img_display=np.zeros(img_gray.shape, dtype="uint8")
path_all=[]
while (shrunken_image==255).any():
    
    # convolution of the image with the kernel same stride of kernel size
    shrunken_image = cv2.resize(img_copy, (int(img_copy.shape[1]/(2*pen_radius)), int(img_copy.shape[0]/(2*pen_radius))), interpolation = cv2.INTER_NEAREST )
    # ret,shrunken_image=cv2.threshold(shrunken_image,126,255,cv2.THRESH_BINARY)
    # #convert all path in shrunk image to 0 to avoid infinite loop bug
    # try:
    #     for p in path:
    #         shrunken_image[p[1],p[0]]=255
    # except:
    #     pass
    

    # cv2.imshow("img", shrunken_image)
    # cv2.waitKey(0)
    # Define the directions for 8-connected neighbors
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    path=[]
    def dfs(x, y):
        # Check if the current pixel is within the image and belongs to the connected component
        if 0 <= x < shrunken_image.shape[1] and 0 <= y < shrunken_image.shape[0]:
            if shrunken_image[y][x] == 255:
                # cv2.imshow("img", shrunken_image)
                # cv2.waitKey(0)
                # Mark the current pixel as visited (e.g., change its value to 0)
                shrunken_image[y][x] = 0
                path.append([x,y])
                # Recursively visit the 8-connected neighbors
                for dx, dy in directions:
                    dfs(x + dx, y + dy)

    #plot image
    # plt.imshow(cv2.bitwise_not(shrunken_image), cmap='gray')
    # plt.show()
    dfs(np.where(shrunken_image==255)[1][0],np.where(shrunken_image==255)[0][0])
    print(path)

    path_original_size=np.array(path)*2*pen_radius
    ###display path in image
    img_temp = np.zeros(img_gray.shape, dtype="uint8")
    for i in range(len(path_original_size)):
        img_temp[path_original_size[i][1],path_original_size[i][0]]=0
        pixels=pixels_in_radius(path_original_size[i][0],path_original_size[i][1],pen_radius*np.sqrt(2),img_gray.shape)
        for p in pixels:
            img_temp[p[1],p[0]]=255
        # #make surronding pixels within pen_radius are also black in radius
        # for j in range(-pen_radius,pen_radius+1):
        #     for k in range(-pen_radius,pen_radius+1):
        #         if 0 <= path_original_size[i][1]+j < img_gray.shape[0] and 0 <= path_original_size[i][0]+k < img_gray.shape[1]:
        #             img_temp[path_original_size[i][1]+j,path_original_size[i][0]+k]=255
        
        img_display=cv2.bitwise_or(img_display,img_temp)
        cv2.imshow("img", cv2.bitwise_not(img_display))
        cv2.waitKey(0)
    

    img_copy=img_copy-img_temp
    ret,img_copy=cv2.threshold(img_copy,126,255,cv2.THRESH_BINARY)

    cv2.imshow("img", cv2.bitwise_not(img_copy))
    cv2.waitKey(0)

    path_all.append(path_original_size)

for i in range(len(path_all)):
    #save path
    np.savetxt('path/pixel_path/%s/%i.csv'%(img_name,i), path_all[i], delimiter=',')