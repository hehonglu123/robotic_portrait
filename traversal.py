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
img_name='glenn_out'
img_gray=cv2.imread('imgs/'+img_name+'.png',cv2.IMREAD_GRAYSCALE)
img_gray=cv2.bitwise_not(img_gray)

img_copy = img_gray.copy()
paper_size=np.loadtxt('config/paper_size.csv',delimiter=',')
pen_radius=np.loadtxt('config/pen_radius.csv',delimiter=',')
#find a ratio to fit image in paper
pixel2mm=min(paper_size/img_gray.shape)
pen_radius_pixel=pen_radius/pixel2mm

# convolution resize
shrunken_image = cv2.resize(img_copy, (int(img_copy.shape[1]/(2*pen_radius_pixel)), int(img_copy.shape[0]/(2*pen_radius_pixel))), interpolation = cv2.INTER_NEAREST )

img_display=np.zeros(img_gray.shape, dtype="uint8")
path_all=[]
while (shrunken_image==255).any():
    
    # convolution of the image with the kernel same stride of kernel size
    shrunken_image = cv2.resize(img_copy, (int(img_copy.shape[1]/(2*pen_radius_pixel)), int(img_copy.shape[0]/(2*pen_radius_pixel))), interpolation = cv2.INTER_NEAREST )
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

    path_original_size=(np.array(path)*2*pen_radius_pixel).astype(int)
    ###display path in image
    img_temp = np.zeros(img_gray.shape, dtype="uint8")
    for i in range(len(path_original_size)):
        img_temp[path_original_size[i][1],path_original_size[i][0]]=0
        pixels=pixels_in_radius(path_original_size[i][0],path_original_size[i][1],pen_radius_pixel*np.sqrt(2),img_gray.shape)
        for p in pixels:
            img_temp[p[1],p[0]]=255
        # #make surronding pixels within pen_radius_pixel are also black in radius
        # for j in range(-pen_radius_pixel,pen_radius_pixel+1):
        #     for k in range(-pen_radius_pixel,pen_radius_pixel+1):
        #         if 0 <= path_original_size[i][1]+j < img_gray.shape[0] and 0 <= path_original_size[i][0]+k < img_gray.shape[1]:
        #             img_temp[path_original_size[i][1]+j,path_original_size[i][0]+k]=255
        
        # img_display=cv2.bitwise_or(img_display,img_temp)
        # cv2.imshow("img", cv2.bitwise_not(img_display))
        # cv2.waitKey(0)
    

    img_copy=img_copy-img_temp
    ret,img_copy=cv2.threshold(img_copy,126,255,cv2.THRESH_BINARY)

    cv2.imshow("img", cv2.bitwise_not(img_copy))
    cv2.waitKey(0)

    path_all.append(path_original_size)

#save path
count=0
for path in path_all:
    indices=[]
    for i in range(len(path)-1):
        if np.linalg.norm(path[i]-path[i+1])>3*pen_radius_pixel:
            indices.append(i+1)
    #split path
    path_split=np.split(path,indices)
    print(indices)
    for p in path_split:
        np.savetxt('path/pixel_path/'+img_name+'/%i.csv'%count, p, delimiter=',')
        count+=1
