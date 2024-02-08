import numpy as np
from copy import deepcopy
import cv2
import sys
sys.setrecursionlimit(10**6)

class DFS(object):
    def __init__(self, binary_image, edges=None):
        self.binary_image = deepcopy(binary_image)
        self.edges = deepcopy(edges)
        
    def search(self,from_edge=False):
        
        def dfs(x, y):
            if x==383 and y==618:
                print('dfs',x,y,'path len',len(path))
            # Check if the current pixel is within the image and belongs to the connected component
            if 0 <= x < self.binary_image.shape[1] and 0 <= y < self.binary_image.shape[0]:
                if self.binary_image[y][x] > 0.01:
                    self.binary_image[y][x] = 0
                    path.append([x,y])
                    
                    for dx, dy in directions:
                        dfs(x + dx, y + dy)
        
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
        paths = []
        last_start_pixel=None
        while (self.binary_image>0).any():
            if from_edge:
                for i in range(len(self.edges)):
                    if self.binary_image[self.edges[i][1]][self.edges[i][0]] > 0:
                        start_pixel=self.edges[i]
                        break
            else:
                start_pixel=np.where(self.binary_image>0)
                start_pixel=[(start_pixel[0][0],start_pixel[1][0])]
            if np.all(last_start_pixel==start_pixel):
                break
            path=[]
            dfs(start_pixel[0],start_pixel[1])
            paths.append(np.array(path))

            last_start_pixel=start_pixel
        
        return paths