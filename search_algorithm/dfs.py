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
    
    def find_path_nodes(self):
        
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
        paths=[]
        paths_start=[]
        paths_end=[]
        for cje in self.edges:
            x_start = cje[0]
            y_start = cje[1]
            self.binary_image[y_start][x_start] = 0
            # start from a conjuction or edge and search directions
            for direction_start in directions:
                x_ = x_start+direction_start[0]
                y_ = y_start+direction_start[1]
                if 0 <= x_ < self.binary_image.shape[1] and 0 <= y_ < self.binary_image.shape[0]:
                    if self.binary_image[y_][x_] > 0 or [x_,y_] in self.edges:
                        self.binary_image[y_][x_] = 0
                        path=[[x_start,y_start],[x_,y_]]
                        x=x_
                        y=y_
                        while True and [x_,y_] not in self.edges:
                            for direction in directions:
                                x_ = x+direction[0]
                                y_ = y+direction[1]
                                if 0 <= x_ < self.binary_image.shape[1] and 0 <= y_ < self.binary_image.shape[0]:
                                    if self.binary_image[y_][x_] > 0 or [x_,y_] in self.edges and [x_,y_]!=[x_start,y_start]:
                                        break
                            self.binary_image[y_][x_] = 0
                            path.append([x_,y_])
                            if [x_,y_] in self.edges: # find a path to another conjuction or edge
                                break
                            x=x_
                            y=y_
                        paths.append(path)
                        paths_start.append(cje)
                        paths_end.append([x_,y_])
                        paths.append(path[::-1])
                        paths_start.append([x_,y_])
                        paths_end.append(cje)
        return paths,paths_start,paths_end
            