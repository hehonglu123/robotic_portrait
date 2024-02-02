import numpy as np
from copy import deepcopy

class DFS(object):
    def __init__(self, binary_image, edges=None):
        self.binary_image = deepcopy(binary_image)
        
    def search(self,from_edge=False):
        def dfs(x, y):
            if self.binary_image[y][x] > 0:
                self.binary_image[y][x] = 0
                path.append([x,y])
                for dx, dy in directions:
                    dfs(x + dx, y + dy)
        
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        path = []
        while (self.binary_image>0).any():
            if from_edge:
                edge_pixels = np.where(np.logical_and(self.binary_image == 255, np.logical_or(np.roll(self.binary_image, 1, axis=0) == 0, np.roll(self.binary_image, -1, axis=0) == 0, np.roll(self.binary_image, 1, axis=1) == 0, np.roll(self.binary_image, -1, axis=1) == 0)))
                for i in range(len(edge_pixels[0])):
                    dfs(edge_pixels[1][i], edge_pixels[0][i])
            else:
                start_pixel=np.where(self.binary_image>0)
            dfs(start_pixel[1][0],start_pixel[0][0])
        return path