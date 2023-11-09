import os
import sys
import cv2
import numpy as np
import timeit
import onnxruntime
import argparse


class AnimeGANv3:
    def __init__(self, model_path):
        self.ort_session = onnxruntime.InferenceSession(model_path)

    def process_image(self, img, x32=True):
        h, w = img.shape[:2]
        ratio = h/w
        if x32: # resize image to multiple of 32s
            def to_32s(x):
                return 256 if x < 256 else x - x%32
            new_h = to_32s(h)
            new_w = int(new_h/ratio) - int(new_h/ratio)%32
            img = cv2.resize(img, (new_w, new_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
        return img

    def forward(self, img):
        img = self.process_image(img)
        img = np.float32(img[np.newaxis,:,:,:])
        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        output = ort_outs[0]
        output_image = (np.squeeze(output) + 1.) / 2 * 255
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        return output

if __name__ == "__main__":

    anime = AnimeGANv3('models/AnimeGANv3_PortraitSketch_25.onnx')
    img_name='me'
    img = cv2.imread('imgs/'+img_name+'.png')
    output = anime.forward(img)
    cv2.imwrite('imgs/'+img_name+'_out.png', output)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Apply Gaussian blur to reduce noise
    # gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # # Perform Canny edge detection
    # edges = cv2.bitwise_not(cv2.Canny(gray_image, 30, 200))
    # cv2.imwrite('output_edge.png', edges)
