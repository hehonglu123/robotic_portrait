import os
import sys
import cv2
import numpy as np
import timeit
import time
import onnxruntime
import argparse
import torch
import facer
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AnimeGANv3:
    def __init__(self, model_path):
        self.ort_session = onnxruntime.InferenceSession(model_path)

    def resize_image_x32(self, img):
        
        def to_32s(x):
                return 256 if x < 256 else x - x%32
        h, w = img.shape[:2]
        ratio = h/w
        new_h = to_32s(h)
        new_w = int(new_h/ratio) - int(new_h/ratio)%32
        img = cv2.resize(img, (new_w, new_h))
        return img

    def process_image(self, img, x32=True):
        if x32: # resize image to multiple of 32s
            img = self.resize_image_x32(img)
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

class FaceSegmentation:
    def __init__(self) -> None:
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        self.face_parser = facer.face_parser('farl/lapa/448', device=device) # optional "farl/celebm/448"
    def forward_faceonly(self,img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = torch.from_numpy(img_rgb)
        img_input = facer.hwc2bchw(img_rgb).to(device=device)  # image: 1 x 3 x h x w
        
        with torch.inference_mode():
            faces = self.face_detector(img_input)
        with torch.inference_mode():
            faces = self.face_parser(img_input, faces)
        
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        n_classes = seg_probs.size(1)
        vis_seg_probs = seg_probs.argmax(dim=1).float()
        image_mask = np.squeeze(vis_seg_probs.cpu().numpy())
        # ret,image_mask = cv2.threshold(image_mask, 0, 255, cv2.THRESH_BINARY)
        image_mask=image_mask.astype(np.uint8)
        return image_mask,faces
    def get_face_mask(self,img):
        image_mask,faces = self.forward_faceonly(img)
        ret,face_mask = cv2.threshold(image_mask, 0, 255, cv2.THRESH_BINARY)
        #convert dark pixels to bright pixels
        # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_image = img
        gray_image_masked = cv2.bitwise_and(gray_image, gray_image, mask = face_mask)
        # get second masked value (background) mask must be inverted
        background = np.full(gray_image.shape, 255, dtype=np.uint8)
        bk = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(face_mask))
        gray_image_masked = cv2.add(gray_image_masked, bk)
        return gray_image_masked,image_mask,face_mask,faces

def brighten_dark_areas(image, alpha=1.2, beta=100):
    # Apply alpha and beta to the image
    result = cv2.addWeighted(image, alpha, np.zeros_like(image), 0, beta)

    return result

if __name__ == "__main__":

    
    # data_dir='imgs/'
    # img_name='me'
    data_dir='temp_data/'
    img_name='img1'
    img = cv2.imread(data_dir+img_name+'.jpg')
    
    fs = FaceSegmentation()
    start_time=time.time()
    
    print('segmentation time:',time.time()-start_time)
    gray_image_masked,image_mask,face_mask = fs.get_face_mask(img)
    plt.imshow(image_mask)
    plt.show()
    cv2.imshow("img", gray_image_masked)
    cv2.waitKey(0)
    
    #TODO: Identify dark cloth and convert brighter
    #display img
    # cv2.imshow("img", brighten_dark_areas(gray_image_masked))
    # cv2.waitKey(0)

    anime = AnimeGANv3('models/AnimeGANv3_PortraitSketch.onnx')
    output = anime.forward(gray_image_masked)
    cv2.imwrite(data_dir+img_name+'_out.jpg', output)
    
    output = anime.forward(gray_image)
    cv2.imwrite(data_dir+img_name+'_out_test.jpg', output)

    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Apply Gaussian blur to reduce noise
    # gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # # Perform Canny edge detection
    # edges = cv2.bitwise_not(cv2.Canny(gray_image, 30, 200))
    # cv2.imwrite('output_edge.png', edges)
