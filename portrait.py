import os
import sys
import cv2
import numpy as np
import timeit
import onnxruntime
import argparse
import torch
import facer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def brighten_dark_areas(image, alpha=1.2, beta=100):
    # Apply alpha and beta to the image
    result = cv2.addWeighted(image, alpha, np.zeros_like(image), 0, beta)

    return result

if __name__ == "__main__":

    anime = AnimeGANv3('models/AnimeGANv3_PortraitSketch.onnx')
    # data_dir='imgs/'
    # img_name='me'
    data_dir='temp_data/'
    img_name='img'
    img = cv2.imread(data_dir+img_name+'.jpg')
    
    image = facer.hwc2bchw(facer.read_hwc(data_dir+img_name+'.jpg')).to(device=device)  # image: 1 x 3 x h x w

    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    face_parser = facer.face_parser('farl/lapa/448', device=device) # optional "farl/celebm/448"
    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    print(type(seg_probs))
    print(seg_probs.size())
    n_classes = seg_probs.size(1)
    vis_seg_probs = seg_probs.argmax(dim=1).float()/n_classes*255
    print(type(vis_seg_probs))
    vis_img = vis_seg_probs.sum(0, keepdim=True)
    print(type(vis_img))
    facer.show_bhw(vis_img)
    facer.show_bchw(facer.draw_bchw(image, faces))
    
    exit()
    #convert dark pixels to bright pixels
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #TODO: Identify dark cloth and convert brighter
    #display img
    cv2.imshow("img", brighten_dark_areas(img))
    cv2.waitKey(0)

    output = anime.forward(gray_image)
    cv2.imwrite(data_dir+img_name+'_out.jpg', output)

    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Apply Gaussian blur to reduce noise
    # gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # # Perform Canny edge detection
    # edges = cv2.bitwise_not(cv2.Canny(gray_image, 30, 200))
    # cv2.imwrite('output_edge.png', edges)
