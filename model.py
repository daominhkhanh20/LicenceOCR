from pytorchyolo import models, detect
import torch
import numpy as np 
from PIL import Image 

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = models.load_model('darknet/cfg/yolov3-tiny.cfg', 'darknet/backup/yolov3-tiny_2000.weights').to(device)


def detector(image):
    boxes = detect.detect_image(model, image)
    return boxes

def ocr(image):
    pass 

def licence_plate(image):
    boxes = detector(image)
    x,y,x1,y1 = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
    numpy_image = image[x:x1, y:y1,:]
    imgage_ocr = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
    plate = ocr(image=imgage_ocr)
