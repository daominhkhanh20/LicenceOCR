from pytorchyolo import models, detect
import torch
import numpy as np 
import torch
from PIL import Image 
from matplotlib import pyplot as plt
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'ocr/weights/transformerocr.pth'
config['cnn']['pretrained']=False
config['vocab'] = 'ABCDEFGHKLMNPSTUVXYZ0123456789-.'
config['device'] = torch.device("cpu")
config['predictor']['beamsearch']=False
model_detector = Predictor(config)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = models.load_model('darknet/cfg/yolov3-tiny.cfg', 'darknet/backup/yolov3-tiny_2000.weights').to(device)


def detector(image):
    boxes = detect.detect_image(model, image)
    return boxes

def ocr(image):
    plate = model_detector.predict(image)
    return plate

def licence_plate(image):
    boxes = detector(image)
    x,y,x1,y1 = int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3])
    numpy_image = image[y:y1, x:x1,:]
    imgage_ocr = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
    plate = ocr(image=imgage_ocr)
    return plate, boxes[0]
