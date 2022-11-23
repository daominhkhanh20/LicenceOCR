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
config['vocab'] = 'ABCDEFGHKLMNPSTUVXYZR0123456789-.'
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
    plates = []
    for i in range(len(boxes)):
        x,y,x1,y1 = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
        numpy_image = image[y:y1, x:x1,:]
        imgage_ocr = Image.fromarray(np.uint8(numpy_image)).convert('RGB')
        plate = ocr(image=imgage_ocr)
        plates.append(plate)
    return plates, boxes
