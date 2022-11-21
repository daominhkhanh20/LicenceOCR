from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse
from PIL import Image
import numpy as np
import cv2
import io
from model import licence_plate
import random 
from io import BytesIO

app = FastAPI()


def read_imagefile(file) -> Image.Image:
    nparr = np.fromstring(file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.get("/")
async def root():
    return {"message": "Hello wordl"}


@app.post('/upload')
async def receive_file(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    plate, boxs = licence_plate(image)
    cv2.rectangle(image, (int(boxs[0]), int(boxs[1])), (int(boxs[2]), int(boxs[3])), (255, 0, 0), 2)
    cv2.putText(image, plate, (int(boxs[0] - 5), int(boxs[1] - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 172, 5), 2)
    res, img_png = cv2.imencode('.png', image)
    return StreamingResponse(io.BytesIO(img_png.tobytes()), media_type='image/png')