"""
Accepts user images and preprocess
"""

import cv2
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
import numpy as np

"""
Helper Function
"""
app = FastAPI()
def convert_to_cv2_image(image_bytes):
    # add error logging
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.post("/upload")
async def uploadImage(image: UploadFile):
    image_bytes = await image.read()
    image = convert_to_cv2_image(image_bytes)
    return {"received_image": "success"}

    