import matplotlib.pyplot as plt
import cv2
import easyocr
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO


# image_path = r'D:/Computer_Vision/LiveProject/Text_Detection/src/test21.png'

def readImage(img):

    # img = cv2.imread(file)

    reader = easyocr.Reader(['en'], gpu=False)

    text_ = reader.readtext(img)
    return_text = ""
    for t_,t in enumerate(text_):

        bbox, text, score = t

        return_text += " "+text +'\n'

    return return_text

app = FastAPI()

origins = [
    'http://localhost',
    'http://localhost:3000',
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def ping():
    return "hello I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/readImage")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    get_text = readImage(image)

    return get_text


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


