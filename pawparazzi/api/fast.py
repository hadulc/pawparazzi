from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pawparazzi.breed_predict.predict import load_model, predict_breed
from typing import Dict
import numpy as np
import cv2
from pawparazzi.breed_predict.names import TSINGHUA_BREEDS

app = FastAPI()
app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
def index():
    pass


@app.post("/predict")
async def predict(img: UploadFile = File(...), decimals: int = 4) -> Dict[str, str]:
    """
    Receives an image file, decodes it using OpenCV, and predicts dog breeds.

    Args:
        img (UploadFile): The image file to be uploaded.

    Returns:
        Dict[str, str]: A dictionary where keys are breed names and values are
                        their predicted scores.
    """
    contents = await img.read()
    nparr = np.frombuffer(buffer=contents, dtype=np.uint8)
    cv2_img = cv2.imdecode(buf=nparr, flags=cv2.IMREAD_COLOR)
    scores, ids = predict_breed(image=cv2_img, model=app.state.model)
    breeds = [TSINGHUA_BREEDS[i] for i in ids]
    prediction = {b:f"{s:.{decimals}f}" for b,s in zip(breeds, scores)}
    return prediction
