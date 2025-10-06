# api/app.py
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .schemas import PredictionResponse, BBox
from .inference import mock_predict
from .utils import draw_boxes, pil_to_base64_png

app = FastAPI(title="KidneyStoneAI Mock API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    label, confidence, boxes = mock_predict(img)
    vis = draw_boxes(img, boxes)
    b64 = pil_to_base64_png(vis)

    return PredictionResponse(
        label=label,
        confidence=confidence,
        boxes=[BBox(x=x, y=y, w=w, h=h, score=0.8) for (x, y, w, h) in boxes],
        image_base64=b64,
        message=("Mocked model: replace with real inference later. "
                 + ("Stones likely present." if label == "HasStone" else "No stones detected.")),
    )
