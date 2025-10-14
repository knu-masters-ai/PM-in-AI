import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .schemas import PredictionResponse, BBox
from .inference import predict
from .utils import draw_boxes, pil_to_base64_png
from .yolo_model import load_yolo_model

app = FastAPI(title="KidneyStoneAI (YOLO)", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _warmup_model():
    weights = os.getenv("MODEL_WEIGHTS", "api/weights/best.pt")
    try:
        load_yolo_model(weights)
        print(f"[startup] YOLO weights loaded: {weights}")
    except Exception as e:
        # якщо ваг немає — краще впасти явно, щоб не було “тихо”
        raise RuntimeError(f"Failed to load YOLO weights at startup: {e}") from e


@app.get("/health")
def health():
    return {"status": "ok"}


MAX_BYTES = 25 * 1024 * 1024
ALLOWED_CT = {"image/png", "image/jpeg", "image/jpg"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Missing file field 'file'.")
    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_CT:
        raise HTTPException(status_code=415, detail="PNG or JPEG only.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_BYTES} bytes.")

    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Unsupported image format (PNG/JPEG only).")

    label, confidence, boxes = predict(img)
    vis = draw_boxes(img, boxes)
    b64 = pil_to_base64_png(vis)

    return PredictionResponse(
        label=label,
        confidence=confidence,
        boxes=[BBox(x=x, y=y, w=w, h=h, score=float(confidence)) for (x, y, w, h) in boxes],
        image_base64=b64,
        message=("Result: stones detected!" if label == "HasStone" else "Result: stones NOT detected."),
    )
