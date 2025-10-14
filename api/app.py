# api/app.py
from __future__ import annotations

import io
import os

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .schemas import PredictionResponse, BBox
from .inference import predict  # -> всередині має викликати ONNX-бекенд (onnxruntime)
from .utils import draw_boxes, pil_to_base64_png
from .onnx_infer import yolo_onnx_predict  # для warm-up

app = FastAPI(title="KidneyStoneAI (ONNX)", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _warmup_model():
    """
    Перевіряємо, що ONNX-ваги доступні та інференс працює.
    Робимо один прогін на «порожньому» зображенні 320×320.
    """
    weights = os.getenv("MODEL_WEIGHTS", "api/weights/best.onnx")
    try:
        dummy = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        # одноразовий інференс прогріє сесію onnxruntime (якщо кеш реалізовано у onnx_infer)
        label, conf, _ = yolo_onnx_predict(dummy, weights_path=weights, conf=0.25, iou=0.45, imgsz=320)
        print(f"[startup] ONNX weights loaded: {weights} -> warmup: {label} ({conf:.2f})")
    except Exception as e:
        # якщо ваг немає/биті — краще впасти явно
        raise RuntimeError(f"Failed to load ONNX weights at startup: {e}") from e


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
