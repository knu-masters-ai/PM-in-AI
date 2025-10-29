# api/app.py
from __future__ import annotations

import base64
import io
import os

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .ai_explainer import explain_annotated_image_as_json
from .mailer import send_results_email
from .schemas import PredictionResponse, BBox
from .inference import predict  # -> всередині має викликати ONNX-бекенд (onnxruntime)
from .utils import draw_boxes, pil_to_base64_png, pil_to_data_url_png, resize_max_width, pil_to_data_url_jpeg
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
async def predict_endpoint(
    file: UploadFile = File(...),
    email: str | None = Form(None),     # <- опційний email у тій самій multipart-формі
):
    # ... (перевірки типу, читання контенту як було)
    original_bytes = await file.read()
    if not original_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(original_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_BYTES} bytes.")

    try:
        img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Unsupported image format (PNG/JPEG only).")

    # інференс
    label, confidence, boxes = predict(img)
    vis = draw_boxes(img, boxes)

    # png для UI
    b64 = pil_to_base64_png(vis)

    # підготовка для OpenAI (у тебе вже інтегровано)
    vis_small = resize_max_width(vis, 1024)
    data_url = pil_to_data_url_jpeg(vis_small, 85)

    # пояснювач
    explanation = {}
    summary = "Результат сформовано локально. Пояснювач тимчасово недоступний."
    try:
        from .ai_explainer import explain_annotated_image_as_json
        hint = f"Кількість виділених ділянок: {len(boxes)}. Загальна впевненість: {confidence:.2f}."
        explanation = explain_annotated_image_as_json(data_url_png=data_url, extra_hint=hint) or {}
        summary = explanation.get("summary_text") or (
            "Виявлено ознаки каменів." if label == "HasStone" else "Ознак каменів не виявлено."
        )
    except Exception as e:
        # логування вже ввімкнене вище
        pass

    emailed = None
    email_err = None
    if email:
        try:
            # визначимо ім'я оригінального файлу для вкладення
            original_name = file.filename or "input_image"
            if not any(original_name.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
                original_name += ".jpg"
            # відправляємо
            send_results_email(
                to_email=email,
                label=label,
                message=summary,
                original_bytes=original_bytes,
                original_filename=original_name,
                annotated_png_bytes=base64.b64decode(b64),
                explanation=explanation or None,
            )
            emailed = True
        except Exception as e:
            emailed = False
            email_err = str(e)[:300]

    return PredictionResponse(
        label=label,
        confidence=confidence,
        boxes=[BBox(x=x, y=y, w=w, h=h, score=float(confidence)) for (x, y, w, h) in boxes],
        image_base64=b64,
        message=summary,
        explanation=explanation or None,
        emailed=emailed,
        email_error=email_err,  # у проді можна не повертати або за прапорцем
    )
