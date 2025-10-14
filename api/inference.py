from __future__ import annotations
import os
from typing import List, Tuple
from PIL import Image
from .onnx_infer import yolo_onnx_predict  # твій onnxruntime код


def predict(image: Image.Image) -> tuple[str, float, List[Tuple[int, int, int, int]]]:
    weights = os.getenv("MODEL_WEIGHTS", "api/weights/best.onnx")
    conf = float(os.getenv("YOLO_CONF", "0.25"))
    iou = float(os.getenv("YOLO_IOU", "0.45"))
    imgsz = int(os.getenv("YOLO_IMGSZ", "640"))
    return yolo_onnx_predict(image, weights_path=weights, conf=conf, iou=iou, imgsz=imgsz)
