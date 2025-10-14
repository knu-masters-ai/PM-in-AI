from __future__ import annotations
from typing import List, Tuple, Optional
import os

from PIL import Image
from ultralytics import YOLO

_YOLO_MODEL: Optional[YOLO] = None
_WEIGHTS_USED: Optional[str] = None


def load_yolo_model(weights_path: str) -> YOLO:
    global _YOLO_MODEL, _WEIGHTS_USED
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    if _YOLO_MODEL is None or _WEIGHTS_USED != weights_path:
        _YOLO_MODEL = YOLO(weights_path)  # device auto (CPU на Fly)
        _WEIGHTS_USED = weights_path
    return _YOLO_MODEL


def yolo_predict(
        image: Image.Image,
        weights_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
) -> tuple[str, float, List[Tuple[int, int, int, int]]]:
    model = load_yolo_model(weights_path)
    results = model.predict(
        source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False, device="cpu"
    )
    r = results[0]
    if r.boxes is None or r.boxes.xyxy is None:
        boxes_xyxy, scores = [], []
    else:
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

    W, H = image.size
    out_boxes: List[Tuple[int, int, int, int]] = []
    max_conf = 0.0

    for (x1, y1, x2, y2), sc in zip(boxes_xyxy, scores):
        x, y = int(max(0, x1)), int(max(0, y1))
        w, h = int(max(1, x2 - x1)), int(max(1, y2 - y1))
        w = min(w, W - x)
        h = min(h, H - y)
        out_boxes.append((x, y, w, h))
        max_conf = max(max_conf, float(sc))

    if out_boxes:
        return "HasStone", max_conf, out_boxes
    else:
        return "NoStone", 1.0, []
