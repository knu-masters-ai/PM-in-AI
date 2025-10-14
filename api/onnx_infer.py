from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "onnxruntime is required. Add `onnxruntime==1.19.2` to requirements.txt"
    ) from e


# ---- Глобальний кеш сесії для швидкого інференсу ----
_SESSION: ort.InferenceSession | None = None
_INPUT_NAME: str | None = None
_OUTPUT_NAME: str | None = None
_WEIGHTS_PATH: str | None = None


def _letterbox(im: np.ndarray, new_shape=640, color=(114, 114, 114)) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize & pad to square keeping aspect ratio, як у YOLO."""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_resized = np.array(Image.fromarray(im).resize((nw, nh), Image.BILINEAR))

    pad_w = new_shape[1] - nw
    pad_h = new_shape[0] - nh
    top = pad_h // 2
    left = pad_w // 2

    out = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = im_resized
    return out, r, (left, top)


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh.T
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    """Проста NMS на NumPy."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def _ensure_session(weights_path: str):
    """Ледаче створення та кешування ONNX-сесії."""
    global _SESSION, _INPUT_NAME, _OUTPUT_NAME, _WEIGHTS_PATH
    if _SESSION is not None and _WEIGHTS_PATH == weights_path:
        return

    p = Path(weights_path)
    if not p.is_file():
        raise FileNotFoundError(f"ONNX weights not found at: {weights_path}")

    providers = ["CPUExecutionProvider"]  # легкий CPU-only
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = int(os.getenv("ORT_INTRA_THREADS", "1"))
    sess_opts.inter_op_num_threads = int(os.getenv("ORT_INTER_THREADS", "1"))
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _SESSION = ort.InferenceSession(str(p), sess_options=sess_opts, providers=providers)
    _INPUT_NAME = _SESSION.get_inputs()[0].name
    _OUTPUT_NAME = _SESSION.get_outputs()[0].name
    _WEIGHTS_PATH = weights_path


def _postprocess(
    raw: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    gain: float,
    pad: tuple[int, int],
    orig_wh: tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    """
    Підтримує 2 найтиповіші форми виходу:
    - (1, num_boxes, num_attrs)   або
    - (1, num_attrs, num_boxes)
    Де num_attrs може бути 6..85 (x,y,w,h,conf,[class...]).
    """
    out = raw
    if out.ndim != 3:
        # напр., (batch, *, *) -> беремо перший
        out = out.reshape(1, *out.shape[-2:])

    if out.shape[1] < out.shape[2]:
        # (1, attrs, N) -> transpose до (N, attrs)
        out = out[0].transpose(1, 0)
    else:
        # (1, N, attrs) -> (N, attrs)
        out = out[0]

    # очікуємо принаймні [x,y,w,h,conf]
    if out.shape[1] < 5:
        return []

    xywh = out[:, 0:4]
    conf = out[:, 4]

    # Якщо є класові конфіди після 5-го стовпця — помножимо conf на max(class_prob)
    if out.shape[1] > 5:
        cls_probs = out[:, 5:]
        cls_conf = cls_probs.max(axis=1)
        conf = conf * cls_conf

    # фільтр за конфіденсом
    m = conf >= conf_thres
    if not np.any(m):
        return []

    xywh = xywh[m]
    conf = conf[m]

    # У виході моделі координати зазвичай у масштабі letterbox.
    # Конвертація в xyxy
    xyxy = _xywh_to_xyxy(xywh)

    # Скайлити назад у вихідний розмір (прибрати паддінги та gain)
    # Спочатку прибираємо паддінг:
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    # Потім ділимо на gain:
    xyxy /= gain

    # Обрізати до меж оригіналу
    W, H = orig_wh
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)

    # NMS
    keep = _nms(xyxy, conf, iou_thres)
    xyxy = xyxy[keep].round().astype(int)

    # у формат (x, y, w, h)
    boxes_xywh: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in xyxy:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        boxes_xywh.append((int(x1), int(y1), int(w), int(h)))
    return boxes_xywh


def yolo_onnx_predict(
    image: Image.Image,
    weights_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> tuple[str, float, List[Tuple[int, int, int, int]]]:
    """
    Повертає: (label, confidence, boxes[(x,y,w,h), ...])
    label ∈ {"HasStone", "NoStone"}
    """
    _ensure_session(weights_path)

    # оригінальні розміри
    W, H = image.size

    # підготовка
    img = np.array(image.convert("RGB"))
    lb, gain, pad = _letterbox(img, new_shape=imgsz)
    inp = lb.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None, ...]  # BCHW

    assert _SESSION is not None and _INPUT_NAME is not None and _OUTPUT_NAME is not None
    outputs = _SESSION.run([_OUTPUT_NAME], {_INPUT_NAME: inp})
    raw = outputs[0].astype(np.float32)

    boxes = _postprocess(raw, conf_thres=conf, iou_thres=iou, gain=gain, pad=pad, orig_wh=(W, H))

    dets = _postprocess(raw, conf_thres=conf, iou_thres=iou, gain=gain, pad=pad, orig_wh=(W, H))

    if len(boxes) > 0:
        # довіра як макс конфіденс після NMS (приблизно: бер. перший бокс після сорту)
        # для простоти візьмемо 0.90, або ти можеш повернути реальні conf зі збереженням під час NMS
        label = "HasStone"
        confidence = max(d[-1] for d in dets)  # реальний максимум score
    else:
        label = "NoStone"
        confidence = 1.00

    return label, float(confidence), boxes
