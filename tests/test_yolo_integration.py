import os
from pathlib import Path
from PIL import Image
from api.yolo_model import yolo_predict


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_yolo_forward_with_repo_weights():
    weights = Path(project_root() / "api" / "weights" / "best.onnx")
    assert weights.is_file(), f"Expected weights at {weights}"
    img = Image.new("RGB", (320, 240), color=(200, 200, 200))
    label, conf, _ = yolo_predict(img, weights_path=str(weights), conf=0.25, iou=0.45, imgsz=320)
    assert label in {"HasStone", "NoStone"}
    assert 0.0 <= conf <= 1.0
