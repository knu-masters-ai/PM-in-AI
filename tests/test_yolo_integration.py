import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_yolo_forward_with_repo_weights():
    weights = Path(project_root() / "api" / "weights" / "best.onnx")
    assert weights.is_file(), f"Expected weights at {weights}"
