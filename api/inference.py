import hashlib
from typing import List, Tuple
import numpy as np
from PIL import Image


def _image_hash(arr: np.ndarray) -> int:
    h = hashlib.md5(arr.tobytes()).hexdigest()
    return int(h[:8], 16)


def mock_predict(image: Image.Image) -> tuple[str, float, List[Tuple[int, int, int, int]]]:
    """Mock classifier & localizer (deterministic).
    Produces random-looking boxes but always the same for same image.
    """
    img = image.convert("L").resize((512, 512))
    arr = np.array(img, dtype=np.uint8)
    std = float(arr.std())

    seed = _image_hash(arr)
    rng = np.random.default_rng(seed)

    has_stone = std > 35 or (seed % 5 == 0)
    label = "HasStone" if has_stone else "NoStone"
    confidence = float(min(0.99, 0.5 + std / 255)) if has_stone else float(1.0 - min(0.49, std / 255))

    boxes: List[Tuple[int, int, int, int]] = []
    if has_stone:
        h, w = arr.shape
        n = int(rng.integers(1, 4))
        for _ in range(n):
            bw = int(rng.integers(w // 12, w // 6))
            bh = int(rng.integers(h // 12, h // 6))
            x = int(rng.integers(0, w - bw))
            y = int(rng.integers(0, h - bh))
            boxes.append((x, y, bw, bh))

    return label, confidence, boxes
