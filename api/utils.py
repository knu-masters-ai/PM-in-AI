import base64
import io
from PIL import Image, ImageDraw
from typing import List, Tuple


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def draw_boxes(img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for (x, y, w, h) in boxes:
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=4)
    return out
