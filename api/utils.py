import base64
import io
from PIL import Image, ImageDraw
from typing import List, Tuple
from io import BytesIO


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


def resize_max_width(pil_img: Image.Image, max_w: int = 1024) -> Image.Image:
    w, h = pil_img.size
    if w <= max_w:
        return pil_img
    new_h = int(h * (max_w / float(w)))
    return pil_img.resize((max_w, new_h), Image.LANCZOS)


def pil_to_data_url_jpeg(pil_img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def pil_to_data_url_png(pil_img) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
