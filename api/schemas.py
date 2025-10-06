from pydantic import BaseModel
from typing import List


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    score: float


class PredictionResponse(BaseModel):
    label: str  # "HasStone" | "NoStone"
    confidence: float
    boxes: List[BBox]
    image_base64: str
    message: str
