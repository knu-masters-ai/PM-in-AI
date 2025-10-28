from pydantic import BaseModel
from typing import List, Optional, Any


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
    explanation: Optional[Any] = None   # JSON від OpenAI (strict schema)
