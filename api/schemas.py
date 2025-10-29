from pydantic import BaseModel
from typing import List, Optional, Any


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    score: float


class PredictionResponse(BaseModel):
    label: str  # "HasStone" | "NoStone"
    confidence: float
    boxes: List[BBox]
    image_base64: str
    message: str
    explanation: Optional[Any] = None
    emailed: Optional[bool] = None  # <- нове
    email_error: Optional[str] = None  # <- нове (лише для дебагу)
