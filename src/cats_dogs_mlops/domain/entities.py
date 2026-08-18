from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
