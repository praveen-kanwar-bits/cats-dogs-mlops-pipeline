from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
