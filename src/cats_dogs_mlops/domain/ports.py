from __future__ import annotations

from typing import Any, Protocol

from cats_dogs_mlops.domain.entities import PredictionResult


class ModelInferencePort(Protocol):
    def predict_bytes(self, image_bytes: bytes) -> PredictionResult:
        ...


class ExperimentTrackerPort(Protocol):
    def log_params(self, params: dict[str, Any]) -> None:
        ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        ...

    def log_artifact(self, path: Any) -> None:
        ...
