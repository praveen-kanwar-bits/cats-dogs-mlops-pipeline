from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError

from cats_dogs_mlops.application.preprocessing import build_eval_transform, select_device
from cats_dogs_mlops.domain.entities import PredictionResult


class InferenceService:
    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
        image_size: int,
        model_version: str,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.model_version = model_version
        self.transform = build_eval_transform(image_size)
        self.device = select_device()

    def predict_bytes(self, image_bytes: bytes) -> PredictionResult:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Invalid image payload") from exc

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        prob_map = {label: float(probs[i]) for i, label in enumerate(self.class_names)}
        predicted_class = max(prob_map, key=prob_map.get)
        confidence = prob_map[predicted_class]
        return PredictionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_map,
            model_version=self.model_version,
        )
