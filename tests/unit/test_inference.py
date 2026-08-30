from pathlib import Path

import torch

from cats_dogs_mlops.application.inference import InferenceService


class DummyModel(torch.nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        logits = torch.tensor([[0.1, 2.0]], dtype=torch.float32).repeat(batch, 1)
        return logits


def test_inference_probabilities_valid():
    model = DummyModel()
    service = InferenceService(model, ["cat", "dog"], image_size=224, model_version="test")
    payload = Path("tests/fixtures/sample_dog.jpg").read_bytes()
    out = service.predict_bytes(payload)
    assert out.predicted_class in {"cat", "dog"}
    assert abs(sum(out.probabilities.values()) - 1.0) < 1e-6
    assert out.confidence == out.probabilities[out.predicted_class]
    assert out.model_version == "test"


def test_inference_rejects_non_image_bytes():
    service = InferenceService(DummyModel(), ["cat", "dog"], 224, "test")
    try:
        service.predict_bytes(b"not an image")
    except ValueError as exc:
        assert "Invalid image payload" in str(exc)
    else:
        raise AssertionError("Non-image bytes must be rejected")
