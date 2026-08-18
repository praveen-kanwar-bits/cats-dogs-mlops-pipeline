from typing import Protocol

import torch


class ModelInferencePort(Protocol):
    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        ...
