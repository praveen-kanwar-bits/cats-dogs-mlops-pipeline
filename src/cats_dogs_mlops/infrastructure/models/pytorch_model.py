from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from cats_dogs_mlops.application.preprocessing import select_device
from cats_dogs_mlops.application.training import load_model_checkpoint


def load_model_bundle(model_path: Path) -> tuple[torch.nn.Module, dict]:
    model, payload = load_model_checkpoint(model_path, device=select_device())
    return model, payload


def write_metadata(metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.setdefault("training_timestamp", datetime.now(UTC).isoformat())
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
