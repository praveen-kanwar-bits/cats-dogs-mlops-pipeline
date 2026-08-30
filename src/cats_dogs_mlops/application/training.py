from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from cats_dogs_mlops.application.preprocessing import (
    build_eval_transform,
    build_train_transform,
    select_device,
    set_global_seed,
)


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def model_factory(model_name: str) -> nn.Module:
    if model_name != "simple_cnn":
        raise ValueError(f"Unsupported model: {model_name}")
    return SimpleCNN()


@dataclass
class EpochMetrics:
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


@dataclass
class TrainResult:
    history: list[EpochMetrics]
    best_val_accuracy: float
    best_model_path: Path
    class_names: list[str]


def _run_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, train_mode: bool, optimizer: torch.optim.Optimizer | None = None
) -> tuple[float, float]:
    model.train(mode=train_mode)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if train_mode and optimizer:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if train_mode and optimizer:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def optimizer_factory(optimizer_name: str, params: Any, lr: float) -> torch.optim.Optimizer:
    name = optimizer_name.lower().strip()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}. Supported: 'adam', 'sgd'.")


def train_model(
    processed_root: Path,
    image_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    model_name: str,
    seed: int,
    checkpoint_path: Path,
    flip_probability: float,
    rotation_degrees: int,
    optimizer_name: str = "adam",
) -> TrainResult:
    set_global_seed(seed)
    device = select_device()
    train_ds = ImageFolder(
        str(processed_root / "train"),
        transform=build_train_transform(image_size, flip_probability, rotation_degrees),
    )
    val_ds = ImageFolder(str(processed_root / "validation"), transform=build_eval_transform(image_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model_factory(model_name).to(device)
    optimizer = optimizer_factory(optimizer_name, model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: list[EpochMetrics] = []
    best_val = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, train_mode=True, optimizer=optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, train_mode=False)
        history.append(EpochMetrics(train_loss, train_acc, val_loss, val_acc))
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": model_name,
                    "class_names": train_ds.classes,
                    "image_size": image_size,
                    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                    "model_version": os.getenv("MODEL_VERSION", "dev-local"),
                },
                checkpoint_path,
            )

    return TrainResult(history=history, best_val_accuracy=best_val, best_model_path=checkpoint_path, class_names=train_ds.classes)


def load_datasets_for_eval(processed_root: Path, image_size: int, batch_size: int) -> tuple[DataLoader, list[str]]:
    test_ds = ImageFolder(str(processed_root / "test"), transform=build_eval_transform(image_size))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False), test_ds.classes


def load_model_checkpoint(checkpoint_path: Path, device: torch.device | None = None) -> tuple[nn.Module, dict[str, Any]]:
    device = device or select_device()
    payload = torch.load(checkpoint_path, map_location=device)
    model = model_factory(payload["model_name"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload
