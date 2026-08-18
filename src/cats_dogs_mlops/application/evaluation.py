from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from cats_dogs_mlops.application.preprocessing import select_device


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, class_names: list[str]) -> dict[str, float | list[list[int]]]:
    device = select_device()
    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", pos_label=1)),
        "confusion_matrix": cm.tolist(),
    }


def save_test_metrics(metrics: dict[str, float | list[list[int]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_confusion_matrix(confusion_matrix: list[list[int]], class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(confusion_matrix)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=arr, display_labels=class_names).plot(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history) + 1))
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["validation_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["validation_accuracy"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="validation_loss")
    axes[0].legend()
    axes[1].plot(epochs, train_acc, label="train_accuracy")
    axes[1].plot(epochs, val_acc, label="validation_accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
