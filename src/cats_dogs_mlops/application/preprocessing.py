from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

from cats_dogs_mlops.constants import CLASS_NAMES


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def open_as_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid image file: {path}") from exc


def build_train_transform(image_size: int, flip_p: float, rotation: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.RandomRotation(degrees=rotation),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collect_image_paths(raw_dir: Path) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    for label in CLASS_NAMES:
        class_dir = raw_dir / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in class_dir.rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                records.append((path, label))
    if not records:
        raise ValueError("No image files found in raw dataset")
    return records


def make_split_manifest(
    records: Iterable[tuple[Path, str]],
    output_csv: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    from sklearn.model_selection import train_test_split

    rows = [(str(path), label) for path, label in records]
    filepaths = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        filepaths,
        labels,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    val_share = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        train_size=val_share,
        random_state=seed,
        stratify=temp_labels,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "split"])
        writer.writerows([(p, label, "train") for p, label in zip(train_paths, train_labels)])
        writer.writerows([(p, label, "validation") for p, label in zip(val_paths, val_labels)])
        writer.writerows([(p, label, "test") for p, label in zip(test_paths, test_labels)])
