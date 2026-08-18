from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    image_size: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    random_seed: int


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float
    rotation_degrees: int


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    random_seed: int


@dataclass(frozen=True)
class MlflowConfig:
    experiment_name: str


@dataclass(frozen=True)
class ServingConfig:
    model_path: str
    host: str
    port: int


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    mlflow: MlflowConfig
    serving: ServingConfig


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text())
    data = DataConfig(**raw["data"])
    total = data.train_ratio + data.validation_ratio + data.test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("data split ratios must sum to 1")
    return AppConfig(
        data=data,
        augmentation=AugmentationConfig(**raw["augmentation"]),
        training=TrainingConfig(**raw["training"]),
        mlflow=MlflowConfig(**raw["mlflow"]),
        serving=ServingConfig(**raw["serving"]),
    )
