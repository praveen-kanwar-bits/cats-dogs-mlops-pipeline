from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from cats_dogs_mlops.application.training import train_model
from cats_dogs_mlops.config import load_config
from cats_dogs_mlops.infrastructure.models.pytorch_model import write_metadata
from cats_dogs_mlops.infrastructure.tracking.mlflow_tracker import MlflowTracker


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_data_version() -> str:
    lock_file = Path("dvc.lock")
    if lock_file.exists():
        import hashlib
        return hashlib.md5(lock_file.read_bytes()).hexdigest()[:12]
    raw_dvc = Path("data/raw.dvc")
    if raw_dvc.exists():
        import hashlib
        return hashlib.md5(raw_dvc.read_bytes()).hexdigest()[:12]
    commit = git_commit()
    return f"git-{commit[:8]}" if commit != "unknown" else "dev-unversioned"


def main() -> int:
    cfg = load_config(Path("params.yaml"))
    tracker = MlflowTracker(cfg.mlflow.experiment_name)
    model_path = Path("artifacts/model/cats_dogs_cnn.pt")
    metadata_path = Path("artifacts/model/metadata.json")

    with tracker.start_run() as run:
        tracker.log_params(
            {
                "model_name": cfg.training.model_name,
                "learning_rate": cfg.training.learning_rate,
                "batch_size": cfg.training.batch_size,
                "epochs": cfg.training.epochs,
                "optimizer": cfg.training.optimizer,
                "image_size": cfg.data.image_size,
                "random_seed": cfg.training.random_seed,
                "augmentation": json.dumps(cfg.augmentation.__dict__),
            }
        )
        result = train_model(
            processed_root=Path("data/processed"),
            image_size=cfg.data.image_size,
            batch_size=cfg.training.batch_size,
            epochs=cfg.training.epochs,
            learning_rate=cfg.training.learning_rate,
            model_name=cfg.training.model_name,
            seed=cfg.training.random_seed,
            checkpoint_path=model_path,
            flip_probability=cfg.augmentation.horizontal_flip_probability,
            rotation_degrees=cfg.augmentation.rotation_degrees,
            optimizer_name=cfg.training.optimizer,
        )

        for idx, metrics in enumerate(result.history, start=1):
            tracker.log_metrics(
                {
                    "train_loss": metrics.train_loss,
                    "train_accuracy": metrics.train_accuracy,
                    "validation_loss": metrics.validation_loss,
                    "validation_accuracy": metrics.validation_accuracy,
                },
                step=idx,
            )
        history_path = Path("artifacts/model/training_history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps([m.__dict__ for m in result.history], indent=2),
            encoding="utf-8",
        )

        metadata = {
            "model_version": os.getenv("MODEL_VERSION", "dev-local"),
            "model_name": cfg.training.model_name,
            "class_names": result.class_names,
            "git_commit": git_commit(),
            "mlflow_run_id": run.info.run_id,
            "image_size": cfg.data.image_size,
            "data_version": get_data_version(),
        }
        write_metadata(metadata, metadata_path)
        tracker.log_artifact(model_path)
        tracker.log_artifact(metadata_path)
        tracker.log_artifact(history_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
