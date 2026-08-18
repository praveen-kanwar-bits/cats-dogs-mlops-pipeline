from __future__ import annotations

import json
from pathlib import Path

from cats_dogs_mlops.application.evaluation import (
    evaluate_model,
    save_confusion_matrix,
    save_test_metrics,
    save_training_curves,
)
from cats_dogs_mlops.application.training import load_datasets_for_eval, load_model_checkpoint
from cats_dogs_mlops.config import load_config
from cats_dogs_mlops.infrastructure.tracking.mlflow_tracker import MlflowTracker


def main() -> int:
    cfg = load_config(Path("params.yaml"))
    test_loader, class_names = load_datasets_for_eval(
        Path("data/processed"),
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
    )
    model, _ = load_model_checkpoint(Path("artifacts/model/cats_dogs_cnn.pt"))
    metrics = evaluate_model(model, test_loader, class_names)

    metrics_path = Path("artifacts/evaluation/test_metrics.json")
    cm_path = Path("artifacts/evaluation/confusion_matrix.png")
    curves_path = Path("artifacts/evaluation/training_curves.png")
    history_path = Path("artifacts/model/training_history.json")
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    save_test_metrics(metrics, metrics_path)
    save_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path)
    save_training_curves(history, curves_path)

    tracker = MlflowTracker(cfg.mlflow.experiment_name)
    with tracker.start_run():
        tracker.log_metrics(
            {
                "test_accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )
        tracker.log_artifact(metrics_path)
        tracker.log_artifact(cm_path)
        if history:
            tracker.log_artifact(curves_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
