from __future__ import annotations

from pathlib import Path

import mlflow


class MlflowTracker:
    def __init__(self, experiment_name: str) -> None:
        mlflow.set_experiment(experiment_name)

    def start_run(self):
        return mlflow.start_run()

    @staticmethod
    def log_params(params: dict) -> None:
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict, step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    @staticmethod
    def log_artifact(path: Path) -> None:
        mlflow.log_artifact(str(path))
