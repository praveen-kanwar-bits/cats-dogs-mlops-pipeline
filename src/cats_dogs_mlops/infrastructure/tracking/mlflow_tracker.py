from __future__ import annotations

import os
from pathlib import Path

import mlflow


class MlflowTracker:
    def __init__(self, experiment_name: str) -> None:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None):
        return mlflow.start_run(run_name=run_name, tags=tags)

    @staticmethod
    def log_params(params: dict) -> None:
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict, step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    @staticmethod
    def log_artifact(path: Path) -> None:
        mlflow.log_artifact(str(path))

    @staticmethod
    def set_tags(tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)
