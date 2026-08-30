from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="monitoring/reports/post_deployment_metrics.json")
    parser.add_argument("--minimum-samples", type=int, default=10)
    parser.add_argument("--minimum-accuracy", type=float, default=0.0)
    args = parser.parse_args()

    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict[str, str | float]] = []
    model_versions: set[str] = set()
    label_to_int = {"cat": 0, "dog": 1}

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_path = Path(row["image_path"])
            true_label = row["true_label"].strip().lower()
            with image_path.open("rb") as img:
                resp = requests.post(f"{args.base_url}/predict", files={"file": img}, timeout=10)
            resp.raise_for_status()
            response = resp.json()
            pred = response["predicted_class"]
            if true_label not in label_to_int or pred not in label_to_int:
                raise ValueError(f"Unsupported label in request/response: {true_label=}, {pred=}")
            y_true.append(label_to_int[true_label])
            y_pred.append(label_to_int[pred])
            model_versions.add(response.get("model_version", "unknown"))
            predictions.append(
                {
                    "image": image_path.name,
                    "true_label": true_label,
                    "predicted_label": pred,
                    "confidence": float(response["confidence"]),
                }
            )

    if len(y_true) < args.minimum_samples:
        raise SystemExit(
            f"Evaluation requires at least {args.minimum_samples} samples; received {len(y_true)}"
        )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    payload = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "samples": len(y_true),
        "model_versions": sorted(model_versions),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": np.asarray(cm).tolist(),
        "predictions": predictions,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["accuracy"] < args.minimum_accuracy:
        raise SystemExit(
            f"Accuracy {payload['accuracy']:.4f} is below threshold {args.minimum_accuracy:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
