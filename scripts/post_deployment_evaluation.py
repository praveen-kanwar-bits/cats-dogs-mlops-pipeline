from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="monitoring/reports/post_deployment_metrics.json")
    args = parser.parse_args()

    y_true: list[int] = []
    y_pred: list[int] = []
    label_to_int = {"cat": 0, "dog": 1}

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_path = Path(row["image_path"])
            true_label = row["true_label"].strip().lower()
            with image_path.open("rb") as img:
                resp = requests.post(f"{args.base_url}/predict", files={"file": img}, timeout=10)
            resp.raise_for_status()
            pred = resp.json()["predicted_class"]
            y_true.append(label_to_int[true_label])
            y_pred.append(label_to_int[pred])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    payload = {
        "samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1)),
        "confusion_matrix": np.asarray(cm).tolist(),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
