from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a validated training checkpoint for CI")
    parser.add_argument("--source-dir", default="artifacts/model")
    parser.add_argument("--release-dir", default="artifacts/release")
    parser.add_argument("--metrics", default="artifacts/evaluation/test_metrics.json")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    release_dir = Path(args.release_dir)
    source_model = source_dir / "cats_dogs_cnn.pt"
    source_metadata = source_dir / "metadata.json"
    metrics_path = Path(args.metrics)

    if not all(path.is_file() for path in (source_model, source_metadata, metrics_path)):
        raise SystemExit("Training checkpoint, metadata, and evaluation metrics must all exist")

    payload = torch.load(source_model, map_location="cpu")
    required_checkpoint_keys = {"state_dict", "model_name", "class_names", "image_size"}
    if not required_checkpoint_keys.issubset(payload):
        raise SystemExit("Checkpoint is missing required serving fields")
    if payload["class_names"] != ["cat", "dog"] or payload["model_name"] != "simple_cnn":
        raise SystemExit("Checkpoint class mapping or architecture is invalid")

    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metadata.get("mlflow_run_id") in {None, "", "bootstrap-run"}:
        raise SystemExit("Refusing to promote a checkpoint without a real MLflow training run")

    release_dir.mkdir(parents=True, exist_ok=True)
    release_model = release_dir / "cats_dogs_cnn.pt"
    shutil.copy2(source_model, release_model)

    release_metadata = {
        **metadata,
        "promoted_at_utc": datetime.now(UTC).isoformat(),
        "source_checkpoint_sha256": sha256(source_model),
        "test_metrics": metrics,
    }
    release_metadata_path = release_dir / "metadata.json"
    release_metadata_path.write_text(json.dumps(release_metadata, indent=2), encoding="utf-8")

    checksum_path = release_dir / "SHA256SUMS"
    checksum_path.write_text(
        "\n".join(
            [
                f"{sha256(release_model)}  cats_dogs_cnn.pt",
                f"{sha256(release_metadata_path)}  metadata.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Promoted evaluated checkpoint to {release_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
