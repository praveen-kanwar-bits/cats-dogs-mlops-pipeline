from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cats_dogs_mlops.application.preprocessing import collect_image_paths, make_split_manifest, open_as_rgb
from cats_dogs_mlops.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    records = collect_image_paths(Path(args.raw_dir))
    manifest_path = Path(args.processed_dir) / "split_manifest.csv"
    make_split_manifest(
        records,
        manifest_path,
        cfg.data.train_ratio,
        cfg.data.validation_ratio,
        cfg.data.test_ratio,
        cfg.data.random_seed,
    )

    processed = Path(args.processed_dir)
    for split in ["train", "validation", "test"]:
        for label in ["cat", "dog"]:
            (processed / split / label).mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = Path(row["filepath"])
            label = row["label"]
            split = row["split"]
            try:
                target = processed / split / label / src.name
                image = open_as_rgb(src).resize((cfg.data.image_size, cfg.data.image_size))
                image.save(target)
            except ValueError as exc:
                print(f"Skipping corrupt image {src}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
