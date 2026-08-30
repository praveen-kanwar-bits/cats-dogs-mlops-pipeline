from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def normalize_raw_structure(output_dir: Path) -> None:
    cat_dir = output_dir / "cat"
    dog_dir = output_dir / "dog"
    cat_dir.mkdir(parents=True, exist_ok=True)
    dog_dir.mkdir(parents=True, exist_ok=True)

    for item in list(output_dir.rglob("*")):
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            rel_parts = [p.lower() for p in item.relative_to(output_dir).parts]
            if item.parent == cat_dir or item.parent == dog_dir:
                continue

            target_dir = None
            if any(part in {"cat", "cats"} for part in rel_parts):
                target_dir = cat_dir
            elif any(part in {"dog", "dogs"} for part in rel_parts):
                target_dir = dog_dir

            if target_dir:
                dest = target_dir / item.name
                if dest != item and not dest.exists():
                    shutil.move(str(item), str(dest))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-slug", default=os.getenv("KAGGLE_DATASET_SLUG", ""))
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_slug:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                args.dataset_slug,
                "-p",
                str(output_dir),
                "--unzip",
            ],
            check=True,
        )

    normalize_raw_structure(output_dir)
    missing = [label for label in ("cat", "dog") if not any((output_dir / label).glob("*"))]
    if missing:
        raise SystemExit(
            "Dataset is incomplete. Supply --dataset-slug (and Kaggle credentials) or place "
            f"images under data/raw/cat and data/raw/dog. Missing: {', '.join(missing)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
