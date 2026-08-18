from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-slug", default=os.getenv("KAGGLE_DATASET_SLUG", ""))
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    if not args.dataset_slug:
        raise SystemExit("Provide --dataset-slug or KAGGLE_DATASET_SLUG")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
