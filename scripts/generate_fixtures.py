from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image


def select_images(class_dir: Path, count: int, seed: int) -> list[Path]:
    candidates = sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if len(candidates) < count:
        raise ValueError(f"Need {count} images in {class_dir}, found {len(candidates)}")
    return random.Random(seed).sample(candidates, count)


def generate() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/processed/test")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fixtures_dir = Path("tests/fixtures/eval_batch")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[tuple[str, str]] = []
    source_root = Path(args.source_root)
    for label_index, label in enumerate(("cat", "dog")):
        sources = select_images(
            source_root / label,
            count=args.per_class,
            seed=args.seed + label_index,
        )
        for index, source in enumerate(sources, start=1):
            destination = fixtures_dir / f"{label}_{index:02d}.jpg"
            with Image.open(source) as image:
                prepared = image.convert("RGB").resize((224, 224))
                prepared.save(destination, format="JPEG", quality=85, optimize=True)
                if index == 1:
                    prepared.save(
                        fixtures_dir.parent / f"sample_{label}.jpg",
                        format="JPEG",
                        quality=85,
                        optimize=True,
                    )
            manifest_rows.append((destination.as_posix(), label))

    manifest_path = Path("tests/fixtures/post_deploy_manifest.csv")
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["image_path", "true_label"])
        writer.writerows(manifest_rows)

    print(
        f"Generated {len(manifest_rows)} distinct labeled fixtures and manifest at {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(generate())
