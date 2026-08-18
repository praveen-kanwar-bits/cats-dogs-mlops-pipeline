from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--cat-image", default="tests/fixtures/sample_cat.jpg")
    parser.add_argument("--dog-image", default="tests/fixtures/sample_dog.jpg")
    args = parser.parse_args()

    images = [Path(args.cat_image), Path(args.dog_image)]
    ok = 0
    for idx in range(args.requests):
        image_path = images[idx % len(images)]
        with image_path.open("rb") as f:
            resp = requests.post(f"{args.base_url}/predict", files={"file": f}, timeout=10)
        if resp.status_code == 200:
            ok += 1
    print({"total": args.requests, "successful": ok, "failed": args.requests - ok})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
