from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import requests


def wait_for_health(base_url: str, retries: int, delay: float) -> None:
    for _ in range(retries):
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200 and resp.json().get("model_loaded"):
                return
        except requests.RequestException:
            pass
        time.sleep(delay)
    raise SystemExit("Health check failed")


def run_predict(base_url: str, image_path: Path) -> None:
    with image_path.open("rb") as f:
        resp = requests.post(f"{base_url}/predict", files={"file": f}, timeout=10)
    if resp.status_code != 200:
        raise SystemExit(f"Predict failed: {resp.status_code} {resp.text}")
    data = resp.json()
    probs = data.get("probabilities", {})
    if data.get("predicted_class") not in {"cat", "dog"}:
        raise SystemExit("Invalid class")
    if not (0 <= data.get("confidence", -1) <= 1):
        raise SystemExit("Invalid confidence")
    if not math.isclose(sum(probs.values()), 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise SystemExit("Probabilities do not sum to 1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--image", default="tests/fixtures/sample_cat.jpg")
    parser.add_argument("--retries", type=int, default=30)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    wait_for_health(args.base_url, args.retries, args.delay)
    run_predict(args.base_url, Path(args.image))
    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
