from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_for_health(base_url: str, retries: int, delay: float) -> None:
    url = f"{base_url}/health"
    last_error = "service did not report model_loaded=true"
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("model_loaded"):
                        return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(delay)
    raise SystemExit(f"Health check failed: {last_error}")


def post_multipart_file(url: str, file_path: Path) -> dict:
    boundary = "----MLOpsSmokeTestBoundary"
    filename = file_path.name
    data = file_path.read_bytes()

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(data)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_msg = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"Predict failed: {exc.code} {err_msg}") from exc


def run_predict(base_url: str, image_path: Path) -> None:
    url = f"{base_url}/predict"
    data = post_multipart_file(url, image_path)

    probs = data.get("probabilities", {})
    if data.get("predicted_class") not in {"cat", "dog"}:
        raise SystemExit(f"Invalid predicted class: {data.get('predicted_class')}")
    if not (0 <= data.get("confidence", -1) <= 1):
        raise SystemExit(f"Invalid confidence: {data.get('confidence')}")
    if not math.isclose(sum(probs.values()), 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise SystemExit("Probabilities do not sum to 1.0")
    if set(probs) != {"cat", "dog"}:
        raise SystemExit(f"Expected cat/dog probabilities, received: {sorted(probs)}")
    if not data.get("model_version"):
        raise SystemExit("Prediction response is missing model_version")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--image", default="tests/fixtures/sample_cat.jpg")
    parser.add_argument("--retries", type=int, default=30)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    wait_for_health(args.base_url, args.retries, args.delay)
    run_predict(args.base_url, Path(args.image))
    print("Smoke test passed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
