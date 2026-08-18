from pathlib import Path

from fastapi.testclient import TestClient

from cats_dogs_mlops.interfaces.api.app import create_app


def test_health_endpoint():
    client = TestClient(create_app(Path("params.yaml")))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "model_loaded" in resp.json()


def test_predict_invalid_image_type():
    client = TestClient(create_app(Path("params.yaml")))
    resp = client.post("/predict", files={"file": ("bad.txt", b"abc", "text/plain")})
    assert resp.status_code == 503 or resp.status_code == 400
