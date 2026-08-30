from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from cats_dogs_mlops.domain.entities import PredictionResult
from cats_dogs_mlops.interfaces.api.app import create_app


def test_health_endpoint():
    app = create_app(Path("params.yaml"))
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_ready_endpoint_uninitialized():
    app = create_app(Path("params.yaml"))
    client = TestClient(app)
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"


def test_ready_and_predict_with_loaded_model():
    app = create_app(Path("params.yaml"))
    mock_service = MagicMock()
    mock_service.predict_bytes.return_value = PredictionResult(
        predicted_class="cat",
        confidence=0.95,
        probabilities={"cat": 0.95, "dog": 0.05},
        model_version="test-mock",
    )
    app.state.inference_service = mock_service
    app.state.model_loaded = True

    client = TestClient(app)

    resp_ready = client.get("/ready")
    assert resp_ready.status_code == 200
    assert resp_ready.json()["status"] == "ready"

    cat_bytes = Path("tests/fixtures/sample_cat.jpg").read_bytes()
    resp_pred = client.post("/predict", files={"file": ("cat.jpg", cat_bytes, "image/jpeg")})
    assert resp_pred.status_code == 200
    body = resp_pred.json()
    assert body["predicted_class"] == "cat"
    assert body["confidence"] == 0.95
    assert body["probabilities"] == {"cat": 0.95, "dog": 0.05}


def test_predict_invalid_image_payload():
    app = create_app(Path("params.yaml"))
    mock_service = MagicMock()
    mock_service.predict_bytes.side_effect = ValueError("Invalid image payload")
    app.state.inference_service = mock_service
    app.state.model_loaded = True

    client = TestClient(app)
    resp = client.post("/predict", files={"file": ("bad.txt", b"not_an_image", "text/plain")})
    assert resp.status_code == 400
    assert "Invalid image payload" in resp.json()["detail"]


def test_metrics_endpoint():
    app = create_app(Path("params.yaml"))
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "ml_api_requests_total" in resp.text
    assert "ml_api_request_latency_seconds" in resp.text


def test_real_release_model_loads_and_predicts(monkeypatch):
    release_model = Path("artifacts/release/cats_dogs_cnn.pt")
    assert release_model.is_file()
    monkeypatch.setenv("MODEL_PATH", str(release_model))

    app = create_app(Path("params.yaml"))
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["model_loaded"] is True

        cat_bytes = Path("tests/fixtures/sample_cat.jpg").read_bytes()
        response = client.post(
            "/predict", files={"file": ("cat.jpg", cat_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_class"] in {"cat", "dog"}
        assert set(body["probabilities"]) == {"cat", "dog"}
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-5
        assert body["model_version"]
