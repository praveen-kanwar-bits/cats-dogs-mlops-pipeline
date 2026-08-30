from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from cats_dogs_mlops.application.inference import InferenceService
from cats_dogs_mlops.config import load_config
from cats_dogs_mlops.infrastructure.models.pytorch_model import load_model_bundle
from cats_dogs_mlops.infrastructure.observability.logging import configure_logging, log_json
from cats_dogs_mlops.infrastructure.observability.metrics import (
    ERRORS_TOTAL,
    PREDICTIONS_TOTAL,
    REQUEST_LATENCY_SECONDS,
    REQUESTS_TOTAL,
)

LOGGER = logging.getLogger("cats_dogs_mlops_api")


def create_app(config_path: Path = Path("params.yaml")) -> FastAPI:
    configure_logging()
    cfg = load_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model_path = Path(cfg.serving.model_path)
        if model_path.exists():
            model, payload = load_model_bundle(model_path)
            app.state.inference_service = InferenceService(
                model=model,
                class_names=payload["class_names"],
                image_size=payload.get("image_size", cfg.data.image_size),
                model_version=os.getenv(
                    "MODEL_VERSION", payload.get("model_version", "unknown")
                ),
            )
            app.state.model_loaded = True
        yield

    app = FastAPI(title="cats-dogs-mlops-api", lifespan=lifespan)
    app.state.inference_service = None
    app.state.model_loaded = False

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            if status_code >= 400:
                ERRORS_TOTAL.labels(path=request.url.path).inc()
            return response
        except Exception:
            ERRORS_TOTAL.labels(path=request.url.path).inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            REQUEST_LATENCY_SECONDS.labels(method=request.method, path=request.url.path).observe(elapsed)
            REQUESTS_TOTAL.labels(
                method=request.method,
                path=request.url.path,
                status_code=str(status_code),
            ).inc()
            log_json(
                LOGGER,
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round(elapsed * 1000, 2),
                },
            )

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy", "model_loaded": app.state.model_loaded}

    @app.get("/ready")
    def ready() -> JSONResponse:
        if not app.state.model_loaded:
            return JSONResponse(status_code=503, content={"status": "not_ready", "model_loaded": False})
        return JSONResponse(status_code=200, content={"status": "ready", "model_loaded": True})

    @app.post("/predict")
    async def predict(request: Request, file: UploadFile = File(...)) -> dict:  # noqa: B008
        if not app.state.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        try:
            result = app.state.inference_service.predict_bytes(image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        PREDICTIONS_TOTAL.labels(predicted_class=result.predicted_class).inc()
        log_json(
            LOGGER,
            {
                "event": "prediction",
                "request_id": getattr(request.state, "request_id", "unknown"),
                "predicted_class": result.predicted_class,
                "confidence": result.confidence,
                "model_version": result.model_version,
            },
        )
        return {
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "model_version": result.model_version,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
