# Rubric Traceability

| Requirement | Implementation | Files | Command | Evidence | Status |
|---|---|---|---|---|---|
| M1 DVC+preprocess+split | DVC stages and preprocessing scripts | `dvc.yaml`, `scripts/preprocess.py` | `dvc repro preprocess` | `data/processed/split_manifest.csv` | Implemented |
| M1 training+MLflow | SimpleCNN training + tracker | `scripts/train.py`, `src/.../training.py` | `dvc repro train` | MLflow run + `.pt` + metadata | Implemented |
| M1 evaluation artifacts | evaluator + plots + metrics | `scripts/evaluate.py`, `src/.../evaluation.py` | `dvc repro evaluate` | JSON + PNG artifacts | Implemented |
| M2 API endpoints | FastAPI `/health` `/predict` `/metrics` | `src/.../interfaces/api/app.py` | `uvicorn ...` | JSON responses + Prom metrics | Implemented |
| M2 Docker/pins | pinned reqs + Dockerfile | `requirements*.txt`, `Dockerfile` | `docker build ...` | image build log | Implemented |
| M3 tests | preprocessing/model/inference/API tests | `tests/` | `pytest -v` | pytest report | Implemented |
| M3 CI + GHCR SHA tags | CI workflow with conditional publish | `.github/workflows/ci.yml` | GitHub Actions | workflow runs | Implemented |
| M4 CD compose deploy | workflow_run + compose + smoke | `.github/workflows/cd.yml`, `docker-compose.yml` | GitHub Actions CD | deploy + smoke logs | Implemented |
| M5 monitoring | structured logs + Prom counters/histograms | `src/.../observability/*` | `curl /metrics` | metrics output | Implemented |
| M5 production evaluation | traffic + post-deploy eval script | `scripts/simulate_traffic.py`, `scripts/post_deployment_evaluation.py` | python scripts | JSON report | Implemented |
