# cats-dogs-mlops-pipeline

Production-quality reference MLOps pipeline for Cats vs Dogs image classification with DVC, PyTorch, MLflow, FastAPI, Docker, GitHub Actions CI/CD, GHCR, Docker Compose, and monitoring.

## Architecture
Pragmatic clean architecture in `src/cats_dogs_mlops`:
- `domain`: entities and contracts
- `application`: preprocessing/training/evaluation/inference use cases
- `infrastructure`: model IO, MLflow tracking, observability
- `interfaces`: FastAPI service and scripts

See `docs/ARCHITECTURE.md`.

## Prerequisites
- Python 3.11
- pip
- make (optional)
- Docker (for container/deployment checks)
- DVC
- Kaggle CLI + credentials (dataset download)

## Install
```bash
pip install -r requirements-dev.txt
pip install -e .
```

## Configuration
Primary ML config: `params.yaml`.
Environment template: `.env.example`.

## Dataset setup
```bash
python scripts/download_dataset.py --dataset-slug "$KAGGLE_DATASET_SLUG"
```
Expected normalized raw layout: `data/raw/cats` and `data/raw/dogs` (or equivalent class dirs mapped into `cat/dog`).

## DVC workflow
```bash
dvc repro
# or by stage:
dvc repro preprocess
dvc repro train
dvc repro evaluate
```

## Training and MLflow
```bash
python scripts/train.py
python scripts/evaluate.py
mlflow ui
```

## API
```bash
uvicorn cats_dogs_mlops.interfaces.api.app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -F "file=@tests/fixtures/sample_cat.jpg" http://localhost:8000/predict
curl http://localhost:8000/metrics
```

## Tests and quality
```bash
ruff check src scripts tests
python -m compileall src scripts tests
pytest -v
```

## Docker
```bash
docker build -t cats-dogs-mlops-api:local .
docker run --rm -p 8000:8000 cats-dogs-mlops-api:local
```

## Docker Compose deployment
```bash
IMAGE_NAME=ghcr.io/<owner>/cats-dogs-mlops-pipeline IMAGE_TAG=<git-sha> PORT=8000 docker compose pull
IMAGE_NAME=ghcr.io/<owner>/cats-dogs-mlops-pipeline IMAGE_TAG=<git-sha> PORT=8000 docker compose up -d
python scripts/smoke_test.py --base-url http://localhost:8000
```

## CI/CD
- CI: `.github/workflows/ci.yml` (lint, compile, tests, docker build, GHCR publish on trusted `main` push).
- CD: `.github/workflows/cd.yml` (trigger on successful CI for `main`, deploy via compose on self-hosted runner, smoke test gate).

## Monitoring and production evaluation
```bash
python scripts/simulate_traffic.py --base-url http://localhost:8000 --requests 50
python scripts/post_deployment_evaluation.py --base-url http://localhost:8000 --manifest tests/fixtures/post_deploy_manifest.csv
```
Output: `monitoring/reports/post_deployment_metrics.json`.

## Make targets
`make install test lint preprocess train evaluate mlflow docker-build docker-up docker-down smoke simulate post-deploy-eval`

## Verification boundaries
- **Automatically verified locally:** lint/compile/pytest.
- **Requires dataset:** end-to-end `dvc repro` training/evaluation artifacts.
- **Requires credentials:** Kaggle download, GHCR publish.
- **Requires GitHub setup:** Actions permissions and secrets.
- **Requires self-hosted runner:** CD deployment execution.

## Additional docs
- `docs/ARCHITECTURE.md`
- `docs/MLOPS_WORKFLOW.md`
- `docs/RUBRIC_TRACEABILITY.md`
- `docs/DEMO_SCRIPT.md`
- `docs/TROUBLESHOOTING.md`
- `MODEL_CARD.md`
- `PROJECT_STATUS.md`
