# cats-dogs-mlops-pipeline

End-to-end MLOps pipeline for Kaggle Cats vs Dogs classification using DVC, PyTorch,
MLflow, FastAPI, Docker, GitHub Actions, GHCR, Docker Compose, Prometheus metrics, and
post-deployment performance tracking.

## Verified result

- Dataset: 24,998 valid images after corrupt-file filtering
- Split: 19,998 train / 2,500 validation / 2,500 held-out test (80%/10%/10%)
- Input: physically preprocessed 224×224 RGB images
- Baseline: `SimpleCNN`, 0.7548 test accuracy and 0.7517 test F1
- Automated checks: 16 tests plus lint and compilation
- Evaluated release: `artifacts/release/cats_dogs_cnn.pt`, protected by SHA-256 checksums

See `docs/EVALUATOR_REPORT.md` for the full 50/50 rubric assessment and
`docs/SUBMISSION_CHECKLIST.md` for external submission steps.

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
python scripts/promote_model.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

The training model remains a DVC output. `promote_model.py` only promotes a structurally valid,
evaluated checkpoint with a real MLflow run ID into the Git-tracked release directory used by CI.

## API
```bash
uvicorn cats_dogs_mlops.interfaces.api.app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl http://localhost:8000/ready
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
# in another terminal
python scripts/smoke_test.py --base-url http://localhost:8000
```

## Docker Compose deployment
```bash
IMAGE_NAME=ghcr.io/<owner>/cats-dogs-mlops-pipeline IMAGE_TAG=<git-sha> PORT=8000 docker compose pull
IMAGE_NAME=ghcr.io/<owner>/cats-dogs-mlops-pipeline IMAGE_TAG=<git-sha> PORT=8000 docker compose up -d
python scripts/smoke_test.py --base-url http://localhost:8000
```

## CI/CD
- CI: `.github/workflows/ci.yml` (lint, compile, 16 tests, release integrity, Docker build/run,
  smoke test, GHCR publish on trusted `main` push).
- CD: `.github/workflows/cd.yml` (successful `main` CI only, exact-SHA Compose deployment on a
  self-hosted runner, mandatory post-deploy smoke gate).
- One-time GitHub configuration: `docs/GITHUB_SETUP.md`.

## Monitoring and production evaluation
```bash
python scripts/simulate_traffic.py --base-url http://localhost:8000 --requests 50
python scripts/post_deployment_evaluation.py --base-url http://localhost:8000 --manifest tests/fixtures/post_deploy_manifest.csv
```
Output: `monitoring/reports/post_deployment_metrics.json`.

The checked-in manifest references 20 distinct, labeled images sampled deterministically from the
held-out test set—not repeated copies of two fixtures.

## Create final submission ZIP

```bash
python scripts/create_submission.py
unzip -l dist/cats-dogs-mlops-assignment-2.zip
```

The archive deliberately excludes the 1.6 GB raw dataset and local caches but includes DVC
pointers/lock data, all code/configuration, the trained release checkpoint, experiment evidence,
evaluation artifacts, tests, and monitoring output.

## Make targets
`make install test lint preprocess train evaluate mlflow docker-build docker-up docker-down smoke simulate post-deploy-eval`

## Verification boundary

Local code, model, data pipeline, API, container, and monitoring checks are reproducible from this
package. The student must still push the final commit to GitHub, capture green CI/GHCR/CD evidence,
and submit the required screen recording; source code cannot manufacture those external records.

## Additional docs
- `docs/ARCHITECTURE.md`
- `docs/MLOPS_WORKFLOW.md`
- `docs/RUBRIC_TRACEABILITY.md`
- `docs/DEMO_SCRIPT.md`
- `docs/TROUBLESHOOTING.md`
- `docs/EVALUATOR_REPORT.md`
- `docs/GITHUB_SETUP.md`
- `docs/SUBMISSION_CHECKLIST.md`
- `MODEL_CARD.md`
- `PROJECT_STATUS.md`
