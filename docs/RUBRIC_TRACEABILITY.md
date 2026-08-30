# Rubric Traceability

| Marks | Prescribed objective | Exact implementation | Primary evidence | Verification |
|---:|---|---|---|---|
| M1.1 | Git source versioning | Git repository with source, tests, scripts, configuration, docs, and workflow history | `.git/`, `git log --oneline` | Commit the final working tree |
| M1.1 | DVC raw + preprocessed data versioning | Raw pointer plus three-stage DAG and content-addressed lock hashes | `.dvc/`, `data/raw.dvc`, `dvc.yaml`, `dvc.lock` | `dvc status`; `dvc dag` |
| M1.2 | Baseline serialized model | Three-block PyTorch `SimpleCNN`; best validation checkpoint saved as `.pt` | `src/.../training.py`, `artifacts/release/cats_dogs_cnn.pt` | Real checkpoint load is integration-tested |
| M1.3 | Runs, params, metrics, artifacts | MLflow training/evaluation runs log config, epoch/final metrics, model, metadata, history, confusion matrix, curves | `mlflow.db`, `mlruns/`, `scripts/train.py`, `scripts/evaluate.py` | `mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns` |
| Data | 224×224 RGB | Valid images are converted to RGB and physically resized to 224×224 | `scripts/preprocess.py`, `data/processed/` | Manifest/count and image-mode checks |
| Data | 80/10/10 + augmentation | Deterministic stratified 19,998/2,500/2,500 split; flip/rotation/color jitter only in training | `params.yaml`, preprocessing/training modules | Exact-ratio and augmentation unit tests |
| M2.4 | Health + prediction REST API | `/health`, `/ready`, multipart `/predict`; label, confidence, both probabilities, version | `src/.../interfaces/api/app.py` | API integration and container smoke tests |
| M2.5 | Pinned requirements | Serving, ML/tracking, DVC/Kaggle, and test tools use compatible exact pins | `requirements*.txt` | `pip check` and full test suite |
| M2.6 | Docker build/run/curl | Non-root image, checksum-verified trained release, model-aware health check | `Dockerfile` | Local image build/run plus `scripts/smoke_test.py` |
| M3.7 | Automated tests | 16 preprocessing, model/inference, real-checkpoint, API, and monitoring tests | `tests/` | `pytest -v` |
| M3.8 | CI on pushes/PRs | Checkout, install, lint, compile, tests, integrity check, Docker build/run/smoke | `.github/workflows/ci.yml` | Green `test-build` job |
| M3.9 | Registry publishing | Successful trusted `main` pushes publish SHA and `latest` to GHCR | CI `publish` job | Green job + GHCR package tag |
| M4.10 | Docker Compose target | Image, port, version env, restart, readiness health, bounded logs | `docker-compose.yml` | `docker compose config`; running service |
| M4.11 | Automatic main deployment | Successful CI workflow-run triggers exact-SHA pull/update on production self-hosted runner | `.github/workflows/cd.yml` | Green CD job for same SHA |
| M4.12 | Failing post-deploy smoke gate | Readiness plus actual image prediction and schema/probability/version validation | `scripts/smoke_test.py`, CD `Smoke test` step | Any failure exits non-zero |
| M5.13 | Logs + count/latency metrics | JSON request/response-summary logging and Prometheus counters/histogram | observability modules, `/metrics` | Container logs and metrics response |
| M5.14 | Labeled post-deployment batch | 20 distinct test images, true labels, per-sample outputs, classification metrics and CM | `tests/fixtures/eval_batch/`, `scripts/post_deployment_evaluation.py`, monitoring report | Run evaluator against deployed API |
| M5.15 | ZIP + <5 min recording | Deterministic artifact ZIP builder and timed video script/checklist | `scripts/create_submission.py`, `docs/DEMO_SCRIPT.md`, `docs/SUBMISSION_CHECKLIST.md` | Inspect ZIP; submit actual video |

Repository-side target: 50/50. Final award additionally depends on visible green remote CI/CD runs and
submission of the real under-five-minute recording.
