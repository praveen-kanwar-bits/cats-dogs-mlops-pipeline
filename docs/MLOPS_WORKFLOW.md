# MLOps Workflow

1. Download and normalize Kaggle data: `python scripts/download_dataset.py --dataset-slug <slug>`.
2. Version raw data: `dvc add data/raw`; commit `data/raw.dvc` and `data/.gitignore`.
3. Reproduce `preprocess -> train -> evaluate`: `dvc repro`.
4. Inspect metrics and experiments: `dvc metrics show`; `mlflow ui --backend-store-uri
   sqlite:///mlflow.db --default-artifact-root ./mlruns`.
5. Promote only the evaluated model: `python scripts/promote_model.py`; commit
   `artifacts/release/`.
6. Verify code: `ruff check src scripts tests`; `pytest -v`.
7. Verify image locally: build, run, then `python scripts/smoke_test.py`.
8. Push the commit. CI tests/builds/runs the image and publishes the SHA tag to GHCR on `main`.
9. Successful `main` CI triggers CD, which pulls the exact SHA into Docker Compose and smoke-tests
   the deployed service.
10. Generate traffic and labeled performance evidence with `scripts/simulate_traffic.py` and
    `scripts/post_deployment_evaluation.py`.
11. Create the deliverable: `python scripts/create_submission.py`.
