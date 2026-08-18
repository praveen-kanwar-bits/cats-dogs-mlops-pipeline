# MLOps Workflow

1. `python scripts/download_dataset.py --dataset-slug <slug>`
2. `dvc repro` (preprocess -> train -> evaluate)
3. `mlflow ui`
4. `uvicorn cats_dogs_mlops.interfaces.api.app:app --host 0.0.0.0 --port 8000`
5. `python scripts/smoke_test.py`
