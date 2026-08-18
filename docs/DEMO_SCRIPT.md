# Demo Script (<=4:30)

- 0:00–0:25 show repo structure + `docs/ARCHITECTURE.md`
- 0:25–0:55 run `dvc repro preprocess`
- 0:55–1:25 run `dvc repro train` and show `mlflow ui`
- 1:25–1:50 run `pytest -v`
- 1:50–2:15 run `docker build -t cats-dogs-mlops-api:local .` and local API
- 2:15–3:20 show CI/CD workflow files and recent runs
- 3:20–3:50 run curl prediction on deployed API
- 3:50–4:15 show `/metrics`
- 4:15–4:30 run post-deployment evaluation script and show JSON report
