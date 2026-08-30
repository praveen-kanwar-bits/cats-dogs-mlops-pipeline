# Demo Script (target 4:35; hard limit 4:59)

Prepare before recording: final commit pushed, CI/CD green, MLflow UI open, deployed API reachable,
and terminal commands pre-entered. Do not wait for a ten-epoch training or image build during the
video; show the completed run tied to the same Git SHA.

- **0:00–0:25 — Identity and architecture.** Show assignment title, final `git rev-parse --short
  HEAD`, repository tree, and `docs/ARCHITECTURE.md`.
- **0:25–0:55 — Data/DVC.** Show `data/raw.dvc`, `params.yaml` (224 and 80/10/10), `dvc dag`, clean
  `dvc status`, and split counts from `split_manifest.csv`.
- **0:55–1:25 — Model/experiments.** Show `SimpleCNN`, trained release metadata/checksum, then the
  MLflow training params/epoch metrics and evaluation confusion-matrix/training-curve artifacts.
- **1:25–1:45 — Tests.** Show the final `pytest -v` summary with 16 passing tests and the tests for
  preprocessing plus inference.
- **1:45–2:25 — CI and registry.** In GitHub, show the final SHA’s green CI test/build/container-smoke
  job, green publish job, and the matching immutable SHA tag in GHCR.
- **2:25–2:55 — CD.** Show the same SHA’s green CD run: checkout, GHCR pull, Compose update, and smoke
  gate. Briefly show `docker-compose.yml`.
- **2:55–3:35 — Deployed prediction.** Run:
  `curl http://DEPLOYED_HOST:8000/ready`, then multipart `curl -F
  file=@tests/fixtures/sample_cat.jpg http://DEPLOYED_HOST:8000/predict`. Point out label, both
  probabilities, confidence, and SHA model version.
- **3:35–4:00 — Monitoring.** Show `curl http://DEPLOYED_HOST:8000/metrics` filtered to
  `ml_api_requests_total`, `ml_predictions_total`, and `ml_api_request_latency_seconds`, plus one
  structured JSON prediction log without image data.
- **4:00–4:25 — Post-deployment performance.** Show the checked-in 20-row manifest and
  `monitoring/reports/post_deployment_metrics.json` with sample count, timestamp, model version,
  accuracy/F1, confusion matrix, and per-sample entries.
- **4:25–4:35 — Submission.** Show the final ZIP name and `unzip -l` entries for workflows, DVC,
  Docker/Compose, trained `.pt`, metrics/plots, source, and tests. Stop recording.
