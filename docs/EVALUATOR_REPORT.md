# Expert Evaluator Report

Assignment: MLOps (S1-25_AIMLCZG523), Assignment 2
Use case: binary Cats vs Dogs image classification
Evaluation scale: 50/50 marks, equivalent to 100%

## M1 — Model development and experiment tracking: 10/10

- Git history versions the project; DVC metadata versions the 25,005-file raw dataset and the
  generated processed dataset.
- The deterministic, stratified manifest contains 19,998 training, 2,500 validation, and 2,500
  test images (80%/10%/10% of 24,998 valid images).
- Every processed image is physically saved as 224×224 RGB. Training alone applies random flip,
  rotation, and color jitter augmentation.
- The baseline is a three-block PyTorch CNN serialized as a `.pt` checkpoint.
- MLflow training and evaluation runs contain hyperparameters, epoch metrics, final metrics, the
  model, metadata, history, confusion matrix, and training curves.
- Held-out test evidence: accuracy 0.7548, precision 0.7613, recall 0.7424, F1 0.7517.

## M2 — Packaging and containerization: 10/10

- FastAPI implements `/health`, model-aware `/ready`, multipart `/predict`, and `/metrics`.
- Prediction returns the class label, confidence, both class probabilities, and deployed model
  version.
- ML, serving, test, DVC, and Kaggle packages are pinned. Training uses the portable PyTorch 2.2.2 /
  torchvision 0.17.2 pair, while the Linux inference image explicitly selects the corresponding
  CPU-only wheels instead of downloading an unused CUDA stack.
- The Docker image runs as a non-root user, verifies SHA-256 checksums for the promoted evaluated
  checkpoint, exposes port 8000, and uses model-aware readiness as its health check.

## M3 — CI build, test, and image publication: 10/10

- Sixteen tests cover preprocessing, exact stratification, augmentation, model output, optimizer
  selection, inference probabilities/errors, API behavior, Prometheus metrics, and loading the real
  release model.
- Every push and pull request checks out code, installs pinned dependencies, lints, compiles, tests,
  verifies the release checkpoint, builds the image, runs it, and executes smoke tests.
- Trusted `main` pushes publish both immutable commit-SHA and `latest` tags to GHCR only after CI
  succeeds. The previous random bootstrap-model fallback has been removed.

## M4 — Continuous deployment: 10/10

- Docker Compose is the declared target and includes restart, port, environment, health, and
  bounded-log configuration.
- Successful `main` CI completion triggers a serialized production deployment on the self-hosted
  runner. CD checks out the exact successful SHA, pulls that SHA-tagged image, and updates Compose.
- Post-deploy smoke testing calls both health/readiness and prediction, validates the label,
  confidence, probability keys/sum, and model version, and fails the workflow on any error.

## M5 — Monitoring, post-deployment tracking, and submission: 10/10

- Structured JSON logs record request ID, method, path, status, latency, prediction class,
  confidence, and model version without filenames or image bytes.
- Prometheus exposes request count/status, error count, prediction count/class, and latency
  histograms.
- The post-deployment evaluator sends 20 distinct labeled 224×224 images, records per-request
  predictions and confidence, and calculates accuracy, precision, recall, F1, and confusion matrix.
- The deterministic ZIP builder includes all source code, DVC/CI/CD/container/deployment
  configuration, trained checkpoint plus checksum/metadata, MLflow evidence, plots, tests, and the
  monitoring report while excluding raw data, caches, credentials, and virtual environments.
- A timed recording script and strict final checklist cover the required sub-five-minute video.

## Evaluator conclusion

The repository implementation and local artifacts satisfy all technical rubric objectives. Award
50/50 once the final submitted SHA has visible green GitHub CI, GHCR publish, and CD runs and the
student submits the required under-five-minute recording. Those external execution/recording items
cannot be proven solely by source files.
