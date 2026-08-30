# Project Status

## Implemented and locally evidenced

Full 25k-image DVC pipeline, evaluated SimpleCNN checkpoint, MLflow runs/artifacts, FastAPI,
Prometheus metrics, structured logging, checksum-gated Docker image, Compose deployment manifest,
GitHub CI/CD, 16 tests, 20-image post-deploy evaluation, documentation, and ZIP builder.
The formal 13-page assignment report is available in both DOCX and PDF under `docs/submission/`,
and the ZIP builder now refuses to package a submission if either format is missing.

## Current measured results

- Test accuracy: 0.7548; F1: 0.7517
- Data split: 19,998 / 2,500 / 2,500
- Release checkpoint: 377 KiB with committed SHA-256 checksum
- Local test suite: 16 passed

## External completion required before submission

- Commit/push the final working tree.
- Configure GitHub settings and the self-hosted production runner per `docs/GITHUB_SETUP.md`.
- Capture green CI, GHCR publish, and CD evidence for the same final SHA.
- Record and submit the real under-five-minute video.

See `docs/SUBMISSION_CHECKLIST.md`. Do not claim those external items are complete until visible in
GitHub and included in the LMS submission.
