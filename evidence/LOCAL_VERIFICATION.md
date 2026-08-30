# Local Verification Evidence

Verified on 2026-08-24 using Python 3.11.9 and a Linux/amd64 Podman 5.8.5 runtime.

| Check | Result |
|---|---|
| Pinned dependency consistency | `pip check`: no broken requirements |
| Static quality | Ruff passed; all source/scripts/tests compiled |
| Automated tests | 16 passed in 7.02 seconds |
| DVC | DAG loaded; `Data and pipelines are up to date.` |
| MLflow | SQLite experiment opened; two completed runs, eight params, 44 metrics, six artifacts |
| Held-out evaluation | accuracy 0.7548; precision 0.7613; recall 0.7424; F1 0.7517 |
| Release integrity | model and metadata SHA-256 checks passed during image build |
| Container build | final source build succeeded, image ID prefix `be75e806acd8` |
| Container smoke | readiness/model load plus prediction contract passed |
| Simulated traffic | 20/20 requests successful |
| Post-deploy batch | 20 distinct labeled images; accuracy/F1 0.70 |
| Prometheus evidence | 41 successful prediction requests observed; request and latency metrics exposed |
| Logging evidence | JSON request/prediction summaries include IDs, status, latency, confidence, version; no image bytes or filenames |

The generated report is `monitoring/reports/post_deployment_metrics.json`. The container was stopped
after verification. Remote GitHub Actions, GHCR, self-hosted CD, and the required screen recording
remain student-executed external evidence and are not represented as locally complete.
