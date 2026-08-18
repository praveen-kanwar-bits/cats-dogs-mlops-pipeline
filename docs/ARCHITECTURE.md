# Architecture

Layers:
- **domain**: core entities/types/ports.
- **application**: preprocessing, training, evaluation, inference use-cases.
- **infrastructure**: PyTorch checkpoint IO, MLflow tracking, observability.
- **interfaces**: FastAPI HTTP API and scripts/CLI entrypoints.

```mermaid
flowchart LR
  A[Kaggle Dataset] --> B[DVC preprocess]
  B --> C[Train SimpleCNN]
  C --> D[MLflow Tracking]
  C --> E[Model Artifact .pt + metadata]
  E --> F[FastAPI Inference]
  F --> G[Docker Image]
  G --> H[GHCR]
  H --> I[Docker Compose Deploy]
  I --> J[Monitoring + Metrics]
```

```mermaid
flowchart TD
  Dev[Developer Push/PR] --> GH[GitHub]
  GH --> CI[CI Workflow]
  CI --> T[Tests + Lint]
  CI --> B[Docker Build]
  CI --> P[GHCR Push SHA Tag]
  P --> CD[CD Workflow]
  CD --> DC[Docker Compose Pull/Up]
  DC --> ST[Smoke Tests]
```
