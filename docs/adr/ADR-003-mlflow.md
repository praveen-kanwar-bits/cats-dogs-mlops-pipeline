# ADR-003 MLflow
## Context
Need experiment and artifact tracking.
## Decision
Use explicit MLflow logging API in train/evaluate scripts.
## Alternatives
No tracker, opaque autologging only.
## Consequences
Clear param/metric/artifact traceability.
