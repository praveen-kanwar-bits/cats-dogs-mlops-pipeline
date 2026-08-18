# ADR-004 Docker Compose Deployment
## Context
Assignment requires GitHub Actions + self-hosted deployment.
## Decision
Use GHCR image tags + compose pull/up + smoke checks.
## Alternatives
Manual server deployment.
## Consequences
Automated immutable release rollout without Kubernetes dependency.
