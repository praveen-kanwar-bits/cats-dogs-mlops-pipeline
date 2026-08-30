# GitHub CI/CD Setup

The repository workflows use GitHub Actions, GitHub Container Registry (GHCR), and a
self-hosted Docker Compose deployment runner. Complete these one-time settings before the final
demonstration.

## Repository settings

1. In **Settings → Actions → General → Workflow permissions**, select **Read and write
   permissions**. The CI `GITHUB_TOKEN` needs `packages: write` to publish to GHCR.
2. Create a GitHub Actions environment named **production**. Optional approval protection is fine,
   but approve it before recording the demo.
3. Install a self-hosted runner on the Docker Compose target host and attach it to this repository.
   Docker Engine with Compose v2 and Python 3.11 must be available to the runner account.
4. Keep the runner online. It must be able to bind host port 8000 and read GHCR packages using the
   workflow `GITHUB_TOKEN`.
5. After the first publish, set the GHCR package visibility to public or grant this repository
   explicit package access.

No personal access token, Kaggle secret, or production SSH key is required by the checked-in
workflows. GitHub supplies `GITHUB_TOKEN` for GHCR login.

## Expected flow

1. Every push and pull request runs lint, compilation, 16 tests, release-checkpoint integrity
   verification, a Docker build, a container start, and health/prediction smoke tests.
2. A push to `main` additionally publishes immutable `${commit-sha}` and mutable `latest` tags to
   `ghcr.io/<owner>/<repository>`.
3. Only after that CI run succeeds, CD checks out the exact successful SHA, pulls that immutable
   image, updates the Compose service, and runs the post-deploy smoke test.
4. Any smoke-test failure fails CD and prints Compose logs. Deployment runs are serialized to avoid
   racing releases.

## Evidence to capture

- Green CI `test-build` and `publish` jobs for the submitted SHA.
- GHCR package showing the submitted SHA tag.
- Green CD `deploy` job for the same SHA.
- The deployment host returning `/ready`, `/predict`, and `/metrics` responses.
