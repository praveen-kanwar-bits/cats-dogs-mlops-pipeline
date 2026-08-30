# Final Submission Checklist

## Formal report (do not omit)

- [ ] Open `docs/submission/MLOps_Assignment_2_Final_Report.docx` and confirm the cover identifies
  the correct course, assignment, use case, and student.
- [ ] Open `docs/submission/MLOps_Assignment_2_Final_Report.pdf` and confirm all 13 pages render,
  including the contents page, training curves, confusion matrix, rubric matrix, and appendices.
- [ ] Confirm both report formats are committed and present inside the final ZIP under
  `cats-dogs-mlops-pipeline/docs/submission/`.

## Repository and artifacts

- [ ] Commit every intended file, especially `.dvc/`, `data/raw.dvc`, `dvc.lock`,
  `artifacts/release/`, the 20 evaluation fixtures, and both workflow files.
- [ ] Push the final commit to `main` and confirm CI, image publication, CD, and smoke test are green.
- [ ] Confirm the GHCR image has an immutable tag equal to the final Git commit SHA.
- [ ] Run `python scripts/create_submission.py` after all evidence files are final.
- [ ] Open the ZIP and confirm it contains source, requirements, DVC files, workflows, Dockerfile,
  Compose manifest, trained `.pt` checkpoint, metadata, metrics, plots, monitoring report, and both
  formal report formats.

## Required recording (strictly under five minutes)

- [ ] Record at 1080p with terminal text readable.
- [ ] Show the final Git SHA and matching green CI/CD runs.
- [ ] Show DVC DAG/status, MLflow parameters/metrics/artifacts, and the trained checkpoint metadata.
- [ ] Show test results and the Docker/GHCR image tag.
- [ ] Show deployed `/ready`, a real `/predict`, `/metrics`, and the 20-sample performance report.
- [ ] Verify the final video duration is below 5:00 before uploading.

## Submission integrity

- [ ] Do not include Kaggle credentials, registry tokens, `.env`, or personal runner secrets.
- [ ] Submit the final ZIP and screen recording in the LMS fields requested by the instructor.
- [ ] Keep the GitHub repository and self-hosted runner available until grading is complete.

The screen recording and successful remote GitHub runs are external deliverables; repository code
alone cannot substitute for them.
