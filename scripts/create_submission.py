from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_NAME = "cats-dogs-mlops-pipeline"

INCLUDED_FILES = {
    ".dockerignore",
    ".dvcignore",
    ".env.example",
    ".gitignore",
    "Dockerfile",
    "Makefile",
    "MODEL_CARD.md",
    "PROJECT_STATUS.md",
    "README.md",
    "mlflow.db",
    "docker-compose.yml",
    "dvc.lock",
    "dvc.yaml",
    "params.yaml",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements-serving.txt",
    "requirements.txt",
}

INCLUDED_DIRECTORIES = (
    ".github",
    "artifacts/evaluation",
    "artifacts/model",
    "artifacts/release",
    "data",
    "docs",
    "evidence",
    "mlruns",
    "monitoring/reports",
    "scripts",
    "src",
    "tests",
)

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "cache",
    "processed",
    "raw",
    "tmp",
}

REQUIRED_DELIVERABLES = (
    Path("artifacts/release/cats_dogs_cnn.pt"),
    Path("artifacts/release/metadata.json"),
    Path("artifacts/release/SHA256SUMS"),
    Path("artifacts/evaluation/confusion_matrix.png"),
    Path("artifacts/evaluation/training_curves.png"),
    Path("artifacts/evaluation/test_metrics.json"),
    Path("data/raw.dvc"),
    Path("dvc.lock"),
    Path(".github/workflows/ci.yml"),
    Path(".github/workflows/cd.yml"),
    Path("docker-compose.yml"),
)


def should_include(path: Path) -> bool:
    if not path.is_file() or any(
        part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in path.parts
    ):
        return False
    if path.name == ".DS_Store" or path.suffix in {".pyc", ".zip"}:
        return False
    return True


def collect_files(root: Path) -> list[Path]:
    selected = [root / path for path in INCLUDED_FILES if (root / path).is_file()]
    selected.extend(
        path
        for directory in INCLUDED_DIRECTORIES
        for path in (root / directory).rglob("*")
        if (root / directory).exists() and should_include(path)
    )
    dvc_config_files = [root / ".dvc/config", root / ".dvc/.gitignore"]
    selected.extend(path for path in dvc_config_files if path.is_file())
    return sorted(set(selected))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the Assignment 2 source/artifact ZIP")
    parser.add_argument(
        "--output",
        default="dist/cats-dogs-mlops-assignment-2.zip",
        help="Destination ZIP path",
    )
    args = parser.parse_args()

    root = Path.cwd()
    missing = [str(path) for path in REQUIRED_DELIVERABLES if not (root / path).is_file()]
    if missing:
        raise SystemExit(f"Cannot package submission; missing: {', '.join(missing)}")

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    files = collect_files(root)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(root)
            archive.write(path, Path(PROJECT_NAME) / relative)

    print(f"Created {output} with {len(files)} files ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
