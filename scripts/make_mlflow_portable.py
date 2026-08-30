from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Make local MLflow artifact URIs project-relative")
    parser.add_argument("--database", default="mlflow.db")
    parser.add_argument("--artifact-root", default="mlruns")
    parser.add_argument("--backup", default="/tmp/cats-dogs-mlflow.db.backup")
    args = parser.parse_args()

    database = Path(args.database)
    backup = Path(args.backup)
    if not database.is_file():
        raise SystemExit(f"MLflow database not found: {database}")
    shutil.copy2(database, backup)

    with sqlite3.connect(database) as connection:
        experiments = connection.execute(
            "SELECT experiment_id FROM experiments ORDER BY experiment_id"
        ).fetchall()
        for (experiment_id,) in experiments:
            artifact_location = f"file:./{args.artifact_root}/{experiment_id}"
            connection.execute(
                "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                (artifact_location, experiment_id),
            )

        runs = connection.execute("SELECT run_uuid, experiment_id FROM runs").fetchall()
        for run_id, experiment_id in runs:
            artifact_uri = (
                f"file:./{args.artifact_root}/{experiment_id}/{run_id}/artifacts"
            )
            connection.execute(
                "UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
                (artifact_uri, run_id),
            )
        connection.commit()
        # Rebuild the database so superseded absolute paths are not retained in
        # unused SQLite pages included with the portable submission artifact.
        connection.execute("VACUUM")

    print(
        f"Updated {len(experiments)} experiments and {len(runs)} runs; backup written to {backup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
