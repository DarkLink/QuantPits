#!/usr/bin/env python3
"""
Migration helper: move a QuantPits workspace from the deprecated MLflow
file-store backend (mlruns/) to the modern SQLite backend (mlflow.db).

Requires mlflow >= 3.0.  On mlflow < 3.0 the file-store is still fully
supported and migration is unnecessary.

Usage
-----
    python -m quantpits.tools.migrate_mlflow_backend \\
        --workspace workspaces/MyWorkspace

What it does
------------
1. Validates that mlruns/ exists and contains experiment data.
2. Checks that mlflow >= 3.0 is installed (migrate-filestore CLI is 3.x-only).
3. Runs ``mlflow migrate-filestore`` to import every run into mlflow.db.
4. Renames mlruns/ → mlruns_backup_<timestamp>/ so the data is preserved.
5. Prints the new MLFLOW_TRACKING_URI for the user's run_env.sh.

Note: The migration is ONE-WAY.  There is no reverse migration from SQLite
back to the file-store.  The backup directory can be deleted once you have
verified that all runs are visible in the SQLite database.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time


# ── helpers ──────────────────────────────────────────────────────────────────

def _mlflow_version() -> tuple:
    """Return the installed mlflow version as a (major, minor, patch) tuple."""
    try:
        import mlflow
        parts = mlflow.__version__.split(".")
        return tuple(int(x) for x in parts[:3])
    except Exception:
        return (0, 0, 0)


def _has_experiment_data(mlruns_dir: str) -> bool:
    """Return True if mlruns/ contains at least one experiment directory."""
    if not os.path.isdir(mlruns_dir):
        return False
    for entry in os.listdir(mlruns_dir):
        if entry != ".gitkeep" and os.path.isdir(os.path.join(mlruns_dir, entry)):
            return True
    return False


# ── main logic ───────────────────────────────────────────────────────────────

def migrate(workspace: str, yes: bool = False) -> int:
    """Run the migration.  Returns 0 on success, non-zero on failure."""
    workspace = os.path.abspath(workspace)
    mlruns_dir = os.path.join(workspace, "mlruns")
    db_path = os.path.join(workspace, "mlflow.db")
    db_uri = f"sqlite:///{db_path}"

    print(f"Workspace : {workspace}")
    print(f"Source    : {mlruns_dir}")
    print(f"Target    : {db_path}")
    print()

    # ── pre-flight checks ─────────────────────────────────────────────────────

    if not os.path.isdir(workspace):
        print(f"Error: Workspace directory does not exist: {workspace}", file=sys.stderr)
        return 1

    if not _has_experiment_data(mlruns_dir):
        print(
            "No experiment data found in mlruns/.\n"
            "Nothing to migrate — the workspace already uses the default SQLite backend."
        )
        return 0

    mv = _mlflow_version()
    if mv < (3, 0, 0):
        print(
            f"mlflow {'.'.join(str(x) for x in mv)} is installed.\n"
            "The migrate-filestore CLI requires mlflow >= 3.0.\n"
            "\n"
            "Workaround for older mlflow:\n"
            "  1. Keep using the file:// backend — it still works on mlflow < 3.0.\n"
            "  2. Upgrade to mlflow >= 3.0 and re-run this tool.",
            file=sys.stderr,
        )
        return 1

    if os.path.exists(db_path):
        print(
            f"Warning: {db_path} already exists.\n"
            "Running migrate-filestore will APPEND runs into the existing database."
        )

    if not yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 0

    # ── run mlflow migrate-filestore ──────────────────────────────────────────

    cmd = [
        sys.executable, "-m", "mlflow", "migrate-filestore",
        "--backend-store-uri", db_uri,
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    print(f"(source mlruns/ is implicitly read from MLFLOW_TRACKING_URI)\n")

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = f"file://{mlruns_dir}"
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("\nMigration failed — mlflow exited with a non-zero status.", file=sys.stderr)
        return result.returncode

    # ── rename mlruns/ to a timestamped backup ────────────────────────────────

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(workspace, f"mlruns_backup_{timestamp}")
    shutil.move(mlruns_dir, backup_dir)
    print(f"\nRenamed mlruns/ → {os.path.basename(backup_dir)}")
    print("(You may delete the backup once you have verified the migration.)")

    # ── success summary ───────────────────────────────────────────────────────

    print(
        f"\n✅  Migration complete!\n"
        f"\n"
        f"   New tracking URI : {db_uri}\n"
        f"\n"
        f"   QuantPits will pick this up automatically on next run.\n"
        f"   If you have MLFLOW_TRACKING_URI set in your run_env.sh, update it:\n"
        f"\n"
        f"     export MLFLOW_TRACKING_URI=\"{db_uri}\"\n"
    )
    return 0


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a QuantPits workspace from the MLflow file-store backend "
            "(mlruns/) to the SQLite backend (mlflow.db). Requires mlflow >= 3.0."
        )
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Path to the workspace directory (e.g. workspaces/MyWorkspace).",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()
    sys.exit(migrate(args.workspace, yes=args.yes))


if __name__ == "__main__":
    main()
