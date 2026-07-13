"""Read-only MLflow workspace lineage audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantpits.runtime.mlflow_integrity import inspect_mlflow_workspace
from quantpits.utils.workspace import WorkspaceContext


class _MlflowClientAdapter:
    def __init__(self):
        import mlflow
        from mlflow.tracking import MlflowClient

        self._mlflow = mlflow
        self._client = MlflowClient()

    def tracking_uri(self):
        return self._mlflow.get_tracking_uri()

    def experiments_by_name(self, name):
        return [exp for exp in self._client.search_experiments() if exp.name == name]

    def recorder(self, recorder_id, experiment_name=None):
        run = self._client.get_run(recorder_id)
        exp = self._client.get_experiment(run.info.experiment_id)
        return {
            "id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "experiment_name": exp.name,
            "artifact_uri": run.info.artifact_uri,
        }

    def all_recorder_requests(self):
        requests = []
        for experiment in self._client.search_experiments():
            for run in self._client.search_runs([experiment.experiment_id]):
                requests.append((run.info.run_id, experiment.name))
        return requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit MLflow resources against an active workspace")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--write-experiment", action="append", default=[])
    parser.add_argument("--recorder-id", action="append", default=[])
    parser.add_argument("--recorder-experiment", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--all-runs", action="store_true", help="inspect every run (potentially expensive)")
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    ctx = WorkspaceContext.from_root(workspace)
    import mlflow

    mlflow.set_tracking_uri(ctx.mlflow_uri)
    expected = list(args.recorder_experiment)
    requests = [
        (recorder_id, expected[index] if index < len(expected) else None)
        for index, recorder_id in enumerate(args.recorder_id)
    ]
    client = _MlflowClientAdapter()
    if args.all_runs:
        requests.extend(client.all_recorder_requests())
    report = inspect_mlflow_workspace(
        workspace_root=ctx.root,
        client=client,
        experiment_names=args.experiment,
        recorder_requests=requests,
        write_experiments=args.write_experiment,
    )
    payload = report.to_public_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(f"Workspace: {payload['workspace']}")
        print(f"Tracking: {payload['tracking']['path']} ({'contained' if payload['tracking']['contained'] else 'external'})")
        for item in payload["experiments"]:
            print(f"Experiment {item['name']}: active={len(item['active_ids'])}, mode={item['access_mode']}")
        for issue in payload["issues"]:
            print(f"{issue['severity'].upper()} {issue['code']}: {issue['message']}")
    if report.has_errors():
        return 2
    if args.fail_on_warning and any(issue.severity == "warning" for issue in report.issues):
        return 2
    return 0


def main(argv=None) -> None:
    parser = build_parser()
    try:
        code = run(parser.parse_args(argv))
    except Exception as exc:
        print(f"Audit failed: {type(exc).__name__}: {exc}")
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
