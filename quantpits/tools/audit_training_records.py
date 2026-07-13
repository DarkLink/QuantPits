"""Read-only audit for latest_train_records.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantpits.training.record_audit import audit_training_records
from quantpits.training.record_audit import RecordAuditIssue, TrainingRecordAuditReport
from quantpits.training.records import snapshot_from_dict
from quantpits.runtime.mlflow_integrity import inspect_mlflow_workspace, inspect_tracking_backend_presence
from quantpits.utils.workspace import WorkspaceContext


def build_parser():
    parser = argparse.ArgumentParser(description="Audit training-record identity without mutation")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--record-file", default="latest_train_records.json")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Include the in-memory V2 migration preview")
    parser.add_argument("--verify-mlflow", action="store_true")
    parser.add_argument("--verify-predictions", action="store_true")
    return parser


def _external_issues(records, ctx, verify_predictions=False, client=None, recorder_getter=None):
    snapshot = snapshot_from_dict(records)
    presence = inspect_tracking_backend_presence(ctx.mlflow_uri, workspace_root=ctx.root)
    if not presence.metadata_ready:
        return (RecordAuditIssue("tracking_backend_missing", "error"),)
    if client is None:
        import mlflow
        from quantpits.tools.audit_mlflow_workspace import _MlflowClientAdapter
        mlflow.set_tracking_uri(ctx.mlflow_uri)
        client = _MlflowClientAdapter()
    requests = [(entry.recorder_id, entry.experiment_name) for entry in snapshot.entries]
    requests.extend(
        (entry.source_recorder_id, entry.source_experiment_name)
        for entry in snapshot.entries if entry.source_recorder_id
    )
    mlflow_report = inspect_mlflow_workspace(
        workspace_root=ctx.root, client=client,
        experiment_names=[entry.experiment_name for entry in snapshot.entries],
        recorder_requests=requests,
    )
    key_by_recorder = {entry.recorder_id: entry.key for entry in snapshot.entries}
    issues = [
        RecordAuditIssue(issue.code, issue.severity, key_by_recorder.get(issue.recorder_id or "", ""))
        for issue in mlflow_report.issues
    ]
    if verify_predictions and not mlflow_report.has_errors():
        if recorder_getter is None:
            from qlib.workflow import R
            recorder_getter = lambda rid, exp: R.get_recorder(recorder_id=rid, experiment_name=exp)
        import pandas as pd
        for entry in snapshot.entries:
            try:
                pred = recorder_getter(entry.recorder_id, entry.experiment_name).load_object("pred.pkl")
                if not isinstance(pred, (pd.Series, pd.DataFrame)) or pred.empty:
                    raise ValueError("invalid prediction")
                dates = pd.to_datetime(pred.index.get_level_values("datetime"), errors="raise")
                actual_end = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
                expected = entry.prediction_end or entry.requested_anchor
                if expected and actual_end != expected:
                    issues.append(RecordAuditIssue("prediction_anchor_mismatch", "error", entry.key))
            except Exception:
                issues.append(RecordAuditIssue("missing_or_invalid_prediction", "error", entry.key))
    return tuple(issues)


def run(args, *, client=None, recorder_getter=None):
    workspace = Path(args.workspace).resolve()
    path = (workspace / args.record_file).resolve()
    try:
        path.relative_to(workspace)
    except ValueError:
        print("Record file must remain inside workspace")
        return 1
    if not path.is_file():
        print("Training record file not found")
        return 1
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        report = audit_training_records(records)
        if args.verify_mlflow or args.verify_predictions:
            ctx = WorkspaceContext.from_root(workspace)
            external = _external_issues(
                records, ctx, verify_predictions=args.verify_predictions,
                client=client, recorder_getter=recorder_getter,
            )
            report = TrainingRecordAuditReport(
                report.schema_version, report.model_count,
                report.issues + external, report.proposed_v2,
            )
    except (OSError, ValueError, TypeError):
        print("Training record audit failed")
        return 1
    public = report.to_public_dict(include_preview=args.preview)
    if args.json:
        print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("Schema: %s" % report.schema_version)
        print("Models: %s" % report.model_count)
        print("Status: %s" % ("OK" if report.ok and not report.issues else "REVIEW"))
        for issue in report.issues:
            suffix = " (%s)" % issue.model_key if issue.model_key else ""
            print("- %s%s" % (issue.code, suffix))
    return 0 if report.ok and not report.issues else 2


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
