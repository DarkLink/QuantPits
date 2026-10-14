"""Workspace-scoped weekly paper account commands."""
from __future__ import annotations

import argparse
import json
import sys

from quantpits.research.paper_accounting import PaperError
from quantpits.research.paper_market import strict_json
from quantpits.research.paper_workflow import PaperWorkflow
from quantpits.utils.workspace import WorkspaceContext


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workspace", required=True)
    result.add_argument("--qlib-data-dir")
    result.add_argument("--mlflow-uri", help="account backend; signal backend is separate")
    result.add_argument("--config", help="defaults to config/paper_config.json")
    actions = result.add_subparsers(dest="action", required=True)
    bootstrap = actions.add_parser("bootstrap", help="copy an explicit source account once")
    bootstrap.add_argument("--source-config", required=True)
    bootstrap.add_argument("--date", required=True)
    plan = actions.add_parser("plan", help="freeze signal, predecessor and ordered candidates")
    plan.add_argument("--cycle", required=True)
    plan.add_argument("--anchor", required=True)
    plan.add_argument("--trade-date", required=True)
    selection = plan.add_mutually_exclusive_group()
    selection.add_argument("--prediction", help="explicit CSV/pickle source")
    selection.add_argument("--recorder-id")
    plan.add_argument("--experiment-id")
    plan.add_argument("--signal-workspace")
    plan.add_argument("--signal-mlflow-uri")
    plan.add_argument("--lineage-file", action="append", help="freeze a source snapshot/manifest; repeatable")
    plan.add_argument("--rules", help="dated market rule JSON; optional template in paper config")
    settle = actions.add_parser("settle", help="sell then buy at the selected session cash open")
    settle.add_argument("--cycle", required=True)
    settle.add_argument("--rules", help="explicit dated rule/reference supplement")
    cancel = actions.add_parser("cancel", help="retire a pending plan explicitly, retaining its history")
    cancel.add_argument("--cycle", required=True)
    cancel.add_argument("--reason", required=True)
    value = actions.add_parser("value", help="append close valuation without making trades")
    value.add_argument("--date", required=True)
    event = actions.add_parser("event", help="register one real, explicitly sourced event")
    event.add_argument("--file", required=True)
    report = actions.add_parser("report", help="read paper valuation and optional actual reference")
    report.add_argument("--date", required=True)
    report.add_argument("--reference-config")
    report.add_argument("--since", help="saved earlier valuation; defaults to the previous observed date")
    report.add_argument("--reference-start-config", help="actual account snapshot at --since")
    report.add_argument("--reference-events", help="explicit actual event list; [] confirms no known cashflows")
    report.add_argument("--save", action="store_true", help="write JSON/Markdown to workspace output/paper")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        ctx = WorkspaceContext.from_root(args.workspace, mlflow_uri=args.mlflow_uri,
                                        qlib_data_dir=args.qlib_data_dir)
        workflow = PaperWorkflow(ctx, config_path=args.config)
        if args.action == "bootstrap":
            result = workflow.bootstrap(args.source_config, args.date)
        elif args.action == "plan":
            if args.recorder_id is not None and args.experiment_id is None:
                raise PaperError("--recorder-id requires --experiment-id")
            if args.experiment_id is not None and args.recorder_id is None:
                raise PaperError("--experiment-id requires --recorder-id")
            override = {}
            for key in ("prediction", "recorder_id", "experiment_id"):
                if getattr(args, key) is not None:
                    override[key] = getattr(args, key)
            if args.signal_workspace:
                override["workspace"] = args.signal_workspace
            if args.signal_mlflow_uri:
                override["mlflow_uri"] = args.signal_mlflow_uri
            if args.lineage_file is not None:
                override["lineage_files"] = args.lineage_file
            rules = strict_json(workflow._input_path(args.rules).read_bytes()) if args.rules else None
            result = workflow.plan(args.cycle, args.anchor, args.trade_date, source_override=override, rules=rules)
        elif args.action == "settle":
            rules = strict_json(workflow._input_path(args.rules).read_bytes()) if args.rules else None
            result = workflow.settle(args.cycle, rules=rules)
        elif args.action == "value":
            result = workflow.value(args.date)
        elif args.action == "cancel":
            result = workflow.cancel(args.cycle, args.reason)
        elif args.action == "event":
            result = workflow.add_event(strict_json(workflow._input_path(args.file).read_bytes()))
        else:
            actual_events = strict_json(workflow._input_path(args.reference_events).read_bytes()) if args.reference_events else None
            result = workflow.report(args.date, since=args.since, reference_config=args.reference_config,
                                     reference_start_config=args.reference_start_config, reference_events=actual_events)
            if args.save:
                result["report_paths"] = workflow.save_report(result)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if result["status"] == "WAITING_FOR_DATA" else 0


if __name__ == "__main__":
    raise SystemExit(main())
