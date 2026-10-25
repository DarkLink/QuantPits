"""Plan, run, inspect and resume one explicitly configured project cycle."""
from __future__ import annotations

import argparse
import json
import signal
import sys

from quantpits.operations.config import cycle_request, load_project, read_json
from quantpits.operations.cycle import execute, preview, status


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    actions = result.add_subparsers(dest="action", required=True)
    for name in ("plan", "run", "status", "resume", "evaluate"):
        command = actions.add_parser(name)
        command.add_argument("--project", required=True)
        command.add_argument("--cycle", required=True)
        if name in ("run", "resume"):
            command.add_argument("--evaluate", action="store_true", help="compare fixed saved signals after core completion")
        if name in ("run", "resume", "evaluate"):
            command.add_argument("--start-date", help="diagnostic backtest start")
            command.add_argument("--end-date", help="diagnostic backtest end")
            command.add_argument("--account", type=float, help="diagnostic initial cash (default 100000; no account-state change)")
        if name in ("plan", "run"):
            command.add_argument("--anchor", required=True)
            command.add_argument("--value-through", required=True)
            command.add_argument("--trade-date", required=True)
            command.add_argument("--since", required=True)
            command.add_argument("--inputs", help="cycle JSON: actual evidence/empty-day assumption, paper_cycles, hold, skip")
    return result


def main(argv=None, *, runner=None, fault=None, evaluator=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if args.action in ("run", "resume") and not args.evaluate and any(
        getattr(args, key) is not None for key in ("start_date", "end_date", "account")
    ):
        command_parser.error("diagnostic dates/account require --evaluate")
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("cycle received termination")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        project = load_project(args.project)
        if args.action in ("plan", "run"):
            request = cycle_request(project, cycle=args.cycle, anchor=args.anchor, value_through=args.value_through,
                trade_date=args.trade_date, since=args.since, inputs=read_json(args.inputs) if args.inputs else None)
            result = preview(project, request) if args.action == "plan" else execute(project, request=request, runner=runner, fault=fault)
        elif args.action == "status":
            result = status(project, args.cycle)
        elif args.action == "resume":
            result = execute(project, cycle=args.cycle, runner=runner, fault=fault)
        if args.action == "evaluate" or (args.action in ("run", "resume") and args.evaluate):
            from quantpits.operations.evaluation import evaluate
            try:
                evaluation = ((evaluator or evaluate)(project, args.cycle, start=args.start_date,
                    end=args.end_date, account=100000 if args.account is None else args.account)
                    if args.action == "evaluate" or result.get("status") == "complete" else
                    {"status": "not_started", "reason": "core cycle is not complete"})
            except Exception as exc:
                if args.action == "evaluate":
                    raise
                evaluation = {"status": "failed", "warning": str(exc), "error_type": type(exc).__name__}
            if args.action == "evaluate":
                result = evaluation
            else:
                result["evaluation"] = evaluation
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "cycle": args.cycle}), file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("status") == "complete" or args.action == "plan" else 2


if __name__ == "__main__":
    raise SystemExit(main())
