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
    for name in ("plan", "run", "status", "resume"):
        command = actions.add_parser(name)
        command.add_argument("--project", required=True)
        command.add_argument("--cycle", required=True)
        if name in ("plan", "run"):
            command.add_argument("--anchor", required=True)
            command.add_argument("--value-through", required=True)
            command.add_argument("--trade-date", required=True)
            command.add_argument("--since", required=True)
            command.add_argument("--inputs", help="cycle JSON: actual evidence/empty-day assumption, paper_cycles, hold, skip")
    return result


def main(argv=None, *, runner=None, fault=None):
    args = parser().parse_args(argv)
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
        else:
            result = execute(project, cycle=args.cycle, runner=runner, fault=fault)
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
