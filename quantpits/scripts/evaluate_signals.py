"""Isolated worker for backtesting one group's fixed saved signals."""
import argparse
import sys

from quantpits.ensemble.saved_evaluation import evaluate_saved_group
from quantpits.research.paper_market import strict_json
from quantpits.utils.workspace import WorkspaceContext
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "mlflow-uri", "qlib-data-dir", "request", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--qlib-region", default="cn")
    args = parser.parse_args(argv)
    try:
        ctx = WorkspaceContext.from_root(args.workspace, mlflow_uri=args.mlflow_uri,
            qlib_data_dir=args.qlib_data_dir, qlib_region=args.qlib_region)
        request = strict_json(Path(args.request).read_bytes())
        result = evaluate_saved_group(ctx, request, Path(args.output))
    except (KeyboardInterrupt, SystemExit):
        print("saved-signal evaluation interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print("saved-signal evaluation failed: %s" % exc, file=sys.stderr)
        return 1
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
