"""Preview/export explicitly chosen existing model record references."""

import argparse
import json

from quantpits.training.model_source import model_source
from quantpits.utils.workspace import WorkspaceContext


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", help="Original native static run ID; never follows current latest")
    source.add_argument("--record-file", help="Explicit existing V1/V2 record file within the workspace")
    parser.add_argument("--models", required=True, help="Comma-separated model keys, in preview order")
    parser.add_argument("--output", help="Explicit new source file within workspace; omit for zero-write preview")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = model_source(WorkspaceContext.from_root(args.workspace), models=args.models,
                          run_id=args.run_id, record_file=args.record_file, output=args.output)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
