#!/usr/bin/env python3
"""Read-only Champion--Challenger Stage-A ranking replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from quantpits.research.replay import (
    ReplayContractError,
    ReplayInputError,
    ResearchRankingReplay,
    load_sealed_replay_inputs,
    write_replay_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-replay")
    parser.add_argument("--workspace", help="read-only workspace; defaults to QLIB_WORKSPACE_DIR")
    parser.add_argument("--sealed-cycle", required=True, help="Phase 37A sealed Champion cycle")
    parser.add_argument("--output-dir", required=True, help="new disposable directory below /tmp")
    parser.add_argument("--qlib-data-dir", help="read-only Qlib provider; defaults to QLIB_DATA_DIR")
    parser.add_argument("--preferred-start", default="2026-07-03")
    parser.add_argument("--preferred-end", default="2026-08-21")
    parser.add_argument("--window-size", type=int, default=6, choices=[4, 5, 6])
    parser.add_argument("--top-k", type=int, default=22)
    parser.add_argument("--score-tolerance", type=float, default=1e-12)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace or os.environ.get("QLIB_WORKSPACE_DIR")
    if not workspace:
        print(json.dumps({"status": "blocked", "error": "--workspace or QLIB_WORKSPACE_DIR is required"}))
        return 2
    try:
        inputs = load_sealed_replay_inputs(
            Path(workspace), args.sealed_cycle,
            qlib_data_dir=Path(args.qlib_data_dir) if args.qlib_data_dir else None,
            coverage_start=args.preferred_start,
            coverage_end=args.preferred_end,
        )
        result = ResearchRankingReplay(
            inputs, top_k=args.top_k, score_tolerance=args.score_tolerance,
        ).run(
            preferred_start=args.preferred_start,
            preferred_end=args.preferred_end,
            window_size=args.window_size,
        )
        output = write_replay_output(result, Path(args.output_dir))
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except (ReplayContractError, ReplayInputError, OSError) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, sort_keys=True))
        return 2
    print(json.dumps({
        "status": result["status"],
        "result_digest": result["result_digest"],
        "output": output,
    }, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
