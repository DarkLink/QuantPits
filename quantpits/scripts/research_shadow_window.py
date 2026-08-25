#!/usr/bin/env python3
"""Run one zero-write Stage B3A sequential historical shadow window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from quantpits.research.historical_cycle import (
    HistoricalCycleContractError,
    HistoricalCycleInputError,
    load_replay_profile,
)
from quantpits.research.historical_window import (
    HistoricalShadowWindowReplay,
    HistoricalWindowContractError,
    HistoricalWindowInputError,
    compact_window_summary,
)
from quantpits.research.replay import (
    ReplayContractError,
    ReplayInputError,
    ResearchRankingReplay,
    load_sealed_replay_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-shadow-window")
    parser.add_argument("--workspace", required=True, help="read-only Production workspace")
    parser.add_argument("--qlib-data-dir", required=True, help="read-only Qlib provider")
    parser.add_argument("--profile", required=True, help="workspace-private strict replay profile")
    parser.add_argument("--sealed-cycle", required=True, help="Phase 37A sealed Champion cycle")
    parser.add_argument("--preferred-start", required=True, help="Stage A coverage start")
    parser.add_argument("--preferred-end", required=True, help="Stage A coverage end")
    parser.add_argument("--window-size", required=True, type=int, choices=(4, 5, 6))
    parser.add_argument("--top-k", required=True, type=int, help="explicit Stage A comparison cutoff")
    return parser


def _reason_code(exc: BaseException) -> str:
    if isinstance(exc, HistoricalWindowContractError):
        return "WINDOW_CONTRACT_ERROR"
    if isinstance(exc, HistoricalWindowInputError):
        return "WINDOW_INPUT_ERROR"
    if isinstance(exc, (HistoricalCycleContractError, ReplayContractError)):
        return "REPLAY_CONTRACT_ERROR"
    if isinstance(exc, (HistoricalCycleInputError, ReplayInputError)):
        return "REPLAY_INPUT_ERROR"
    return "READ_ERROR"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = load_replay_profile(Path(args.profile))
        inputs = load_sealed_replay_inputs(
            Path(args.workspace), args.sealed_cycle,
            qlib_data_dir=Path(args.qlib_data_dir),
            coverage_start=args.preferred_start, coverage_end=args.preferred_end,
        )
        stage_a = ResearchRankingReplay(inputs, top_k=args.top_k).run(
            preferred_start=args.preferred_start, preferred_end=args.preferred_end,
            window_size=args.window_size,
        )
        result = HistoricalShadowWindowReplay(
            stage_a_result=stage_a, stage_a_inputs=inputs, profile=profile,
            provider_root=Path(args.qlib_data_dir),
            market=stage_a["universe_identity"]["name"],
            window_size=args.window_size,
        ).run()
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except (
        HistoricalWindowContractError, HistoricalWindowInputError,
        HistoricalCycleContractError, HistoricalCycleInputError,
        ReplayContractError, ReplayInputError, OSError,
    ) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": {"type": type(exc).__name__, "reason_code": _reason_code(exc)},
        }, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(compact_window_summary(result), sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
