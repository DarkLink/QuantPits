#!/usr/bin/env python3
"""Run one read-only Stage B2 historical shadow cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from quantpits.research.historical_cycle import (
    HistoricalCycleContractError,
    HistoricalCycleInputError,
    HistoricalShadowCycleReplay,
    load_replay_profile,
    write_historical_cycle_output,
)
from quantpits.research.replay import (
    ReplayContractError,
    ReplayInputError,
    ResearchRankingReplay,
    load_sealed_replay_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-shadow-cycle")
    parser.add_argument("--workspace", required=True, help="read-only Production workspace")
    parser.add_argument("--qlib-data-dir", required=True, help="read-only Qlib provider")
    parser.add_argument("--profile", required=True, help="workspace-private strict replay profile")
    parser.add_argument("--sealed-cycle", required=True, help="Phase 37A sealed Champion cycle")
    parser.add_argument("--anchor", required=True, help="one explicit selected historical anchor")
    parser.add_argument("--preferred-start", required=True, help="Stage A coverage start")
    parser.add_argument("--preferred-end", required=True, help="Stage A coverage end")
    parser.add_argument("--top-k", required=True, type=int, help="explicit Stage A comparison cutoff")
    parser.add_argument("--output-dir", required=True, help="new disposable root physically below /tmp")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = load_replay_profile(Path(args.profile))
        inputs = load_sealed_replay_inputs(
            Path(args.workspace), args.sealed_cycle, qlib_data_dir=Path(args.qlib_data_dir),
            coverage_start=args.preferred_start, coverage_end=args.preferred_end,
        )
        stage_a = ResearchRankingReplay(inputs, top_k=args.top_k).run(
            preferred_start=args.preferred_start, preferred_end=args.preferred_end,
        )
        runner = HistoricalShadowCycleReplay(
            stage_a_result=stage_a, stage_a_inputs=inputs, profile=profile,
            provider_root=Path(args.qlib_data_dir), anchor_date=args.anchor,
            market=stage_a["universe_identity"]["name"],
        )
        result = runner.run()
        if result["status"] != "COMPLETE":
            print(json.dumps({
                "status": "blocked", "result_digest": result["result_digest"],
                "arm_counts": result["arm_counts"],
            }, sort_keys=True))
            return 2
        publication = write_historical_cycle_output(runner, result, Path(args.output_dir))
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except (
        HistoricalCycleContractError, HistoricalCycleInputError,
        ReplayContractError, ReplayInputError, OSError,
    ) as exc:
        print(json.dumps({
            "status": "blocked", "error": {"type": type(exc).__name__, "message": str(exc)},
        }, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "complete" if publication.status == "COMMITTED" else "blocked",
        "result_digest": result["result_digest"], "publication": publication.to_dict(),
    }, sort_keys=True))
    return 0 if publication.status == "COMMITTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
