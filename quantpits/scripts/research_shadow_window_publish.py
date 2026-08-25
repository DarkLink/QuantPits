#!/usr/bin/env python3
"""Compute once and create one private retrospective B3B bundle."""

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
)
from quantpits.research.historical_window_publication import (
    HistoricalWindowPublicationContractError,
    HistoricalWindowPublicationInputError,
    write_historical_window_output,
)
from quantpits.research.replay import (
    ReplayContractError,
    ReplayInputError,
    ResearchRankingReplay,
    load_sealed_replay_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-shadow-window-publish")
    parser.add_argument("--workspace", required=True, help="read-only Production workspace")
    parser.add_argument("--qlib-data-dir", required=True, help="read-only Qlib provider")
    parser.add_argument("--profile", required=True, help="workspace-private strict replay profile")
    parser.add_argument("--sealed-cycle", required=True, help="Phase 37A sealed Champion cycle")
    parser.add_argument("--preferred-start", required=True, help="Stage A coverage start")
    parser.add_argument("--preferred-end", required=True, help="Stage A coverage end")
    parser.add_argument("--window-size", required=True, type=int, choices=(4, 5, 6))
    parser.add_argument("--top-k", required=True, type=int, help="explicit Stage A comparison cutoff")
    parser.add_argument("--output-dir", required=True, help="new physical /tmp direct child")
    return parser


def _reason_code(exc: BaseException) -> str:
    if isinstance(exc, HistoricalWindowPublicationContractError):
        return "PUBLICATION_CONTRACT_ERROR"
    if isinstance(exc, HistoricalWindowPublicationInputError):
        return "PUBLICATION_INPUT_ERROR"
    if isinstance(exc, HistoricalWindowContractError):
        return "WINDOW_CONTRACT_ERROR"
    if isinstance(exc, HistoricalWindowInputError):
        return "WINDOW_INPUT_ERROR"
    if isinstance(exc, (HistoricalCycleContractError, ReplayContractError)):
        return "REPLAY_CONTRACT_ERROR"
    if isinstance(exc, (HistoricalCycleInputError, ReplayInputError)):
        return "REPLAY_INPUT_ERROR"
    return "READ_ERROR"


def _receipt_payload(receipt) -> dict:
    return {
        "status": receipt.status,
        "reason_code": receipt.reason_code,
        "did_write": receipt.did_write,
        "operation_id": receipt.operation_id,
        "result_digest": dict(receipt.result_digest),
        "manifest_digest": (
            None if receipt.manifest_digest is None else dict(receipt.manifest_digest)
        ),
        "member_count": receipt.member_count,
        "prospective_claim": False,
        "promotion_capability": False,
    }


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
        runner = HistoricalShadowWindowReplay(
            stage_a_result=stage_a, stage_a_inputs=inputs, profile=profile,
            provider_root=Path(args.qlib_data_dir),
            market=stage_a["universe_identity"]["name"], window_size=args.window_size,
        )
        result = runner.run()
        receipt = write_historical_window_output(runner, result, Path(args.output_dir))
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except (
        HistoricalWindowPublicationContractError,
        HistoricalWindowPublicationInputError,
        HistoricalWindowContractError,
        HistoricalWindowInputError,
        HistoricalCycleContractError,
        HistoricalCycleInputError,
        ReplayContractError,
        ReplayInputError,
        OSError,
    ) as exc:
        print(json.dumps({
            "status": "blocked",
            "error": {"type": type(exc).__name__, "reason_code": _reason_code(exc)},
        }, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(_receipt_payload(receipt), sort_keys=True, separators=(",", ":")))
    return {"COMMITTED": 0, "CONFLICT": 3, "UNCERTAIN": 4}[receipt.status]


if __name__ == "__main__":
    raise SystemExit(main())
