"""Privacy-safe CLI for split-root fresh Champion segment observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


class _PrivacySafeParseError(ValueError):
    pass


class _PrivacySafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _PrivacySafeParseError("command line is invalid")


def _parser() -> argparse.ArgumentParser:
    parser = _PrivacySafeArgumentParser(
        description="Observe one split-root fresh Champion segment candidate.",
    )
    parser.add_argument("--production-workspace", required=True, type=Path)
    parser.add_argument("--research-workspace", required=True, type=Path)
    parser.add_argument("--evidence-cycle", required=True)
    parser.add_argument("--activation", required=True, type=Path)
    return parser


def _canonical_line(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
        from quantpits.research.forward_observation import (
            ForwardObservationContractError,
            ForwardObservationInputError,
            observe_fresh_champion_segment_candidate,
        )
        candidate = observe_fresh_champion_segment_candidate(
            args.production_workspace, args.research_workspace,
            args.evidence_cycle, args.activation,
        )
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        error_type = "FreshChampionSegmentBlocked"
        reason_code = "OBSERVATION_FAILED"
        try:
            if isinstance(exc, _PrivacySafeParseError):
                error_type = "FreshChampionSegmentArgumentError"
                reason_code = "ARGUMENT_INVALID"
            elif isinstance(exc, (ForwardObservationInputError, ForwardObservationContractError)):
                error_type = type(exc).__name__
                reason_code = exc.reason_code
        except NameError:
            pass
        print(_canonical_line({
            "status": "blocked",
            "error": {"type": error_type, "reason_code": reason_code},
        }))
        return 2
    print(_canonical_line(candidate.to_safe_summary_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
