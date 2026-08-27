"""Privacy-safe CLI for read-only shadow-forward definition observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Observe one sealed engineering-only shadow definition candidate.",
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--evidence-cycle", required=True)
    parser.add_argument("--activation", required=True, type=Path)
    return parser


def _canonical_line(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        from quantpits.research.forward_observation import (
            ForwardObservationContractError,
            ForwardObservationInputError,
            observe_shadow_forward_definition_candidate,
        )

        candidate = observe_shadow_forward_definition_candidate(
            args.workspace, args.evidence_cycle, args.activation,
        )
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        # The public CLI deliberately never renders a private exception value.
        error_type = "ForwardObservationBlocked"
        reason_code = "OBSERVATION_FAILED"
        try:
            if isinstance(exc, (ForwardObservationInputError, ForwardObservationContractError)):
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
