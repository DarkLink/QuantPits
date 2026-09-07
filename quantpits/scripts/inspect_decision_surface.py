"""Privacy-safe CLI for the read-only decision-surface observer."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional, Sequence

from quantpits.evidence.contracts import canonical_json_bytes


_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class _SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("invalid arguments")

    def exit(self, status: int = 0, message: Optional[str] = None) -> None:
        if status:
            raise ValueError("invalid arguments")
        raise SystemExit(0)


def _parser() -> _SafeParser:
    parser = _SafeParser(add_help=False, prog="inspect_decision_surface")
    parser.add_argument("--help", action="store_true")
    parser.add_argument("--research-workspace")
    parser.add_argument("--production-workspace")
    parser.add_argument("--engine-root")
    parser.add_argument("--current-cycle")
    parser.add_argument("--activation")
    parser.add_argument("--definition-store")
    parser.add_argument("--evidence-store")
    parser.add_argument("--bootstrap-store")
    parser.add_argument("--bootstrap-set-id")
    parser.add_argument("--reference-source", choices=("research", "production"), default="research")
    return parser


def _blocked(reason: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "observation_kind": "PRODUCTION_DECISION_SURFACE_CONTINUITY_V1",
        "status": "blocked",
        "reason_code": reason,
        "same_champion_segment": False,
        "may_continue_to_retention_gates": False,
        "intent_capability": False,
        "epoch_started": False,
        "prospective_claim": False,
        "promotion_capability": False,
        "did_write": False,
    }


def _help() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "observation_kind": "PRODUCTION_DECISION_SURFACE_CONTINUITY_V1",
        "status": "help",
        "required_arguments": [
            "research-workspace", "production-workspace", "engine-root",
            "current-cycle", "activation", "definition-store",
            "evidence-store", "bootstrap-store", "bootstrap-set-id",
        ],
        "did_write": False,
    }


def _emit(value: Dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args, unknown = _parser().parse_known_args(argv)
        if unknown:
            raise ValueError("invalid arguments")
        if args.help:
            _emit(_help())
            return 0
        fields = (
            "research_workspace", "production_workspace", "engine_root",
            "current_cycle", "activation", "definition_store",
            "evidence_store", "bootstrap_store", "bootstrap_set_id",
        )
        if any(getattr(args, field) is None for field in fields):
            raise ValueError("missing arguments")
        from quantpits.research.decision_surface import (
            DecisionSurfaceContractError,
            observe_production_decision_surface,
        )
        try:
            result = observe_production_decision_surface(
                args.research_workspace, args.production_workspace, args.engine_root,
                args.current_cycle, args.activation, args.definition_store,
                args.evidence_store, args.bootstrap_store, args.bootstrap_set_id,
                reference_source=args.reference_source,
            )
        except _PROCESS_CONTROL:
            raise
        except DecisionSurfaceContractError as exc:
            _emit(_blocked(getattr(exc, "reason_code", "INPUT_INCOMPARABLE")))
            return 2
        payload = result.to_safe_summary_dict()
        _emit(payload)
        return {
            "SAME_CHAMPION_SEGMENT": 0,
            "VERSION_BREAK": 3,
            "INCOMPARABLE": 4,
        }[payload["status"]]
    except _PROCESS_CONTROL:
        raise
    except Exception:
        _emit(_blocked("CONTRACT_INVALID"))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
