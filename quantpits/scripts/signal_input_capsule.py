"""Privacy-safe manual CLI for weekly critical signal retention."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional, Sequence

from quantpits.evidence.contracts import canonical_json_bytes


_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class _SafeParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise ValueError("invalid arguments")

    def exit(self, status: int = 0, message: Optional[str] = None) -> None:
        if status:
            raise ValueError("invalid arguments")
        raise SystemExit(0)


def _parser() -> _SafeParser:
    parser = _SafeParser(add_help=False, prog="signal_input_capsule")
    parser.add_argument("action", nargs="?")
    parser.add_argument("--help", action="store_true")
    parser.add_argument("--production-workspace")
    parser.add_argument("--research-workspace")
    parser.add_argument("--cycle")
    parser.add_argument("--capsule-store")
    parser.add_argument("--capsule-id")
    parser.add_argument("--request-digest")
    parser.add_argument("--authorization-action")
    return parser


def _blocked(reason: str, status: str = "blocked") -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "capsule_kind": "WEEKLY_CRITICAL_SIGNAL_INPUT_CAPSULE_V1",
        "status": status, "reason_code": reason,
        "critical_signal_retention_complete": False,
        "same_champion_segment": False, "definition_bound": False,
        "intent_capability": False, "epoch_started": False,
        "prospective_claim": False, "promotion_capability": False,
        "did_write": False,
    }


def _emit(value: Dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args, unknown = _parser().parse_known_args(argv)
        if unknown:
            raise ValueError
        if args.help:
            _emit({
                "schema_version": 1,
                "capsule_kind": "WEEKLY_CRITICAL_SIGNAL_INPUT_CAPSULE_V1",
                "status": "help", "actions": ["prepare", "publish", "adopt"],
                "did_write": False,
            })
            return 0
        common = (args.production_workspace, args.research_workspace, args.cycle, args.capsule_store)
        if args.action not in {"prepare", "publish", "adopt"} or any(value is None for value in common):
            raise ValueError
        from quantpits.research.signal_input_capsule import (
            SignalInputCapsuleContractError, adopt_signal_input_capsule,
            prepare_signal_input_capsule, publish_signal_input_capsule,
        )
        try:
            if args.action == "prepare":
                if any(value is not None for value in (args.capsule_id, args.request_digest, args.authorization_action)):
                    raise ValueError
                result = prepare_signal_input_capsule(*common)
            elif args.action == "publish":
                if None in (args.capsule_id, args.request_digest, args.authorization_action):
                    raise ValueError
                try:
                    request_digest = json.loads(args.request_digest)
                except Exception as exc:
                    raise ValueError from exc
                result = publish_signal_input_capsule(
                    *common, args.capsule_id, request_digest, args.authorization_action,
                )
            else:
                if args.capsule_id is None or args.request_digest is not None or args.authorization_action is not None:
                    raise ValueError
                result = adopt_signal_input_capsule(*common, args.capsule_id)
        except _PROCESS_CONTROL:
            raise
        except SignalInputCapsuleContractError as exc:
            reason = getattr(exc, "reason_code", "CONTRACT_INVALID")
            status = "PRECONDITION_BLOCKED" if reason != "CONTRACT_INVALID" else "blocked"
            _emit(_blocked(reason, status))
            return 3 if status == "PRECONDITION_BLOCKED" else 2
        payload = result.to_safe_dict()
        _emit(payload)
        return {
            "PREPARED": 0, "COMMITTED": 0, "ADOPTED": 0,
            "PRECONDITION_BLOCKED": 3, "CONFLICT": 4, "UNCERTAIN": 5,
        }[payload["status"]]
    except _PROCESS_CONTROL:
        raise
    except Exception:
        _emit(_blocked("CONTRACT_INVALID"))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
