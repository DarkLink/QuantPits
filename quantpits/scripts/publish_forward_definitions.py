"""Privacy-safe CLI for one formally frozen definition publication."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence


_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class _ArgumentInvalid(Exception):
    pass


class _SafeParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _ArgumentInvalid()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeParser(prog="publish_forward_definitions")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "publish"):
        command = subparsers.add_parser(action)
        command.add_argument("--workspace", required=True)
        command.add_argument("--evidence-cycle", required=True)
        command.add_argument("--activation", required=True)
        command.add_argument("--definition-store", required=True)
        if action == "publish":
            command.add_argument("--expected-definition-set-id", required=True)
            command.add_argument("--expected-request-digest", required=True)
            command.add_argument("--authorization-action", required=True)
    return parser


def _blocked(error_type: str, reason_code: str) -> Dict[str, Any]:
    return {
        "status": "blocked",
        "error": {"type": error_type, "reason_code": reason_code},
    }


def _emit(value: Dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _digest(raw: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _ArgumentInvalid() from exc
    if type(value) is not dict:
        raise _ArgumentInvalid()
    if json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ) != raw:
        raise _ArgumentInvalid()
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except _ArgumentInvalid:
        _emit(_blocked("FrozenDefinitionPublicationArgumentError", "ARGUMENT_INVALID"))
        return 2
    try:
        from quantpits.research.forward_definition_publication import (
            FrozenDefinitionPublicationContractError,
            prepare_frozen_shadow_forward_definition_publication,
            publish_frozen_shadow_forward_definition_bundle,
        )
        if args.action == "preflight":
            result = prepare_frozen_shadow_forward_definition_publication(
                args.workspace, args.evidence_cycle, args.activation,
                args.definition_store,
            )
            summary = result.to_safe_summary_dict()
            code = 0
        else:
            result = publish_frozen_shadow_forward_definition_bundle(
                args.workspace, args.evidence_cycle, args.activation,
                args.definition_store, args.expected_definition_set_id,
                _digest(args.expected_request_digest), args.authorization_action,
            )
            summary = result.to_safe_summary_dict()
            code = {"COMMITTED": 0, "ADOPTED": 0, "CONFLICT": 3, "UNCERTAIN": 4}[
                summary["status"]
            ]
    except _PROCESS_CONTROL:
        raise
    except _ArgumentInvalid:
        _emit(_blocked("FrozenDefinitionPublicationArgumentError", "ARGUMENT_INVALID"))
        return 2
    except FrozenDefinitionPublicationContractError as exc:
        _emit(_blocked(type(exc).__name__, getattr(exc, "reason_code", "CONTRACT_INVALID")))
        return 2
    except Exception:
        _emit(_blocked("FrozenDefinitionPublicationRuntimeError", "RUNTIME_BLOCKED"))
        return 2
    _emit(summary)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
