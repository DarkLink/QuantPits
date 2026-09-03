"""Privacy-safe CLI for split-root fresh definition evidence."""

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
    parser = _SafeParser(
        prog="publish_fresh_champion_segment_definition_evidence",
    )
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "publish"):
        command = commands.add_parser(action)
        command.add_argument("--production-workspace", required=True)
        command.add_argument("--research-workspace", required=True)
        command.add_argument("--evidence-cycle", required=True)
        command.add_argument("--activation", required=True)
        command.add_argument("--definition-store", required=True)
        command.add_argument("--evidence-store", required=True)
        if action == "publish":
            command.add_argument("--expected-definition-set-id", required=True)
            command.add_argument("--expected-definition-request-digest", required=True)
            command.add_argument("--expected-evidence-request-digest", required=True)
            command.add_argument("--authorization-action", required=True)
    return parser


def _digest(raw: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _ArgumentInvalid() from exc
    if type(value) is not dict or json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ) != raw:
        raise _ArgumentInvalid()
    return value


def _blocked(error_type: str, reason_code: str) -> Dict[str, Any]:
    return {
        "status": "blocked",
        "error": {"type": error_type, "reason_code": reason_code},
    }


def _emit(value: Dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except _ArgumentInvalid:
        _emit(_blocked("FreshDefinitionEvidenceArgumentError", "ARGUMENT_INVALID"))
        return 2
    try:
        from quantpits.research.forward_definition_evidence import (
            ForwardDefinitionEvidenceContractError,
            prepare_fresh_champion_segment_definition_evidence,
            publish_fresh_champion_segment_definition_evidence,
        )
        common = (
            args.production_workspace, args.research_workspace,
            args.evidence_cycle, args.activation, args.definition_store,
            args.evidence_store,
        )
        if args.action == "preflight":
            result = prepare_fresh_champion_segment_definition_evidence(*common)
            code = 0
        else:
            result = publish_fresh_champion_segment_definition_evidence(
                *common, args.expected_definition_set_id,
                _digest(args.expected_definition_request_digest),
                _digest(args.expected_evidence_request_digest),
                args.authorization_action,
            )
            code = {"COMMITTED": 0, "ADOPTED": 0, "CONFLICT": 3, "UNCERTAIN": 4}[
                result.status
            ]
        summary = result.to_safe_summary_dict()
    except _PROCESS_CONTROL:
        raise
    except _ArgumentInvalid:
        _emit(_blocked("FreshDefinitionEvidenceArgumentError", "ARGUMENT_INVALID"))
        return 2
    except ForwardDefinitionEvidenceContractError as exc:
        _emit(_blocked(type(exc).__name__, getattr(exc, "reason_code", "CONTRACT_INVALID")))
        return 2
    except Exception:
        _emit(_blocked("FreshDefinitionEvidenceRuntimeError", "RUNTIME_BLOCKED"))
        return 2
    _emit(summary)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
