"""Privacy-safe CLI for one definition-bound model artifact capsule."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Mapping, Optional, Sequence


_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class _ArgumentInvalid(Exception):
    pass


class _SafeParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise _ArgumentInvalid()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeParser(prog="publish_model_artifact_capsule")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "publish", "adopt"):
        command = commands.add_parser(action)
        command.add_argument("--production-workspace", required=True)
        command.add_argument("--research-workspace", required=True)
        command.add_argument("--evidence-cycle", required=True)
        command.add_argument("--activation", required=True)
        command.add_argument("--definition-store", required=True)
        command.add_argument("--evidence-store", required=True)
        command.add_argument("--capsule-store", required=True)
        if action == "publish":
            command.add_argument("--expected-capsule-id", required=True)
            command.add_argument("--expected-request-digest", required=True)
            command.add_argument("--authorization-action", required=True)
        elif action == "adopt":
            command.add_argument("--capsule-id", required=True)
    return parser


def _digest(raw: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        raise _ArgumentInvalid() from exc
    if (
        type(value) is not dict
        or json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        != raw
    ):
        raise _ArgumentInvalid()
    return value


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ))


def _blocked(error_type: str, reason_code: str) -> Dict[str, Any]:
    return {
        "status": "blocked",
        "error": {"type": error_type, "reason_code": reason_code},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except _ArgumentInvalid:
        _emit(_blocked("ModelArtifactCapsuleArgumentError", "ARGUMENT_INVALID"))
        return 2
    contract_errors = ()
    try:
        from quantpits.research.model_artifact_capsule import (
            ModelArtifactCapsuleContractError,
            adopt_definition_bound_model_artifact_capsule,
            prepare_definition_bound_model_artifact_capsule,
            publish_definition_bound_model_artifact_capsule,
        )
        contract_errors = (ModelArtifactCapsuleContractError,)
        common = (
            args.production_workspace, args.research_workspace,
            args.evidence_cycle, args.activation, args.definition_store,
            args.evidence_store, args.capsule_store,
        )
        if args.action == "preflight":
            result = prepare_definition_bound_model_artifact_capsule(*common)
        elif args.action == "publish":
            result = publish_definition_bound_model_artifact_capsule(
                *common, args.expected_capsule_id,
                _digest(args.expected_request_digest),
                args.authorization_action,
            )
        else:
            result = adopt_definition_bound_model_artifact_capsule(
                *common, args.capsule_id,
            )
        summary = result.to_safe_summary_dict()
        code = {
            "PREPARED": 0, "COMMITTED": 0, "ADOPTED": 0,
            "CONFLICT": 3, "UNCERTAIN": 4,
        }.get(result.status, 2)
    except _PROCESS_CONTROL:
        raise
    except _ArgumentInvalid:
        _emit(_blocked("ModelArtifactCapsuleArgumentError", "ARGUMENT_INVALID"))
        return 2
    except contract_errors:
        _emit(_blocked("ModelArtifactCapsuleContractError", "CONTRACT_INVALID"))
        return 2
    except Exception:
        _emit(_blocked("ModelArtifactCapsuleRuntimeError", "RUNTIME_BLOCKED"))
        return 2
    _emit(summary)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
