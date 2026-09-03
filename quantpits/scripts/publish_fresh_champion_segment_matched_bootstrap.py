"""Privacy-safe CLI for one split-root fresh matched bootstrap."""

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
    parser = _SafeParser(
        prog="publish_fresh_champion_segment_matched_bootstrap",
    )
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "publish"):
        command = commands.add_parser(action)
        command.add_argument("--production-workspace", required=True)
        command.add_argument("--research-workspace", required=True)
        command.add_argument("--definition-evidence-cycle", required=True)
        command.add_argument("--bootstrap-source-cycle", required=True)
        command.add_argument("--activation", required=True)
        command.add_argument("--definition-store", required=True)
        command.add_argument("--evidence-store", required=True)
        command.add_argument("--bootstrap-store", required=True)
        if action == "publish":
            command.add_argument("--expected-bootstrap-set-id", required=True)
            command.add_argument(
                "--expected-bootstrap-request-digest", required=True,
            )
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
        _emit(_blocked("FreshMatchedBootstrapArgumentError", "ARGUMENT_INVALID"))
        return 2
    try:
        from quantpits.research.forward_bootstrap import (
            ForwardBootstrapContractError,
            prepare_fresh_champion_segment_matched_bootstrap,
            publish_fresh_champion_segment_matched_bootstrap,
        )
        common = (
            args.production_workspace, args.research_workspace,
            args.definition_evidence_cycle, args.bootstrap_source_cycle,
            args.activation, args.definition_store, args.evidence_store,
            args.bootstrap_store,
        )
        if args.action == "preflight":
            result = prepare_fresh_champion_segment_matched_bootstrap(*common)
            code = 0
        else:
            result = publish_fresh_champion_segment_matched_bootstrap(
                *common, args.expected_bootstrap_set_id,
                _digest(args.expected_bootstrap_request_digest),
                args.authorization_action,
            )
            code = {
                "COMMITTED": 0, "ADOPTED": 0,
                "CONFLICT": 3, "UNCERTAIN": 4,
            }[result.status]
        summary = result.to_safe_summary_dict()
    except _PROCESS_CONTROL:
        raise
    except _ArgumentInvalid:
        _emit(_blocked("FreshMatchedBootstrapArgumentError", "ARGUMENT_INVALID"))
        return 2
    except ForwardBootstrapContractError:
        _emit(_blocked("FreshMatchedBootstrapContractError", "CONTRACT_INVALID"))
        return 2
    except Exception:
        _emit(_blocked("FreshMatchedBootstrapRuntimeError", "RUNTIME_BLOCKED"))
        return 2
    _emit(summary)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
