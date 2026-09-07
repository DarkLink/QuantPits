"""Privacy-safe CLI for one read-only first forward intent preparation."""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from quantpits.evidence.contracts import canonical_json_bytes

_ARGUMENTS = (
    "production-workspace", "research-workspace", "engine-root", "qlib-provider",
    "current-cycle", "activation", "definition-store", "evidence-store",
    "bootstrap-store", "bootstrap-set-id", "signal-capsule-store", "signal-capsule-id",
    "model-capsule-store", "model-capsule-id",
)


class _SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("invalid arguments")


def _emit(payload):
    sys.stdout.write(canonical_json_bytes(payload).decode("utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _SafeParser(prog="prepare_forward_intent", add_help=False)
    parser.add_argument("--help", action="store_true")
    for name in _ARGUMENTS:
        parser.add_argument("--" + name)
    try:
        args = parser.parse_args(argv)
        if args.help:
            _emit({"schema_version": 1, "status": "help", "required_arguments": list(_ARGUMENTS), "did_write": False})
            return 0
        values = tuple(getattr(args, name.replace("-", "_")) for name in _ARGUMENTS)
        if any(value is None for value in values):
            raise ValueError("missing arguments")
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        from quantpits.research.forward_intent_preparation import _summary
        payload = _summary()
        payload["reason_codes"] = ["CLI_ARGUMENT_INVALID"]
        _emit(payload)
        return 2
    from quantpits.research.forward_intent_preparation import prepare_first_forward_intent
    result = prepare_first_forward_intent(*values)
    _emit(result.to_safe_summary_dict())
    return {"PREPARED": 0, "VERSION_BREAK": 3, "PRECONDITION_BLOCKED": 2}[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
