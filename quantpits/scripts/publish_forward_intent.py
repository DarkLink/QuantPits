"""Safe CLI for prepare / publish / offline inspect of a first intent pair."""
from __future__ import annotations

from quantpits.scripts.prepare_forward_intent import _SafeParser, _emit, _ARGUMENTS

_EXTRA = ("intent-store-root", "epoch-id", "decision-deadline-utc", "next-open-utc", "market-timezone", "opening-policy")
_EXITS = {"READY": 0, "COMMITTED": 0, "ADOPTED": 0, "VERIFIED": 0,
          "PRECONDITION_BLOCKED": 2, "REQUEST_MISMATCH": 2, "VERSION_BREAK": 3,
          "CONFLICT": 4, "UNCERTAIN": 5, "INCOMPLETE": 5}


def main(argv=None):
    parser = _SafeParser(prog="publish_forward_intent", add_help=False)
    parser.add_argument("action", nargs="?", choices=("prepare", "publish", "inspect"))
    parser.add_argument("--help", action="store_true")
    for name in _ARGUMENTS + _EXTRA + ("expected-request-digest",):
        parser.add_argument("--" + name)
    try:
        args = parser.parse_args(argv)
        if args.help:
            _emit(dict(schema_version=1, status="help", actions=["prepare", "publish", "inspect"],
                       preparation_arguments=list(_ARGUMENTS + _EXTRA),
                       inspect_arguments=["intent-store-root", "epoch-id", "expected-request-digest"],
                       publish_extra_arguments=["expected-request-digest"], did_write=False))
            return 0
        if args.action is None:
            raise ValueError("action required")
        required = ("intent-store-root", "epoch-id", "expected-request-digest") if args.action == "inspect" else _ARGUMENTS + _EXTRA
        if args.action == "publish":
            required += ("expected-request-digest",)
        if any(getattr(args, name.replace("-", "_")) is None for name in required):
            raise ValueError("arguments required")
        if args.action == "inspect" and any(getattr(args, name.replace("-", "_")) is not None for name in _ARGUMENTS + _EXTRA[2:]):
            raise ValueError("unexpected inspect arguments")
        if args.action == "prepare" and args.expected_request_digest is not None:
            raise ValueError("unexpected digest")
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        _emit(dict(schema_version=1, status="PRECONDITION_BLOCKED", reason_codes=["CLI_ARGUMENT_INVALID"],
                   did_write=False, prospective_claim=False, epoch_started=False))
        return 2
    from quantpits.research import forward_intent_publication as publication
    if args.action == "inspect":
        result = publication.inspect_first_forward_intent_pair(args.intent_store_root, args.epoch_id,
            expected_request_digest=args.expected_request_digest)
    else:
        values = [getattr(args, name.replace("-", "_")) for name in _ARGUMENTS]
        kwargs = {name.replace("-", "_"): getattr(args, name.replace("-", "_")) for name in _EXTRA}
        function = publication.prepare_first_forward_intent_publication
        if args.action == "publish":
            function = publication.publish_first_forward_intent_pair
            kwargs["expected_request_digest"] = args.expected_request_digest
        result = function(*values, **kwargs)
    _emit(result.to_safe_summary_dict())
    return _EXITS[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
