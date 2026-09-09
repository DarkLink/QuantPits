"""Privacy-safe prepare / publish / offline inspect for first-pair settlement."""
from __future__ import annotations

from quantpits.scripts.prepare_forward_intent import _SafeParser, _emit

_INPUTS = ("intent-store-root", "expected-intent-request-digest", "publication-success-record-path", "qlib-provider-root")
_BASE = ("settlement-store-root", "epoch-id")
_EXITS = {"READY": 0, "COMMITTED": 0, "ADOPTED": 0, "VERIFIED": 0, "WAITING_FOR_DATA": 2,
          "PRECONDITION_BLOCKED": 2, "REQUEST_MISMATCH": 2, "CONFLICT": 4, "INCOMPLETE": 5, "UNCERTAIN": 5}


def main(argv=None):
    parser = _SafeParser(prog="settle_forward_intent", add_help=False)
    parser.add_argument("action", nargs="?", choices=("prepare", "publish", "inspect"))
    parser.add_argument("--help", action="store_true")
    for name in _BASE + _INPUTS + ("expected-request-digest",):
        parser.add_argument("--" + name)
    try:
        args = parser.parse_args(argv)
        if args.help:
            _emit(dict(schema_version=1, status="help", actions=["prepare", "publish", "inspect"],
                preparation_arguments=list(_BASE + _INPUTS), inspect_arguments=list(_BASE + ("expected-request-digest",)),
                publish_extra_arguments=["expected-request-digest"], did_write=False))
            return 0
        if args.action is None:
            raise ValueError("action required")
        required = _BASE + (("expected-request-digest",) if args.action == "inspect" else _INPUTS)
        if args.action == "publish":
            required += ("expected-request-digest",)
        if any(getattr(args, n.replace("-", "_")) is None for n in required):
            raise ValueError("arguments required")
        if args.action == "inspect" and any(getattr(args, n.replace("-", "_")) is not None for n in _INPUTS):
            raise ValueError("unexpected inspect arguments")
        if args.action == "prepare" and args.expected_request_digest is not None:
            raise ValueError("unexpected digest")
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        _emit(dict(schema_version=1, status="PRECONDITION_BLOCKED", reason_codes=["CLI_ARGUMENT_INVALID"],
                   did_write=False, state_chain_ready=False, prospective_claim=False, epoch_started=False))
        return 2
    from quantpits.research import forward_settlement as settlement
    kwargs = dict(settlement_store_root=args.settlement_store_root)
    if args.action == "inspect":
        result = settlement.inspect_first_forward_settlement(args.settlement_store_root, args.epoch_id,
            expected_request_digest=args.expected_request_digest)
    else:
        function = settlement.prepare_first_forward_settlement
        if args.action == "publish":
            function = settlement.publish_first_forward_settlement
            kwargs["expected_request_digest"] = args.expected_request_digest
        result = function(args.intent_store_root, args.epoch_id, args.expected_intent_request_digest,
            args.publication_success_record_path, args.qlib_provider_root, **kwargs)
    _emit(result.to_safe_summary_dict())
    return _EXITS[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
