"""Privacy-safe explicit continuing intent / settlement commands."""
from __future__ import annotations

from quantpits.scripts.prepare_forward_intent import _SafeParser, _emit

_CURRENT = ('production-workspace-root', 'research-workspace-root', 'engine-root', 'qlib-provider-root',
    'current-cycle-id', 'activation-path', 'definition-store-root', 'evidence-store-root', 'bootstrap-store-root',
    'bootstrap-set-id', 'signal-capsule-store-root', 'signal-capsule-id', 'model-capsule-store-root', 'model-capsule-id')
_JOIN = ('intent-store-root', 'settlement-store-root', 'epoch-id', 'cycle-index', 'first-intent-store-root',
    'expected-first-intent-request-digest', 'predecessor-settlement-store-root', 'predecessor-cycle-index',
    'expected-predecessor-request-digest', 'decision-deadline-utc', 'next-open-utc', 'market-timezone')
_SETTLE = ('intent-store-root', 'settlement-store-root', 'epoch-id', 'cycle-index',
    'expected-intent-request-digest', 'publication-success-record-path', 'qlib-provider-root')
_ACTIONS = ('prepare-intent', 'publish-intent', 'inspect-intent', 'prepare-settlement', 'publish-settlement', 'inspect-settlement')
_EXITS = {'READY': 0, 'COMMITTED': 0, 'ADOPTED': 0, 'VERIFIED': 0, 'WAITING_FOR_DATA': 2,
    'PRECONDITION_BLOCKED': 2, 'REQUEST_MISMATCH': 2, 'VERSION_BREAK': 3, 'CONFLICT': 4, 'INCOMPLETE': 5, 'UNCERTAIN': 5}


def main(argv=None):
    parser = _SafeParser(prog='continue_forward', add_help=False)
    parser.add_argument('action', nargs='?', choices=_ACTIONS)
    parser.add_argument('--help', action='store_true')
    names = tuple(dict.fromkeys(_CURRENT + _JOIN + _SETTLE + ('expected-request-digest',)))
    for name in names:
        parser.add_argument('--' + name)
    try:
        args = parser.parse_args(argv)
        if args.help:
            _emit(dict(schema_version=1, status='help', actions=list(_ACTIONS),
                intent_arguments=list(_CURRENT + _JOIN), settlement_arguments=list(_SETTLE),
                inspect_arguments=['intent-store-root OR settlement-store-root', 'epoch-id', 'cycle-index', 'expected-request-digest'],
                publish_extra_arguments=['expected-request-digest'], did_write=False))
            return 0
        if args.action is None:
            raise ValueError('action required')
        action, kind = args.action.split('-')
        required = (kind + '-store-root', 'epoch-id', 'cycle-index') if action == 'inspect' else (
            _CURRENT + _JOIN if kind == 'intent' else _SETTLE)
        if action != 'prepare':
            required += ('expected-request-digest',)
        values = {name: getattr(args, name.replace('-', '_')) for name in names}
        if any(values[name] is None for name in required) or any(value is not None and name not in required for name, value in values.items()):
            raise ValueError('arguments invalid')
        kwargs = {name.replace('-', '_'): values[name] for name in required}
        for name in ('cycle_index', 'predecessor_cycle_index'):
            if name in kwargs:
                raw = kwargs[name]
                parsed = int(raw)
                if str(parsed) != raw or parsed < (1 if name == 'predecessor_cycle_index' else 2):
                    raise ValueError('index invalid')
                kwargs[name] = parsed
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        _emit(dict(schema_version=1, status='PRECONDITION_BLOCKED', reason_codes=['CLI_ARGUMENT_INVALID'],
                   did_write=False, prospective_claim=False, epoch_started=False))
        return 2
    from quantpits.research import forward_continuation as continuation
    functions = {
        'prepare-intent': continuation.prepare_next_forward_intent_publication,
        'publish-intent': continuation.publish_next_forward_intent_pair,
        'inspect-intent': continuation.inspect_next_forward_intent_pair,
        'prepare-settlement': continuation.prepare_next_forward_settlement,
        'publish-settlement': continuation.publish_next_forward_settlement,
        'inspect-settlement': continuation.inspect_next_forward_settlement,
    }
    result = functions[args.action](**kwargs)
    _emit(result.to_safe_summary_dict())
    return _EXITS[result.status]


if __name__ == '__main__':
    raise SystemExit(main())
