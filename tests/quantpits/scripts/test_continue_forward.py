import json
import os
import subprocess
import sys

import pytest

from quantpits.scripts import continue_forward as cli
from tests.quantpits.research.test_forward_continuation import (
    m, DATES, _calendar, settle, fresh_evidence_workspace, fresh_bootstrap_workspace,
    prepared_inputs, publication, continuation,
)


def arguments(action, kwargs):
    result = [action]
    for key, value in kwargs.items():
        result += ['--' + key.replace('_', '-'), str(value)]
    return result


def test_help_import_without_workspace():
    environment = dict(os.environ)
    for name in ('QLIB_WORKSPACE_DIR', 'MLFLOW_TRACKING_URI'):
        environment.pop(name, None)
    for code in ([sys.executable, '-m', 'quantpits.scripts.continue_forward', '--help'],
                 [sys.executable, '-c', 'import quantpits.research.forward_continuation']):
        result = subprocess.run(code, env=environment, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert not result.stderr
        if '--help' in code:
            assert len(json.loads(result.stdout)['actions']) == 6


@pytest.mark.parametrize('args', [[], ['publish-intent'], ['--now', '/private'],
    ['inspect-intent', '--intent-store-root', '/private', '--epoch-id', 'private', '--cycle-index', '02', '--expected-request-digest', 'f' * 64],
    ['inspect-settlement', '--settlement-store-root', '/private', '--epoch-id', 'private', '--cycle-index', '2', '--expected-request-digest', 'f' * 64, '--qlib-provider-root', '/private'],
    ['prepare-intent', '--expected-request-digest', 'f' * 64]])
def test_cli_invalid_is_safe(args, capsys):
    assert cli.main(args) == 2
    output = capsys.readouterr().out
    assert json.loads(output)['reason_codes'] == ['CLI_ARGUMENT_INVALID']
    assert 'private' not in output


def test_all_six_cli_actions(continuation, capsys, monkeypatch):
    from datetime import datetime, timezone
    args, kw = continuation[:2]
    inputs = dict(zip(m._CURRENT_ARGUMENTS, args), **kw)
    assert cli.main(arguments('prepare-intent', inputs)) == 0
    digest = json.loads(capsys.readouterr().out)['request_digest']
    assert cli.main(arguments('publish-intent', dict(inputs, expected_request_digest=digest))) == 0
    success = json.loads(capsys.readouterr().out)
    assert success['cycle_intent_published'] and not success['epoch_started']
    inspect = dict(intent_store_root=kw['intent_store_root'], epoch_id=kw['epoch_id'], cycle_index=2, expected_request_digest=digest)
    assert cli.main(arguments('inspect-intent', inspect)) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'VERIFIED'
    path = continuation[-1] / 'cli_success.json'
    path.write_bytes(m.c4.canonical(success))
    path.chmod(0o600)
    _calendar(args[3] / 'calendars/day.txt', DATES[:5])
    monkeypatch.setattr(m.d1, '_clock', lambda: datetime(2026, 9, 14, 2, tzinfo=timezone.utc))
    params = dict(intent_store_root=kw['intent_store_root'], settlement_store_root=kw['settlement_store_root'],
        epoch_id=kw['epoch_id'], cycle_index=2, expected_intent_request_digest=digest,
        publication_success_record_path=path, qlib_provider_root=args[3])
    assert cli.main(arguments('prepare-settlement', params)) == 0
    digest = json.loads(capsys.readouterr().out)['request_digest']
    assert cli.main(arguments('publish-settlement', dict(params, expected_request_digest=digest))) == 0
    assert json.loads(capsys.readouterr().out)['state_chain_ready']
    inspect = dict(settlement_store_root=kw['settlement_store_root'], epoch_id=kw['epoch_id'], cycle_index=2, expected_request_digest=digest)
    assert cli.main(arguments('inspect-settlement', inspect)) == 0
    output = capsys.readouterr().out
    assert json.loads(output)['status'] == 'VERIFIED'
    for private in (kw['epoch_id'], str(args[0]), 'SZ000002', 'portfolio.'):
        assert private not in output


@pytest.mark.parametrize('status,code', [('WAITING_FOR_DATA', 2), ('VERSION_BREAK', 3), ('CONFLICT', 4), ('UNCERTAIN', 5), ('INCOMPLETE', 5)])
def test_exit_codes(monkeypatch, capsys, status, code):
    monkeypatch.setattr(m, 'inspect_next_forward_intent_pair', lambda **kwargs:
        m.c4._result(m.c4.FirstIntentBundleObservation, m._intent_summary(2, status=status)))
    kwargs = dict(intent_store_root='/private', epoch_id='private', cycle_index=2, expected_request_digest='f' * 64)
    assert cli.main(arguments('inspect-intent', kwargs)) == code
    assert json.loads(capsys.readouterr().out)['status'] == status


def test_paper_first_and_continuing_cli_without_deadline_or_stdout(continuation, capsys, monkeypatch):
    from datetime import datetime, timezone
    from quantpits.scripts import publish_forward_intent as first_cli
    from quantpits.scripts import settle_forward_intent as settlement_cli
    from quantpits.scripts import report_forward_window as report_cli
    args, old_kw, _, _, tmp = continuation
    roots = []
    epoch = 'paper.cli'
    for name in ('paper_first_intents', 'paper_first_settlements', 'paper_continuing_intents', 'paper_continuing_settlements'):
        root = tmp / name
        root.mkdir(mode=0o700)
        roots.append(root)
    for root in roots[2:]:
        (root / epoch).mkdir(mode=0o700)
    first_args = list(args)
    first_args[4], first_args[11] = '2026-09-04', 'signal.2026-09-04'
    monkeypatch.setattr(m.c4, '_clock', lambda: (datetime(2026, 9, 9, 12, tzinfo=timezone.utc), 100.))
    first_inputs = dict(zip((n.replace('-', '_') for n in first_cli._ARGUMENTS), first_args),
        intent_store_root=roots[0], epoch_id=epoch, next_open_utc='2026-09-07T01:30:00Z',
        market_timezone='Asia/Shanghai', opening_policy=m.c4.OPENING_POLICY)
    assert first_cli.main(arguments('prepare', first_inputs)) == 0
    first_digest = json.loads(capsys.readouterr().out)['request_digest']
    assert first_cli.main(arguments('publish', dict(first_inputs, expected_request_digest=first_digest))) == 0
    first = json.loads(capsys.readouterr().out)
    assert not first['prospective_claim'] and first['accounting_time']['generation_relation'] == 'AFTER_OPEN_SIMULATION'
    _calendar(args[3] / 'calendars/day.txt', DATES[:5])
    monkeypatch.setattr(m.d1, '_clock', lambda: datetime(2026, 9, 15, 12, tzinfo=timezone.utc))
    params = dict(intent_store_root=roots[0], epoch_id=epoch, expected_intent_request_digest=first_digest,
        qlib_provider_root=args[3], settlement_store_root=roots[1])
    assert settlement_cli.main(arguments('prepare', params)) == 0
    first_settlement = json.loads(capsys.readouterr().out)['request_digest']
    assert settlement_cli.main(arguments('publish', dict(params, expected_request_digest=first_settlement))) == 0
    assert json.loads(capsys.readouterr().out)['state_chain_ready']
    kw = dict(old_kw, intent_store_root=roots[2], settlement_store_root=roots[3], epoch_id=epoch,
        first_intent_store_root=roots[0], expected_first_intent_request_digest=first_digest,
        predecessor_settlement_store_root=roots[1], expected_predecessor_request_digest=first_settlement)
    kw.pop('publication_mode')
    kw.pop('decision_deadline_utc')
    monkeypatch.setattr(m.c4, '_clock', lambda: (datetime(2026, 9, 16, 12, tzinfo=timezone.utc), 100.))
    inputs = dict(zip(m._CURRENT_ARGUMENTS, args), **kw)
    assert cli.main(arguments('prepare-intent', inputs)) == 0
    next_digest = json.loads(capsys.readouterr().out)['request_digest']
    assert cli.main(arguments('publish-intent', dict(inputs, expected_request_digest=next_digest))) == 0
    second = json.loads(capsys.readouterr().out)
    assert second['accounting_time']['generation_relation'] == 'AFTER_OPEN_SIMULATION'
    params = dict(intent_store_root=roots[2], settlement_store_root=roots[3], epoch_id=epoch, cycle_index=2,
        expected_intent_request_digest=next_digest, qlib_provider_root=args[3])
    monkeypatch.setattr(m.d1, '_clock', lambda: datetime(2026, 9, 17, 12, tzinfo=timezone.utc))
    assert cli.main(arguments('prepare-settlement', params)) == 0
    next_settlement = json.loads(capsys.readouterr().out)['request_digest']
    assert cli.main(arguments('publish-settlement', dict(params, expected_request_digest=next_settlement))) == 0
    assert json.loads(capsys.readouterr().out)['state_chain_ready']
    request = dict(schema_version=1, first_intent_store_root=str(roots[0]), first_settlement_store_root=str(roots[1]),
        continuing_intent_store_root=str(roots[2]), continuing_settlement_store_root=str(roots[3]), epoch_id=epoch,
        requested_cycles=[dict(cycle_index=1, current_cycle_id='2026-09-04', expected_intent_request_digest=first_digest,
                               expected_settlement_request_digest=first_settlement),
                          dict(cycle_index=2, current_cycle_id='2026-09-11', expected_intent_request_digest=next_digest,
                               expected_settlement_request_digest=next_settlement)])
    request_path, output = tmp / 'paper_window.json', tmp / 'paper_report'
    request_path.write_text(json.dumps(request))
    assert report_cli.main(['--request-file', str(request_path), '--output-dir', str(output)]) == 0
    assert json.loads(capsys.readouterr().out)['settled_count'] == 2
    report = json.loads((output / 'report.json').read_text())
    assert all(r['accounting_time']['generation_relation'] == 'AFTER_OPEN_SIMULATION' for r in report['rows'])
    assert all(r['original_success'] == 'LOCAL_BUNDLE_VERIFIED_WITHOUT_STDOUT' for r in report['rows'])
    assert [r['trade_date'] for r in report['rows']] == ['2026-09-07', '2026-09-14']


def test_explicit_hold_cli_does_not_infer_missing_week(continuation, capsys, monkeypatch):
    from tests.quantpits.research.test_forward_continuation import _hold_fixture
    fixture = _hold_fixture(continuation, monkeypatch)
    args, kw = fixture[:2]
    inputs = dict(zip(m._CURRENT_ARGUMENTS, args), **kw)
    without_hold = {k: v for k, v in inputs.items() if not k.startswith('hold_')}
    assert cli.main(arguments('prepare-intent', without_hold)) == 2
    assert json.loads(capsys.readouterr().out)['reason_codes'] == ['GAP_DETECTED']
    assert cli.main(arguments('prepare-intent', inputs)) == 0
    prepared = json.loads(capsys.readouterr().out)
    context = prepared['continuation_context']
    assert context['predecessor_trade_date'] == '2026-09-07'
    assert context['hold']['end_date'] == context['selected_anchor'] == '2026-09-18'
    assert context['trade_date'] == '2026-09-21'
    assert cli.main(arguments('publish-intent', dict(inputs, expected_request_digest=prepared['request_digest']))) == 0
    safe = json.loads(capsys.readouterr().out)
    assert safe['continuation_context'] == context and safe['cycle_intent_published']
    assert not safe['prospective_claim']
