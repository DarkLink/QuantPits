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
