import json
import os
import subprocess
import sys

import pytest

from quantpits.scripts import settle_forward_intent as cli
from tests.quantpits.research.test_forward_settlement import (
    fresh_evidence_workspace, fresh_bootstrap_workspace, prepared_inputs, publication, settlement,
)


def arguments(kw, action, digest=None):
    names = ('settlement_store_root', 'epoch_id') if action == 'inspect' else tuple(kw)
    result = [action]
    for name in names:
        result.extend(['--' + name.replace('_', '-'), str(kw[name])])
    if digest is not None:
        result.extend(['--expected-request-digest', digest])
    return result


def test_help_without_workspace():
    environment = dict(os.environ)
    environment.pop('QLIB_WORKSPACE_DIR', None)
    environment.pop('MLFLOW_TRACKING_URI', None)
    result = subprocess.run([sys.executable, '-m', 'quantpits.scripts.settle_forward_intent', '--help'],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)['status'] == 'help'


@pytest.mark.parametrize('args', [[], ['publish'], ['--now', 'private'], ['--force-prospective'],
    ['inspect', '--settlement-store-root', '/private', '--epoch-id', 'private', '--expected-request-digest', 'f' * 64,
     '--qlib-provider-root', '/private/provider']])
def test_invalid_arguments_safe(args, capsys):
    assert cli.main(args) == 2
    output = capsys.readouterr().out
    assert json.loads(output)['status'] == 'PRECONDITION_BLOCKED'
    assert 'private' not in output


def test_cli_roundtrip(settlement, capsys):
    assert cli.main(arguments(settlement, 'prepare')) == 0
    prepared = json.loads(capsys.readouterr().out)
    digest = prepared['request_digest']
    assert cli.main(arguments(settlement, 'publish', digest)) == 0
    committed = json.loads(capsys.readouterr().out)
    assert committed['status'] == 'COMMITTED' and committed['did_write']
    assert cli.main(arguments(settlement, 'inspect', digest)) == 0
    output = capsys.readouterr().out
    assert json.loads(output)['state_chain_ready']
    for private in (str(settlement['settlement_store_root']), settlement['epoch_id'], 'SH600', 'portfolio.'):
        assert private not in output
    assert cli.main(arguments(settlement, 'publish', digest)) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'ADOPTED'
    assert cli.main(arguments(settlement, 'inspect', 'f' * 64)) == 4
    assert json.loads(capsys.readouterr().out)['status'] == 'CONFLICT'


@pytest.mark.parametrize('status,code', [('WAITING_FOR_DATA', 2), ('REQUEST_MISMATCH', 2), ('UNCERTAIN', 5), ('INCOMPLETE', 5)])
def test_exit_mapping(settlement, monkeypatch, capsys, status, code):
    from quantpits.research import forward_settlement as m
    monkeypatch.setattr(m, 'prepare_first_forward_settlement', lambda *a, **kw: m._result(m._summary(status)))
    assert cli.main(arguments(settlement, 'prepare')) == code
    assert json.loads(capsys.readouterr().out)['status'] == status
