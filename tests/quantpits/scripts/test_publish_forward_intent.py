import json
import os
import subprocess
import sys

import pytest

from quantpits.scripts import publish_forward_intent as cli
from quantpits.research import forward_intent_publication as m


def test_help_without_workspace():
    env = dict(os.environ)
    env.pop('QLIB_WORKSPACE_DIR', None)
    env.pop('MLFLOW_TRACKING_URI', None)
    code = "from quantpits.scripts.publish_forward_intent import main; import sys; assert main(['--help']) == 0; assert 'quantpits.utils.env' not in sys.modules; assert 'qlib' not in sys.modules; assert 'mlflow' not in sys.modules"
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'help'


@pytest.mark.parametrize('status', list(cli._EXITS))
def test_status_and_safe_output(monkeypatch, capsys, status):
    result = m._result(m.FirstIntentBundleObservation, m._summary(status))
    monkeypatch.setattr(m, 'inspect_first_forward_intent_pair', lambda *a, **k: result)
    code = cli.main(['inspect', '--intent-store-root', '/private/root', '--epoch-id', 'private-epoch', '--expected-request-digest', 'a' * 64])
    assert code == cli._EXITS[status]
    output = capsys.readouterr()
    assert json.loads(output.out)['status'] == status
    assert 'private' not in output.out and not output.err


@pytest.mark.parametrize('argv', [[], ['publish'], ['--now', 'private'], ['bad-private-action'],
    ['inspect', '--intent-store-root', '/private', '--epoch-id', 'private', '--expected-request-digest', 'a' * 64, '--current-cycle', '2026-09-04']])
def test_invalid_arguments_safe(argv, capsys):
    assert cli.main(argv) == 2
    output = capsys.readouterr()
    assert 'private' not in output.out and output.err == ''
    assert not json.loads(output.out)['did_write']


def test_interrupt_propagates(monkeypatch):
    def interrupt(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(m, 'inspect_first_forward_intent_pair', interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli.main(['inspect', '--intent-store-root', '/private', '--epoch-id', 'private', '--expected-request-digest', 'a' * 64])
