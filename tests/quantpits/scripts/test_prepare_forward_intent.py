import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quantpits.evidence.contracts import canonical_json_bytes
from quantpits.research import forward_intent_preparation as m
from quantpits.scripts import prepare_forward_intent as cli


def _args():
    return [part for name in cli._ARGUMENTS for part in ("--" + name, "/private/SECRET")]


@pytest.mark.parametrize("status,code", [("PREPARED", 0), ("VERSION_BREAK", 3), ("PRECONDITION_BLOCKED", 2)])
def test_cli_status_and_privacy(monkeypatch, capsys, status, code):
    summary = m._summary()
    summary.update(status=status, intent_pair_prepared=status == "PREPARED")
    monkeypatch.setattr(m, "prepare_first_forward_intent", lambda *args: m._result(summary))
    assert cli.main(_args()) == code
    captured = capsys.readouterr()
    assert captured.out.encode() == canonical_json_bytes(summary)
    assert captured.err == "" and "SECRET" not in captured.out


def test_invalid_cli_has_no_private_diagnostic(capsys):
    assert cli.main(["--unknown", "SECRET"]) == 2
    captured = capsys.readouterr()
    assert captured.err == "" and "SECRET" not in captured.out
    assert json.loads(captured.out)["reason_codes"] == ["CLI_ARGUMENT_INVALID"]


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_propagates(monkeypatch, exception, capsys):
    def stop(*args):
        raise exception()
    monkeypatch.setattr(m, "prepare_first_forward_intent", stop)
    with pytest.raises(exception):
        cli.main(_args())
    assert capsys.readouterr().out == ""


def test_import_help_no_workspace_and_no_writes(tmp_path):
    root = Path(__file__).resolve().parents[3]
    environment = {key: value for key, value in os.environ.items() if key not in ("QLIB_WORKSPACE_DIR", "MLFLOW_TRACKING_URI")}
    environment["PYTHONPATH"] = str(root)
    script = '''
import os, sys
from pathlib import Path
before = (os.getcwd(), dict(os.environ), list(sys.argv), tuple(Path('.').iterdir()))
from quantpits.scripts import prepare_forward_intent as cli
import quantpits.research.forward_intent_preparation
assert cli.main(['--help']) == 0
assert 'quantpits.utils.env' not in sys.modules
assert 'qlib' not in sys.modules and 'mlflow' not in sys.modules
assert before == (os.getcwd(), dict(os.environ), list(sys.argv), tuple(Path('.').iterdir()))
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=str(tmp_path), env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    assert json.loads(result.stdout)["status"] == "help"
    assert result.stderr == b""
    result = subprocess.run([sys.executable, "-B", "-m", "quantpits.scripts.prepare_forward_intent", "--help"],
                            cwd=str(tmp_path), env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    assert json.loads(result.stdout)["status"] == "help" and result.stderr == b""
