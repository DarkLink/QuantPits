import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import quantpits.research.decision_surface as surface
from quantpits.evidence.contracts import canonical_json_bytes
from quantpits.scripts import inspect_decision_surface as cli


def _result(kind):
    rows = []
    for index, name in enumerate(surface.COMPONENT_NAMES):
        reference = {"position": index}
        current = dict(reference)
        if kind == "VERSION_BREAK" and index == 1:
            current["changed"] = True
        if kind == "INCOMPARABLE" and index == 2:
            current = None
        rows.append(surface._component(name, reference, current, "SYNTHETIC_BLOCK"))
    return surface._make_result(rows, "2026-08-14", "2026-08-21")


def _args():
    return [
        "--research-workspace", "/private/research",
        "--production-workspace", "/private/production",
        "--engine-root", "/private/engine",
        "--current-cycle", "2026-08-21",
        "--activation", "/private/research/activation.json",
        "--definition-store", "/private/research/definitions",
        "--evidence-store", "/private/research/evidence",
        "--bootstrap-store", "/private/research/bootstraps",
        "--bootstrap-set-id", "bootstrap.private",
    ]


@pytest.mark.parametrize("status,code", [
    ("SAME_CHAMPION_SEGMENT", 0), ("VERSION_BREAK", 3), ("INCOMPARABLE", 4),
])
def test_cli_status_exit_code_privacy_and_canonical_json_are_consistent(
    monkeypatch, capsys, status, code,
):
    monkeypatch.setattr(surface, "observe_production_decision_surface", lambda *_, **kw: _result(status))
    assert cli.main(_args()) == code
    output = capsys.readouterr().out.encode()
    payload = json.loads(output)
    assert payload["status"] == status
    assert output == canonical_json_bytes(payload)
    assert b"/private" not in output
    assert b"bootstrap.private" not in output


def test_cli_unknown_argument_does_not_echo_private_token(monkeypatch, capsys):
    token = "PRIVATE_TOKEN_DO_NOT_ECHO"
    assert cli.main(_args() + ["--unknown", token]) == 2
    captured = capsys.readouterr()
    assert token not in captured.out
    assert token not in captured.err
    assert json.loads(captured.out)["reason_code"] == "CONTRACT_INVALID"


def test_cli_contract_error_is_privacy_safe(monkeypatch, capsys):
    def fail(*_args, **kwargs):
        raise surface.DecisionSurfaceInputError("/private/model/RECORDER_SECRET")
    monkeypatch.setattr(surface, "observe_production_decision_surface", fail)
    assert cli.main(_args()) == 2
    output = capsys.readouterr().out
    assert "/private" not in output and "RECORDER_SECRET" not in output
    assert json.loads(output)["reason_code"] == "INPUT_INCOMPARABLE"


def test_cli_help_import_preserve_env_cwd_and_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repository),
        "QLIB_WORKSPACE_DIR": str(workspace),
    })
    script = (
        "import os; from pathlib import Path; "
        "before=(Path.cwd(),dict(os.environ),tuple(Path(os.environ['QLIB_WORKSPACE_DIR']).iterdir())); "
        "from quantpits.scripts import inspect_decision_surface as m; "
        "assert m.main(['--help']) == 0; "
        "after=(Path.cwd(),dict(os.environ),tuple(Path(os.environ['QLIB_WORKSPACE_DIR']).iterdir())); "
        "assert before == after"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=str(tmp_path), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "help" and completed.stderr == b""


def test_cli_process_control_propagates(monkeypatch):
    def stop(*_args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(surface, "observe_production_decision_surface", stop)
    with pytest.raises(KeyboardInterrupt):
        cli.main(_args())



@pytest.mark.parametrize("selected", ["research", "production"])
def test_reference_source_option_is_explicit(monkeypatch, capsys, selected):
    calls = []
    def observe(*args, **kwargs):
        calls.append(kwargs)
        return _result("SAME_CHAMPION_SEGMENT")
    monkeypatch.setattr(surface, "observe_production_decision_surface", observe)
    assert cli.main(_args() + ["--reference-source", selected]) == 0
    assert calls == [{"reference_source": selected}]
    assert json.loads(capsys.readouterr().out)["status"] == "SAME_CHAMPION_SEGMENT"
