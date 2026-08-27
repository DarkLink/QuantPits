import json
import os
import sys

import pytest

from quantpits.scripts import observe_forward_definitions as cli


def test_cli_requires_exact_workspace_cycle_and_activation_arguments(capsys):
    assert cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "blocked",
        "error": {
            "type": "ForwardObservationArgumentError",
            "reason_code": "ARGUMENT_INVALID",
        },
    }


def test_cli_unknown_arguments_never_echo_private_tokens(capsys):
    private = "PRIVATE_TOKEN_123"
    assert cli.main([
        "--workspace", "/private/workspace", "--evidence-cycle", "2026-08-14",
        "--activation", "/private/activation.json", "--unexpected", private,
    ]) == 2
    captured = capsys.readouterr()
    assert private not in captured.out + captured.err
    assert captured.err == ""
    assert json.loads(captured.out)["error"] == {
        "type": "ForwardObservationArgumentError",
        "reason_code": "ARGUMENT_INVALID",
    }


def test_cli_success_prints_one_safe_machine_readable_summary_and_writes_nothing(monkeypatch, capsys):
    class Candidate:
        def to_safe_summary_dict(self):
            return {"status": "complete", "publication_capability": False}

    import quantpits.research.forward_observation as observation
    monkeypatch.setattr(observation, "observe_shadow_forward_definition_candidate", lambda *_args: Candidate())
    assert cli.main(["--workspace", "/private/workspace", "--evidence-cycle", "2026-08-14", "--activation", "/private/activation.json"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"status": "complete", "publication_capability": False}


def test_cli_blocked_result_redacts_private_values_and_returns_nonzero(monkeypatch, capsys):
    import quantpits.research.forward_observation as observation
    private = "/private/workspace/secret MODEL_A 0.001"
    monkeypatch.setattr(observation, "observe_shadow_forward_definition_candidate", lambda *_args: (_ for _ in ()).throw(observation.ForwardObservationInputError(private)))
    assert cli.main(["--workspace", "/private/workspace", "--evidence-cycle", "2026-08-14", "--activation", "/private/activation.json"]) == 2
    output = capsys.readouterr().out
    assert private not in output
    assert json.loads(output)["status"] == "blocked"


def test_cli_help_and_import_do_not_resolve_workspace_or_load_qlib_mlflow(capsys):
    before_env, before_cwd = dict(os.environ), os.getcwd()
    before_modules = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "--workspace" in capsys.readouterr().out
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd
    assert {name for name in sys.modules if name.startswith(("qlib", "mlflow"))} == before_modules


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cli_process_control_is_not_converted_to_a_normal_blocked_result(monkeypatch, exception):
    import quantpits.research.forward_observation as observation
    monkeypatch.setattr(observation, "observe_shadow_forward_definition_candidate", lambda *_args: (_ for _ in ()).throw(exception()))
    with pytest.raises(exception):
        cli.main(["--workspace", "/private/workspace", "--evidence-cycle", "2026-08-14", "--activation", "/private/activation.json"])
