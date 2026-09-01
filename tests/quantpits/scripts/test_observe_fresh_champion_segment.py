import json
import os
import sys

import pytest

from quantpits.scripts import observe_fresh_champion_segment as cli


ARGS = [
    "--production-workspace", "/private/production",
    "--research-workspace", "/private/research",
    "--evidence-cycle", "2026-08-14",
    "--activation", "/private/research/activation.json",
]


def test_cli_requires_exact_arguments_and_never_echoes_unknown_private_values(capsys):
    private = "PRIVATE_TOKEN_123"
    assert cli.main(ARGS + ["--unknown", private]) == 2
    captured = capsys.readouterr()
    assert captured.err == "" and private not in captured.out
    assert json.loads(captured.out)["error"] == {
        "type": "FreshChampionSegmentArgumentError", "reason_code": "ARGUMENT_INVALID",
    }


def test_cli_success_and_blocked_are_one_canonical_privacy_safe_json(monkeypatch, capsys):
    class Candidate:
        def to_safe_summary_dict(self):
            return {"status": "READY", "fresh_segment_candidate_ready": True}

    import quantpits.research.forward_observation as observation
    monkeypatch.setattr(observation, "observe_fresh_champion_segment_candidate", lambda *_args: Candidate())
    assert cli.main(ARGS) == 0
    output = capsys.readouterr().out
    assert output == json.dumps(json.loads(output), sort_keys=True, separators=(",", ":")) + "\n"
    private = "/private/research SECRET_MODEL 0.001"
    monkeypatch.setattr(
        observation, "observe_fresh_champion_segment_candidate",
        lambda *_args: (_ for _ in ()).throw(observation.ForwardObservationInputError(private)),
    )
    assert cli.main(ARGS) == 2
    output = capsys.readouterr().out
    assert private not in output and json.loads(output)["status"] == "blocked"


def test_cli_help_import_env_cwd_and_optional_modules_are_unchanged(capsys):
    before_env, before_cwd = dict(os.environ), os.getcwd()
    before_modules = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "--production-workspace" in capsys.readouterr().out
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd
    assert {name for name in sys.modules if name.startswith(("qlib", "mlflow"))} == before_modules


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cli_process_control_propagates(monkeypatch, exception):
    import quantpits.research.forward_observation as observation
    monkeypatch.setattr(
        observation, "observe_fresh_champion_segment_candidate",
        lambda *_args: (_ for _ in ()).throw(exception()),
    )
    with pytest.raises(exception):
        cli.main(ARGS)
