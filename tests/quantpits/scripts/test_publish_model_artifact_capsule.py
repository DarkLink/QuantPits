import json
import os
import sys

import pytest

import quantpits.scripts.publish_model_artifact_capsule as cli


class _Result:
    def __init__(self, status="PREPARED"):
        self.status = status

    def to_safe_summary_dict(self):
        return {
            "status": self.status, "capsule_id": "modelcapsule." + "a" * 64,
            "model_artifact_retention_complete": self.status in {"COMMITTED", "ADOPTED"},
        }


def _common():
    return [
        "--production-workspace", "/private/production",
        "--research-workspace", "/private/research",
        "--evidence-cycle", "2026-08-28",
        "--activation", "/private/research/activation.json",
        "--definition-store", "/private/research/definitions",
        "--evidence-store", "/private/research/evidence",
        "--capsule-store", "/private/research/capsules",
    ]


def test_cli_preflight_publish_adopt_and_exit_mapping(monkeypatch, capsys):
    import quantpits.research.model_artifact_capsule as module
    calls = []
    monkeypatch.setattr(module, "prepare_definition_bound_model_artifact_capsule", lambda *args: calls.append(("preflight", args)) or _Result())
    monkeypatch.setattr(module, "publish_definition_bound_model_artifact_capsule", lambda *args: calls.append(("publish", args)) or _Result("COMMITTED"))
    monkeypatch.setattr(module, "adopt_definition_bound_model_artifact_capsule", lambda *args: calls.append(("adopt", args)) or _Result("ADOPTED"))
    assert cli.main(["preflight", *_common()]) == 0
    digest = json.dumps({"algorithm": "sha256", "domain": "canonical_json", "size_bytes": 1, "value": "0" * 64}, sort_keys=True, separators=(",", ":"))
    assert cli.main(["publish", *_common(), "--expected-capsule-id", "modelcapsule." + "a" * 64,
                     "--expected-request-digest", digest, "--authorization-action", "ACTION"]) == 0
    assert cli.main(["adopt", *_common(), "--capsule-id", "modelcapsule." + "a" * 64]) == 0
    assert [row[0] for row in calls] == ["preflight", "publish", "adopt"]
    for line in capsys.readouterr().out.splitlines():
        assert json.dumps(json.loads(line), sort_keys=True, separators=(",", ":")) == line


def test_cli_parse_error_does_not_echo_private_argument(capsys):
    secret = "PRIVATE_TOKEN_MUST_NOT_APPEAR"
    assert cli.main(["preflight", *_common(), "--unknown", secret]) == 2
    output = capsys.readouterr()
    assert secret not in output.out and secret not in output.err
    assert json.loads(output.out)["error"]["reason_code"] == "ARGUMENT_INVALID"


def test_cli_contract_error_is_privacy_safe(monkeypatch, capsys):
    import quantpits.research.model_artifact_capsule as module
    def fail(*_args):
        raise module.ModelArtifactCapsuleInputError("PRIVATE recorder/path/config")
    monkeypatch.setattr(module, "prepare_definition_bound_model_artifact_capsule", fail)
    assert cli.main(["preflight", *_common()]) == 2
    output = capsys.readouterr().out
    assert "PRIVATE" not in output
    assert json.loads(output)["error"]["reason_code"] == "CONTRACT_INVALID"


def test_cli_import_and_help_have_no_environment_cwd_or_optional_dependency_effects():
    before_env = dict(os.environ)
    before_cwd = os.getcwd()
    before_modules = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    with pytest.raises(SystemExit) as stopped:
        cli._parser().parse_args(["--help"])
    assert stopped.value.code == 0
    assert dict(os.environ) == before_env
    assert os.getcwd() == before_cwd
    assert {name for name in sys.modules if name.startswith(("qlib", "mlflow"))} == before_modules
