import json
import os
import sys

import pytest

from tests.quantpits.research.test_forward_definition_evidence import evidence_workspace
from tests.quantpits.research.test_forward_observation import CYCLE
from quantpits.scripts import publish_forward_evidence as cli


def _preflight_args(value):
    root, activation, definitions, evidence = value
    return [
        "preflight", "--workspace", str(root), "--evidence-cycle", CYCLE,
        "--activation", str(activation), "--definition-store", str(definitions),
        "--evidence-store", str(evidence),
    ]


def test_cli_requires_explicit_action_and_all_physical_inputs(capsys):
    assert cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "blocked",
        "error": {
            "type": "ForwardDefinitionEvidenceArgumentError",
            "reason_code": "ARGUMENT_INVALID",
        },
    }


def test_cli_preflight_is_safe_machine_readable_and_zero_write(evidence_workspace, capsys):
    before = sorted(
        path.relative_to(evidence_workspace[0]).as_posix()
        for path in evidence_workspace[0].rglob("*")
    )
    assert cli.main(_preflight_args(evidence_workspace)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PREPARED"
    assert summary["definition_store_status"] == "ADOPTED"
    assert sorted(
        path.relative_to(evidence_workspace[0]).as_posix()
        for path in evidence_workspace[0].rglob("*")
    ) == before


def test_cli_publish_requires_two_exact_digests_and_authorization_literal(
    evidence_workspace, capsys,
):
    args = _preflight_args(evidence_workspace)
    assert cli.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    publish = [
        "publish", *args[1:],
        "--expected-definition-set-id", plan["definition_set_id"],
        "--expected-definition-request-digest", json.dumps(
            plan["definition_request_digest"], sort_keys=True, separators=(",", ":"),
        ),
        "--expected-evidence-request-digest", json.dumps(
            plan["evidence_request_digest"], sort_keys=True, separators=(",", ":"),
        ),
        "--authorization-action", "WRONG",
    ]
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    publish[-1] = "PUBLISH_ONE_FORWARD_DEFINITION_EVIDENCE_RECORD_V1"
    definition_digest_index = publish.index("--expected-definition-request-digest") + 1
    compact_definition = publish[definition_digest_index]
    publish[definition_digest_index] = json.dumps(plan["definition_request_digest"])
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["error"]["reason_code"] == "ARGUMENT_INVALID"
    publish[definition_digest_index] = compact_definition
    assert cli.main(publish) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "COMMITTED"
    assert cli.main(publish) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["status"] == "ADOPTED" and replay["did_write"] is False


def test_cli_unknown_or_invalid_arguments_never_echo_private_values(capsys):
    private = "PRIVATE_EVIDENCE_PATH_TOKEN_0.001"
    assert cli.main(["preflight", "--workspace", private, "--unknown", private]) == 2
    captured = capsys.readouterr()
    assert captured.err == "" and private not in captured.out


@pytest.mark.parametrize("status,code", [
    ("COMMITTED", 0), ("ADOPTED", 0), ("CONFLICT", 3), ("UNCERTAIN", 4),
])
def test_cli_maps_terminal_statuses_to_stable_exit_codes(monkeypatch, capsys, status, code):
    class Result:
        @property
        def status(self):
            return status

        def to_safe_summary_dict(self):
            return {"status": status, "definition_evidence_complete": status in {"COMMITTED", "ADOPTED"}}

    import quantpits.research.forward_definition_evidence as module
    monkeypatch.setattr(module, "publish_forward_definition_evidence", lambda *_args: Result())
    digest = {
        "algorithm": "sha256", "domain": "canonical_json",
        "value": "0" * 64, "size_bytes": 1,
    }
    raw = json.dumps(digest, sort_keys=True, separators=(",", ":"))
    assert cli.main([
        "publish", "--workspace", "/private/workspace",
        "--evidence-cycle", CYCLE, "--activation", "/private/activation",
        "--definition-store", "/private/definitions",
        "--evidence-store", "/private/evidence",
        "--expected-definition-set-id", "definition.id",
        "--expected-definition-request-digest", raw,
        "--expected-evidence-request-digest", raw,
        "--authorization-action", "PUBLISH_ONE_FORWARD_DEFINITION_EVIDENCE_RECORD_V1",
    ]) == code
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_cli_help_and_import_do_not_resolve_workspace_or_load_optional_backends(capsys):
    before_env, before_cwd = dict(os.environ), os.getcwd()
    before = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "preflight" in capsys.readouterr().out
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd
    assert {name for name in sys.modules if name.startswith(("qlib", "mlflow"))} == before


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cli_process_control_is_not_converted_to_a_normal_result(monkeypatch, exception):
    import quantpits.research.forward_definition_evidence as module
    monkeypatch.setattr(
        module, "prepare_forward_definition_evidence",
        lambda *_args: (_ for _ in ()).throw(exception()),
    )
    with pytest.raises(exception):
        cli.main([
            "preflight", "--workspace", "/private/workspace",
            "--evidence-cycle", CYCLE, "--activation", "/private/activation",
            "--definition-store", "/private/definitions",
            "--evidence-store", "/private/evidence",
        ])
