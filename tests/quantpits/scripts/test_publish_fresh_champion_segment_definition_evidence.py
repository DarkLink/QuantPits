import json
import os
import sys

import pytest

from tests.quantpits.research.test_forward_definition_evidence import (
    fresh_evidence_workspace,
)
from tests.quantpits.research.test_forward_observation import CYCLE
from quantpits.scripts import publish_fresh_champion_segment_definition_evidence as cli


ACTION = "PUBLISH_ONE_FRESH_CHAMPION_SEGMENT_DEFINITION_EVIDENCE_RECORD_V1"


def _preflight_args(value):
    production, research, activation, definitions, evidence = value
    return [
        "preflight",
        "--production-workspace", str(production),
        "--research-workspace", str(research),
        "--evidence-cycle", CYCLE,
        "--activation", str(activation),
        "--definition-store", str(definitions),
        "--evidence-store", str(evidence),
    ]


def _publish_args(preflight, plan):
    return [
        "publish", *preflight[1:],
        "--expected-definition-set-id", plan["definition_set_id"],
        "--expected-definition-request-digest", json.dumps(
            plan["definition_request_digest"], sort_keys=True, separators=(",", ":"),
        ),
        "--expected-evidence-request-digest", json.dumps(
            plan["evidence_request_digest"], sort_keys=True, separators=(",", ":"),
        ),
        "--authorization-action", ACTION,
    ]


def test_fresh_evidence_cli_requires_complete_arguments_without_echo(capsys):
    private = "PRIVATE_PATH_TOKEN_MODEL_A_0.001"
    assert cli.main([
        "preflight", "--production-workspace", private, "--unknown", private,
    ]) == 2
    captured = capsys.readouterr()
    assert captured.err == "" and private not in captured.out
    assert json.loads(captured.out) == {
        "status": "blocked",
        "error": {
            "type": "FreshDefinitionEvidenceArgumentError",
            "reason_code": "ARGUMENT_INVALID",
        },
    }


def test_fresh_evidence_cli_preflight_publish_adopt_are_canonical_and_private(
    fresh_evidence_workspace, capsys,
):
    preflight = _preflight_args(fresh_evidence_workspace)
    assert cli.main(preflight) == 0
    raw = capsys.readouterr().out.strip()
    plan = json.loads(raw)
    assert raw == json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert plan["status"] == "PREPARED"
    publish = _publish_args(preflight, plan)
    assert cli.main(publish) == 0
    committed = json.loads(capsys.readouterr().out)
    assert committed["status"] == "COMMITTED"
    assert committed["definition_evidence_complete"] is True
    assert cli.main(publish) == 0
    adopted = json.loads(capsys.readouterr().out)
    assert adopted["status"] == "ADOPTED" and adopted["did_write"] is False
    text = json.dumps({"plan": plan, "committed": committed, "adopted": adopted})
    for private in map(str, fresh_evidence_workspace[:2]):
        assert private not in text


def test_fresh_evidence_cli_rejects_noncanonical_digest_and_old_action(
    fresh_evidence_workspace, capsys,
):
    preflight = _preflight_args(fresh_evidence_workspace)
    assert cli.main(preflight) == 0
    plan = json.loads(capsys.readouterr().out)
    publish = _publish_args(preflight, plan)
    definition_index = publish.index("--expected-definition-request-digest") + 1
    compact = publish[definition_index]
    publish[definition_index] = json.dumps(plan["definition_request_digest"])
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["error"]["reason_code"] == "ARGUMENT_INVALID"
    publish[definition_index] = compact
    publish[-1] = "PUBLISH_ONE_FORWARD_DEFINITION_EVIDENCE_RECORD_V1"
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


@pytest.mark.parametrize("status,code", [
    ("COMMITTED", 0), ("ADOPTED", 0), ("CONFLICT", 3), ("UNCERTAIN", 4),
])
def test_fresh_evidence_cli_maps_terminal_statuses_to_stable_codes(
    monkeypatch, capsys, status, code,
):
    class Result:
        @property
        def status(self):
            return status

        def to_safe_summary_dict(self):
            return {"status": status, "definition_evidence_complete": False}

    import quantpits.research.forward_definition_evidence as evidence
    monkeypatch.setattr(
        evidence, "publish_fresh_champion_segment_definition_evidence",
        lambda *_args: Result(),
    )
    digest = {
        "algorithm": "sha256", "domain": "canonical_json",
        "value": "0" * 64, "size_bytes": 1,
    }
    raw = json.dumps(digest, sort_keys=True, separators=(",", ":"))
    assert cli.main([
        "publish", "--production-workspace", "/private/production",
        "--research-workspace", "/private/research",
        "--evidence-cycle", CYCLE, "--activation", "/private/activation",
        "--definition-store", "/private/definitions",
        "--evidence-store", "/private/evidence",
        "--expected-definition-set-id", "definition.id",
        "--expected-definition-request-digest", raw,
        "--expected-evidence-request-digest", raw,
        "--authorization-action", ACTION,
    ]) == code
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_fresh_evidence_cli_help_import_env_cwd_and_backends_are_unchanged(capsys):
    before_env, before_cwd = dict(os.environ), os.getcwd()
    before = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "production-workspace" not in capsys.readouterr().out
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd
    assert {name for name in sys.modules if name.startswith(("qlib", "mlflow"))} == before


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_fresh_evidence_cli_process_control_is_not_converted(monkeypatch, exception):
    import quantpits.research.forward_definition_evidence as evidence
    monkeypatch.setattr(
        evidence, "prepare_fresh_champion_segment_definition_evidence",
        lambda *_args: (_ for _ in ()).throw(exception()),
    )
    with pytest.raises(exception):
        cli.main([
            "preflight", "--production-workspace", "/private/production",
            "--research-workspace", "/private/research",
            "--evidence-cycle", CYCLE, "--activation", "/private/activation",
            "--definition-store", "/private/definitions",
            "--evidence-store", "/private/evidence",
        ])
