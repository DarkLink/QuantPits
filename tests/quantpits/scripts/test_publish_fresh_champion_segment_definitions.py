import json
import os
import sys

import pytest

from tests.quantpits.research.test_forward_observation import CYCLE, _fresh_split_bundle
from quantpits.scripts import publish_fresh_champion_segment_definitions as cli


ACTION = "PUBLISH_ONE_FRESH_CHAMPION_SEGMENT_DEFINITION_BUNDLE_V1"


@pytest.fixture
def fresh_cli_workspace(tmp_path):
    production, research, activation, _cycle, _reference = _fresh_split_bundle(tmp_path)
    store = research / "research" / "shadow_v1" / "definitions"
    store.mkdir(mode=0o700)
    return production, research, activation, store


def _preflight_args(value):
    production, research, activation, store = value
    return [
        "preflight",
        "--production-workspace", str(production),
        "--research-workspace", str(research),
        "--evidence-cycle", CYCLE,
        "--activation", str(activation),
        "--definition-store", str(store),
    ]


def _publish_args(preflight, plan):
    return [
        "publish", *preflight[1:],
        "--expected-definition-set-id", plan["definition_set_id"],
        "--expected-request-digest", json.dumps(
            plan["request_digest"], sort_keys=True, separators=(",", ":"),
        ),
        "--authorization-action", ACTION,
    ]


def test_fresh_cli_requires_explicit_complete_arguments_without_echo(capsys):
    private = "PRIVATE_PATH_TOKEN_MODEL_A_0.001"
    assert cli.main(["preflight", "--production-workspace", private, "--unknown", private]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert private not in captured.out
    assert json.loads(captured.out) == {
        "status": "blocked",
        "error": {
            "type": "FreshDefinitionPublicationArgumentError",
            "reason_code": "ARGUMENT_INVALID",
        },
    }


def test_fresh_cli_preflight_publish_and_adopt_are_canonical_and_private(
    fresh_cli_workspace, capsys,
):
    preflight = _preflight_args(fresh_cli_workspace)
    assert cli.main(preflight) == 0
    raw = capsys.readouterr().out.strip()
    plan = json.loads(raw)
    assert raw == json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert plan["status"] == "PREPARED"
    publish = _publish_args(preflight, plan)
    assert cli.main(publish) == 0
    committed = json.loads(capsys.readouterr().out)
    assert committed["status"] == "COMMITTED"
    assert committed["definition_bytes_durable"] is True
    assert cli.main(publish) == 0
    adopted = json.loads(capsys.readouterr().out)
    assert adopted["status"] == "ADOPTED" and adopted["did_write"] is False
    text = json.dumps({"plan": plan, "committed": committed, "adopted": adopted})
    for private in map(str, fresh_cli_workspace[:2]):
        assert private not in text


def test_fresh_cli_rejects_noncanonical_digest_and_old_action(
    fresh_cli_workspace, capsys,
):
    preflight = _preflight_args(fresh_cli_workspace)
    assert cli.main(preflight) == 0
    plan = json.loads(capsys.readouterr().out)
    publish = _publish_args(preflight, plan)
    publish[-3] = json.dumps(plan["request_digest"])
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["error"]["reason_code"] == "ARGUMENT_INVALID"
    publish[-3] = json.dumps(plan["request_digest"], sort_keys=True, separators=(",", ":"))
    publish[-1] = "PUBLISH_ONE_FROZEN_DEFINITION_BUNDLE_V1"
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


@pytest.mark.parametrize("status,code", [
    ("COMMITTED", 0), ("ADOPTED", 0), ("CONFLICT", 3), ("UNCERTAIN", 4),
])
def test_fresh_cli_maps_all_publication_outcomes_to_stable_codes(
    monkeypatch, capsys, status, code,
):
    class Result:
        def to_safe_summary_dict(self):
            return {
                "status": status,
                "definition_bytes_durable": status in {"COMMITTED", "ADOPTED"},
            }

    import quantpits.research.forward_definition_publication as publication
    monkeypatch.setattr(
        publication, "publish_fresh_champion_segment_definition_bundle",
        lambda *_args: Result(),
    )
    digest = {
        "algorithm": "sha256", "domain": "canonical_json",
        "value": "0" * 64, "size_bytes": 1,
    }
    assert cli.main([
        "publish",
        "--production-workspace", "/private/production",
        "--research-workspace", "/private/research",
        "--evidence-cycle", CYCLE,
        "--activation", "/private/activation",
        "--definition-store", "/private/store",
        "--expected-definition-set-id", "definition.id",
        "--expected-request-digest", json.dumps(digest, sort_keys=True, separators=(",", ":")),
        "--authorization-action", ACTION,
    ]) == code
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_fresh_cli_help_import_environment_cwd_and_optional_backends_are_unchanged(capsys):
    before_env, before_cwd = dict(os.environ), os.getcwd()
    before = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "production-workspace" not in capsys.readouterr().out
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd
    assert {name for name in sys.modules if name.startswith(("qlib", "mlflow"))} == before


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_fresh_cli_process_control_is_not_converted(monkeypatch, exception):
    import quantpits.research.forward_definition_publication as publication
    monkeypatch.setattr(
        publication, "prepare_fresh_champion_segment_definition_publication",
        lambda *_args: (_ for _ in ()).throw(exception()),
    )
    with pytest.raises(exception):
        cli.main([
            "preflight",
            "--production-workspace", "/private/production",
            "--research-workspace", "/private/research",
            "--evidence-cycle", CYCLE,
            "--activation", "/private/activation",
            "--definition-store", "/private/store",
        ])
