import json
import os
import sys

import pytest

from tests.quantpits.research.test_forward_bootstrap import (
    CYCLE,
    SOURCE_CYCLE,
    fresh_bootstrap_workspace,
)
from tests.quantpits.research.test_forward_definition_evidence import (
    fresh_evidence_workspace,
)
from quantpits.scripts import (
    publish_fresh_champion_segment_matched_bootstrap as cli,
)


ACTION = "PUBLISH_ONE_FRESH_CHAMPION_SEGMENT_MATCHED_BOOTSTRAP_V1"


def _preflight_args(value):
    production, research, activation, definitions, evidence, bootstraps = value
    return [
        "preflight",
        "--production-workspace", str(production),
        "--research-workspace", str(research),
        "--definition-evidence-cycle", CYCLE,
        "--bootstrap-source-cycle", SOURCE_CYCLE,
        "--activation", str(activation),
        "--definition-store", str(definitions),
        "--evidence-store", str(evidence),
        "--bootstrap-store", str(bootstraps),
    ]


def _publish_args(preflight, plan):
    return [
        "publish", *preflight[1:],
        "--expected-bootstrap-set-id", plan["bootstrap_set_id"],
        "--expected-bootstrap-request-digest", json.dumps(
            plan["bootstrap_request_digest"],
            sort_keys=True, separators=(",", ":"),
        ),
        "--authorization-action", ACTION,
    ]


def test_fresh_bootstrap_cli_requires_complete_arguments_without_echo(capsys):
    private = "PRIVATE_PATH_TOKEN_MODEL_A_0.001"
    assert cli.main([
        "preflight", "--production-workspace", private,
        "--unknown", private,
    ]) == 2
    captured = capsys.readouterr()
    assert captured.err == "" and private not in captured.out
    assert json.loads(captured.out) == {
        "status": "blocked",
        "error": {
            "type": "FreshMatchedBootstrapArgumentError",
            "reason_code": "ARGUMENT_INVALID",
        },
    }


def test_fresh_bootstrap_cli_preflight_commit_adopt_are_private_and_canonical(
    fresh_bootstrap_workspace, capsys,
):
    preflight = _preflight_args(fresh_bootstrap_workspace)
    assert cli.main(preflight) == 0
    raw = capsys.readouterr().out.strip()
    plan = json.loads(raw)
    assert raw == json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    assert plan["status"] == "PREPARED" and plan["target_state"] == "ABSENT"
    publish = _publish_args(preflight, plan)
    assert cli.main(publish) == 0
    committed = json.loads(capsys.readouterr().out)
    assert committed["status"] == "COMMITTED"
    assert committed["portfolio_bootstrap_complete"] is True
    assert cli.main(publish) == 0
    adopted = json.loads(capsys.readouterr().out)
    assert adopted["status"] == "ADOPTED" and adopted["did_write"] is False
    rendered = json.dumps((plan, committed, adopted))
    for private in map(str, fresh_bootstrap_workspace[:2]):
        assert private not in rendered
    assert all(token not in rendered for token in (
        "current_cash", "positions", "instrument", "book_cost",
    ))


def test_fresh_bootstrap_cli_rejects_noncanonical_digest_and_old_action(
    fresh_bootstrap_workspace, capsys,
):
    preflight = _preflight_args(fresh_bootstrap_workspace)
    assert cli.main(preflight) == 0
    plan = json.loads(capsys.readouterr().out)
    publish = _publish_args(preflight, plan)
    digest_index = publish.index("--expected-bootstrap-request-digest") + 1
    compact = publish[digest_index]
    publish[digest_index] = json.dumps(plan["bootstrap_request_digest"])
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["error"]["reason_code"] == (
        "ARGUMENT_INVALID"
    )
    publish[digest_index] = compact
    publish[-1] = "PUBLISH_ONE_MATCHED_FORWARD_BOOTSTRAP_V1"
    assert cli.main(publish) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


@pytest.mark.parametrize("status,code", [
    ("COMMITTED", 0), ("ADOPTED", 0), ("CONFLICT", 3), ("UNCERTAIN", 4),
])
def test_fresh_bootstrap_cli_maps_terminal_statuses(
    monkeypatch, capsys, status, code,
):
    class Result:
        @property
        def status(self):
            return status

        def to_safe_summary_dict(self):
            return {"status": status, "portfolio_bootstrap_complete": False}

    import quantpits.research.forward_bootstrap as bootstrap
    monkeypatch.setattr(
        bootstrap, "publish_fresh_champion_segment_matched_bootstrap",
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
        "--definition-evidence-cycle", CYCLE,
        "--bootstrap-source-cycle", SOURCE_CYCLE,
        "--activation", "/private/activation",
        "--definition-store", "/private/definitions",
        "--evidence-store", "/private/evidence",
        "--bootstrap-store", "/private/bootstraps",
        "--expected-bootstrap-set-id", "bootstrap.private",
        "--expected-bootstrap-request-digest", raw,
        "--authorization-action", ACTION,
    ]) == code
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_fresh_bootstrap_cli_help_import_env_cwd_and_backends_are_unchanged(
    capsys,
):
    before_env, before_cwd = dict(os.environ), os.getcwd()
    before = {name for name in sys.modules if name.startswith(("qlib", "mlflow"))}
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    assert "production-workspace" not in capsys.readouterr().out
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd
    assert {name for name in sys.modules if name.startswith(("qlib", "mlflow"))} == before


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_fresh_bootstrap_cli_process_control_propagates(monkeypatch, exception):
    import quantpits.research.forward_bootstrap as bootstrap
    monkeypatch.setattr(
        bootstrap, "prepare_fresh_champion_segment_matched_bootstrap",
        lambda *_args: (_ for _ in ()).throw(exception()),
    )
    with pytest.raises(exception):
        cli.main([
            "preflight", "--production-workspace", "/private/production",
            "--research-workspace", "/private/research",
            "--definition-evidence-cycle", CYCLE,
            "--bootstrap-source-cycle", SOURCE_CYCLE,
            "--activation", "/private/activation",
            "--definition-store", "/private/definitions",
            "--evidence-store", "/private/evidence",
            "--bootstrap-store", "/private/bootstraps",
        ])
