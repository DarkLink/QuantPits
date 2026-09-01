"""CLI contract tests for signal input capsules."""

import json

import pytest

import quantpits.scripts.signal_input_capsule as cli
from quantpits.evidence.contracts import TypedDigest


class _Result:
    def __init__(self, status):
        self.status = status

    def to_safe_dict(self):
        return {
            "schema_version": 1,
            "capsule_kind": "WEEKLY_CRITICAL_SIGNAL_INPUT_CAPSULE_V1",
            "status": self.status, "reason_code": self.status,
            "critical_signal_retention_complete": self.status in {"COMMITTED", "ADOPTED"},
            "same_champion_segment": False, "definition_bound": False,
            "intent_capability": False, "epoch_started": False,
            "prospective_claim": False, "promotion_capability": False,
            "did_write": self.status == "COMMITTED",
        }


@pytest.mark.parametrize("action,status", [
    ("prepare", "PREPARED"), ("publish", "COMMITTED"), ("adopt", "ADOPTED"),
])
def test_cli_actions_emit_canonical_privacy_safe_json(monkeypatch, capsys, action, status):
    import quantpits.research.signal_input_capsule as api
    monkeypatch.setattr(api, "prepare_signal_input_capsule", lambda *_: _Result(status))
    monkeypatch.setattr(api, "publish_signal_input_capsule", lambda *_: _Result(status))
    monkeypatch.setattr(api, "adopt_signal_input_capsule", lambda *_: _Result(status))
    argv = [
        action, "--production-workspace", "/private/production",
        "--research-workspace", "/private/research", "--cycle", "2026-08-21",
        "--capsule-store", "/private/store",
    ]
    if action in {"publish", "adopt"}:
        argv += ["--capsule-id", "signalcapsule." + "1" * 64]
    if action == "publish":
        argv += [
            "--request-digest", json.dumps(TypedDigest.canonical({}).to_dict()),
            "--authorization-action", api.AUTHORIZATION_ACTION,
        ]
    assert cli.main(argv) == 0
    raw = capsys.readouterr().out.encode()
    assert raw == (
        json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()
    assert b"/private" not in raw
    assert json.loads(raw)["status"] == status


def test_cli_unknown_argument_and_private_token_are_not_echoed(capsys):
    assert cli.main(["prepare", "--unknown", "PRIVATE_TOKEN"]) == 2
    captured = capsys.readouterr()
    assert "PRIVATE_TOKEN" not in captured.out
    assert "PRIVATE_TOKEN" not in captured.err


def test_cli_precondition_blocked_has_stable_exit_and_no_capability(monkeypatch, capsys):
    import quantpits.research.signal_input_capsule as api
    monkeypatch.setattr(
        api, "prepare_signal_input_capsule",
        lambda *_: _Result("PRECONDITION_BLOCKED"),
    )
    assert cli.main([
        "prepare", "--production-workspace", "/p", "--research-workspace", "/r",
        "--cycle", "2026-08-21", "--capsule-store", "/s",
    ]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PRECONDITION_BLOCKED"
    assert payload["critical_signal_retention_complete"] is False


def test_cli_help_and_import_have_no_workspace_effects(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", "unchanged")
    before = (tuple(tmp_path.iterdir()), dict(__import__("os").environ), str(tmp_path))
    assert cli.main(["--help"]) == 0
    after = (tuple(tmp_path.iterdir()), dict(__import__("os").environ), str(tmp_path))
    assert before == after
    assert json.loads(capsys.readouterr().out)["status"] == "help"


@pytest.mark.parametrize("status,exit_code", [("CONFLICT", 4), ("UNCERTAIN", 5)])
def test_cli_terminal_negative_exit_codes(monkeypatch, capsys, status, exit_code):
    import quantpits.research.signal_input_capsule as api
    monkeypatch.setattr(api, "adopt_signal_input_capsule", lambda *_: _Result(status))
    assert cli.main([
        "adopt", "--production-workspace", "/p", "--research-workspace", "/r",
        "--cycle", "2026-08-21", "--capsule-store", "/s",
        "--capsule-id", "signalcapsule." + "2" * 64,
    ]) == exit_code
    assert json.loads(capsys.readouterr().out)["status"] == status
