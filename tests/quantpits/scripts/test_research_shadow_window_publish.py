from __future__ import annotations

import importlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import quantpits.scripts.research_shadow_window_publish as publish_cli
from quantpits.scripts.research_shadow_window import build_parser as build_zero_write_parser
from tests.quantpits.research.test_historical_cycle import DATES
from tests.quantpits.research.test_historical_window import build_window


ARGS = [
    "--workspace", "/private/workspace", "--qlib-data-dir", "/private/qlib",
    "--profile", "/private/profile.json", "--sealed-cycle", "2026-08-21",
    "--preferred-start", "2026-07-17", "--preferred-end", "2026-08-14",
    "--window-size", "5", "--top-k", "22", "--output-dir", "/tmp/b3b-test",
]


def _wire(monkeypatch, receipt=None, error=None):
    monkeypatch.setattr(publish_cli, "load_replay_profile", lambda _path: object())
    monkeypatch.setattr(publish_cli, "load_sealed_replay_inputs", lambda *_args, **_kwargs: object())
    stage = {"universe_identity": {"name": "market"}}
    monkeypatch.setattr(publish_cli, "ResearchRankingReplay", lambda *_args, **_kwargs: SimpleNamespace(run=lambda **_kw: stage))
    result = object()
    monkeypatch.setattr(publish_cli, "HistoricalShadowWindowReplay", lambda **_kwargs: SimpleNamespace(run=lambda: result))
    if error is not None:
        monkeypatch.setattr(publish_cli, "write_historical_window_output", lambda *_args: (_ for _ in ()).throw(error))
    else:
        monkeypatch.setattr(publish_cli, "write_historical_window_output", lambda *_args: receipt)


def _receipt(status):
    return SimpleNamespace(
        status=status,
        reason_code={"COMMITTED": "COMMITTED", "CONFLICT": "OUTPUT_ALREADY_EXISTS", "UNCERTAIN": "PUBLICATION_POSTCONDITION_UNCERTAIN"}[status],
        did_write=status != "CONFLICT", operation_id="a" * 64,
        result_digest={"algorithm": "sha256", "value": "b" * 64},
        manifest_digest=({"algorithm": "sha256", "value": "c" * 64} if status == "COMMITTED" else None),
        member_count=29 if status == "COMMITTED" else 0,
    )


def test_publish_cli_requires_exact_arguments_and_rejects_resume_selection_and_research_options():
    parser = publish_cli.build_parser()
    assert set(vars(parser.parse_args(ARGS))) == {
        "workspace", "qlib_data_dir", "profile", "sealed_cycle", "preferred_start",
        "preferred_end", "window_size", "top_k", "output_dir",
    }
    for forbidden in ("--resume", "--overwrite", "--best-arm", "--research-root"):
        with pytest.raises(SystemExit):
            parser.parse_args(ARGS + [forbidden, "x"])


def test_publish_cli_committed_receipt_is_one_line_safe_and_exact(tmp_path, capsys):
    runner, _stage_a, _inputs, profile, workspace, qlib = build_window(tmp_path / "source", 4)
    output = Path("/tmp") / ("quantpits_b3b_cli_%s" % tmp_path.name)
    shutil.rmtree(str(output), ignore_errors=True)
    args = [
        "--workspace", str(workspace), "--qlib-data-dir", str(qlib),
        "--profile", str(profile._private_path), "--sealed-cycle", DATES[-1],
        "--preferred-start", DATES[0], "--preferred-end", DATES[-1],
        "--window-size", "4", "--top-k", "2", "--output-dir", str(output),
    ]
    try:
        environment_before = dict(os.environ)
        cwd_before = os.getcwd()
        assert publish_cli.main(args) == 0
        assert dict(os.environ) == environment_before and os.getcwd() == cwd_before
        captured = capsys.readouterr()
        assert captured.err == "" and captured.out.count("\n") == 1
        payload = json.loads(captured.out)
        assert set(payload) == {
            "status", "reason_code", "did_write", "operation_id", "result_digest",
            "manifest_digest", "member_count", "prospective_claim", "promotion_capability",
        }
        assert payload["status"] == "COMMITTED" and payload["member_count"] == 24
        assert str(output) not in captured.out and str(profile._private_path) not in captured.out
        assert (output / "output_manifest.json").is_file()
    finally:
        shutil.rmtree(str(output), ignore_errors=True)


def test_publish_cli_blocked_conflict_and_uncertain_exit_schemas_are_exact(monkeypatch, capsys):
    for status, code in (("CONFLICT", 3), ("UNCERTAIN", 4)):
        _wire(monkeypatch, _receipt(status))
        assert publish_cli.main(ARGS) == code
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == status and payload["manifest_digest"] is None
    _wire(monkeypatch, error=publish_cli.HistoricalWindowPublicationContractError("/private/detail"))
    assert publish_cli.main(ARGS) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "blocked",
        "error": {
            "type": "HistoricalWindowPublicationContractError",
            "reason_code": "PUBLICATION_CONTRACT_ERROR",
        },
    }
    assert "/private/detail" not in captured.out


def test_publish_cli_fresh_import_preserves_environment_stderr_and_b3a_zero_write_cli(capsys):
    before = dict(os.environ)
    module = importlib.reload(publish_cli)
    assert dict(os.environ) == before and capsys.readouterr().err == ""
    assert "output_dir" in vars(module.build_parser().parse_args(ARGS))
    zero_options = {action.dest for action in build_zero_write_parser()._actions}
    assert "output_dir" not in zero_options
