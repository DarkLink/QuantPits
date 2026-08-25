from __future__ import annotations

import json
import os
import sys

import pytest

from quantpits.scripts.research_shadow_window import build_parser, main
from tests.quantpits.research.test_historical_cycle import DATES
from tests.quantpits.research.test_historical_window import build_window


def _argv(workspace, qlib, profile):
    return [
        "--workspace", str(workspace), "--qlib-data-dir", str(qlib),
        "--profile", str(profile), "--sealed-cycle", DATES[-1],
        "--preferred-start", DATES[0], "--preferred-end", DATES[-1],
        "--window-size", "4", "--top-k", "2",
    ]


def test_cli_requires_exact_read_only_arguments_and_rejects_output_or_selection_options():
    parser = build_parser()
    args = parser.parse_args(_argv("/workspace", "/qlib", "/profile.json"))
    assert args.window_size == 4
    assert set(vars(args)) == {
        "workspace", "qlib_data_dir", "profile", "sealed_cycle",
        "preferred_start", "preferred_end", "window_size", "top_k",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(_argv("/workspace", "/qlib", "/profile.json") + ["--output-dir", "/tmp/x"])
    with pytest.raises(SystemExit):
        parser.parse_args(_argv("/workspace", "/qlib", "/profile.json") + ["--best-arm", "DROP_1_3"])


def test_cli_compact_stdout_matches_normalized_privacy_allow_list(
    tmp_path, capsys, monkeypatch,
):
    _runner, _stage, _inputs, profile, workspace, qlib = build_window(tmp_path, 4)
    monkeypatch.delitem(sys.modules, "quantpits.utils.strategy", raising=False)
    monkeypatch.delitem(sys.modules, "quantpits.utils.env", raising=False)
    environment_before = dict(os.environ)
    assert main(_argv(workspace, qlib, profile._private_path)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    assert dict(os.environ) == environment_before
    payload = json.loads(captured.out)
    assert set(payload) == {
        "status", "evidence_class", "warnings", "prospective_claim",
        "promotion_capability", "window_digest", "requested_cycle_count",
        "terminal_cycle_count", "complete_cycle_arm_count", "blocked_cycle_arm_count",
        "chain_continuity_checked", "metric_capability", "arms",
    }
    rendered = json.dumps(payload).lower()
    assert "cash" not in rendered and "position" not in rendered
    assert str(profile._private_path) not in rendered


def test_cli_runs_earlier_window_with_sealed_parity_anchor_outside_range(tmp_path, capsys):
    _runner, _stage, _inputs, profile, workspace, qlib = build_window(tmp_path, 4)
    argv = _argv(workspace, qlib, profile._private_path)
    argv[argv.index("--preferred-end") + 1] = DATES[-2]

    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert payload["requested_cycle_count"] == 4


def test_cli_blocked_exception_emits_only_type_and_reason_code(tmp_path, capsys):
    missing = tmp_path / "private-secret-profile.json"
    assert main(_argv(tmp_path, tmp_path, missing)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert set(payload["error"]) == {"type", "reason_code"}
    assert payload["error"]["reason_code"] == "READ_ERROR"
    assert str(missing) not in json.dumps(payload)
