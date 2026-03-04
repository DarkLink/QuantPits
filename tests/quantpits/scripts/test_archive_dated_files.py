import os
import sys
import json
import pytest
import importlib
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    for d in ["output", "data", "archive", "config"]:
        (workspace / d).mkdir()
    (workspace / "data" / "order_history").mkdir()

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    # Ensure scripts dir is in sys.path for bare `import env`
    scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'quantpits', 'scripts')
    scripts_dir = os.path.abspath(scripts_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from quantpits.scripts import env
    importlib.reload(env)

    from quantpits.scripts import archive_dated_files as adf
    importlib.reload(adf)

    # Patch module-level constants to use tmp_path
    monkeypatch.setattr(adf, 'ROOT_DIR', str(workspace))
    monkeypatch.setattr(adf, 'OUTPUT_DIR', str(workspace / "output"))
    monkeypatch.setattr(adf, 'DATA_DIR', str(workspace / "data"))
    monkeypatch.setattr(adf, 'ARCHIVE_DIR', str(workspace / "archive"))
    monkeypatch.setattr(adf, 'ORDER_HISTORY_DIR', str(workspace / "data" / "order_history"))

    yield adf, workspace


# ── extract_date_info ────────────────────────────────────────────────────

def test_extract_date_info_suffix_format(mock_env):
    adf, _ = mock_env
    result = adf.extract_date_info("model_performance_2026-03-01.json")
    assert result is not None
    logical_name, date_str, sort_key = result
    assert date_str == "2026-03-01"
    assert "2026-03-01" in sort_key


def test_extract_date_info_suffix_with_timestamp(mock_env):
    adf, _ = mock_env
    result = adf.extract_date_info("predictions_2026-02-28_143022.csv")
    assert result is not None
    logical_name, date_str, sort_key = result
    assert date_str == "2026-02-28"
    assert sort_key == "2026-02-28_143022"


def test_extract_date_info_prefix_format(mock_env):
    adf, _ = mock_env
    result = adf.extract_date_info("2026-03-01-table.xlsx")
    assert result is not None
    logical_name, date_str, sort_key = result
    assert date_str == "2026-03-01"
    assert "*-" in logical_name


def test_extract_date_info_no_date(mock_env):
    adf, _ = mock_env
    assert adf.extract_date_info("readme.md") is None
    assert adf.extract_date_info("config.json") is None
    assert adf.extract_date_info("") is None


# ── is_protected ─────────────────────────────────────────────────────────

def test_is_protected(mock_env):
    adf, _ = mock_env
    assert adf.is_protected("trade_log_full.csv") is True
    assert adf.is_protected("holding_log_full.csv") is True
    assert adf.is_protected("run_state.json") is True
    assert adf.is_protected("model_log.csv") is True
    assert adf.is_protected("some_other_file.csv") is False


# ── is_trade_data ────────────────────────────────────────────────────────

def test_is_trade_data(mock_env):
    adf, _ = mock_env
    assert adf.is_trade_data("buy_suggestion_2026-03-01.csv") is True
    assert adf.is_trade_data("sell_suggestion_2026-03-01.csv") is True
    assert adf.is_trade_data("model_opinions_2026-03-01.csv") is True
    assert adf.is_trade_data("trade_detail_2026-03-01.csv") is True
    assert adf.is_trade_data("2026-03-01-table.xlsx") is True
    assert adf.is_trade_data("random_report.csv") is False


# ── scan_dated_files ─────────────────────────────────────────────────────

def test_scan_dated_files(mock_env):
    adf, workspace = mock_env
    output_dir = workspace / "output"
    # Create some dated files
    (output_dir / "predictions_2026-03-01.csv").write_text("data")
    (output_dir / "predictions_2026-02-28.csv").write_text("data")
    (output_dir / "readme.md").write_text("not dated")
    (output_dir / "trade_log_full.csv").write_text("protected")

    groups = adf.scan_dated_files(str(output_dir), "output")
    # Should find 2 dated files, skip readme and protected
    total_files = sum(len(v) for v in groups.values())
    assert total_files == 2


def test_scan_dated_files_empty_dir(mock_env):
    adf, workspace = mock_env
    empty_dir = workspace / "empty"
    empty_dir.mkdir()
    groups = adf.scan_dated_files(str(empty_dir))
    assert len(groups) == 0


def test_scan_dated_files_nonexistent_dir(mock_env):
    adf, workspace = mock_env
    groups = adf.scan_dated_files(str(workspace / "nonexistent"))
    assert len(groups) == 0


# ── get_anchor_date ──────────────────────────────────────────────────────

def test_get_anchor_date_override(mock_env):
    adf, _ = mock_env
    assert adf.get_anchor_date(override="2026-01-15") == "2026-01-15"


def test_get_anchor_date_from_records(mock_env):
    adf, workspace = mock_env
    records = {"anchor_date": "2026-02-28", "models": {}}
    with open(workspace / "latest_train_records.json", "w") as f:
        json.dump(records, f)

    assert adf.get_anchor_date() == "2026-02-28"


def test_get_anchor_date_no_source(mock_env):
    adf, _ = mock_env
    with pytest.raises(ValueError, match="无法确定锚点日期"):
        adf.get_anchor_date()


# ── plan_archive ─────────────────────────────────────────────────────────

def test_plan_archive(mock_env):
    adf, workspace = mock_env
    output_dir = workspace / "output"
    # Old file (should be archived)
    (output_dir / "predictions_2026-02-01.csv").write_text("old")
    # New file (should be kept)
    (output_dir / "predictions_2026-03-01.csv").write_text("new")
    # Trade data (should go to order_history)
    (output_dir / "buy_suggestion_2026-02-01.csv").write_text("old trade")

    moves = adf.plan_archive("2026-03-01")
    assert len(moves) == 2  # old predictions + old trade data

    categories = {cat for _, _, cat in moves}
    assert "trade_data" in categories
    assert "output" in categories


# ── execute_moves ────────────────────────────────────────────────────────

def test_execute_moves_dry_run(mock_env, capsys):
    adf, workspace = mock_env
    src = workspace / "output" / "test_file.csv"
    src.write_text("content")
    dest = workspace / "archive" / "output" / "test_file.csv"

    moves = [(str(src), str(dest), "output")]
    total = adf.execute_moves(moves, dry_run=True)
    assert total == 1
    # File should still exist at source
    assert src.exists()
    assert not dest.exists()


def test_execute_moves_real(mock_env):
    adf, workspace = mock_env
    src = workspace / "output" / "test_file.csv"
    src.write_text("content")
    dest = workspace / "archive" / "output" / "test_file.csv"

    moves = [(str(src), str(dest), "output")]
    total = adf.execute_moves(moves, dry_run=False)
    assert total == 1
    assert not src.exists()
    assert dest.exists()


def test_execute_moves_empty(mock_env, capsys):
    adf, _ = mock_env
    total = adf.execute_moves([], dry_run=False)
    assert total == 0
    captured = capsys.readouterr()
    assert "没有需要归档的文件" in captured.out


# ── print_summary ────────────────────────────────────────────────────────

def test_print_summary(mock_env, capsys):
    adf, workspace = mock_env
    moves = [
        ("/src1", "/dest1", "output"),
        ("/src2", "/dest2", "trade_data"),
        ("/src3", "/dest3", "output"),
    ]
    adf.print_summary(moves, dry_run=True)
    captured = capsys.readouterr()
    assert "总计: 3" in captured.out
    assert "DRY-RUN" in captured.out
