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

    from quantpits.utils import env
    importlib.reload(env)

    from quantpits.tools import archive_dated_files as adf
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

def test_archive_notebooks_output_dir(mock_env):
    adf, workspace = mock_env
    nb_dir = workspace / "notebooks"
    nb_dir.mkdir()
    out_dir = nb_dir / "output"
    out_dir.mkdir()
    
    # Empty dir should be removed in real run
    adf.archive_legacy_notebooks(dry_run=False)
    assert not out_dir.exists()

def test_archive_notebooks_output_dir_dry_run(mock_env, capsys):
    adf, workspace = mock_env
    nb_dir = workspace / "notebooks"
    nb_dir.mkdir()
    out_dir = nb_dir / "output"
    out_dir.mkdir()
    
    adf.archive_legacy_notebooks(dry_run=True)
    assert out_dir.exists()
    captured = capsys.readouterr()
    assert "[DRY-RUN] 将删除空目录" in captured.out

# ── legacy items ─────────────────────────────────────────────────────────

def test_archive_legacy_notebooks(mock_env, monkeypatch):
    adf, workspace = mock_env
    nb_dir = workspace / "notebooks"
    nb_dir.mkdir()
    (nb_dir / "old.ipynb").write_text("{}")
    
    monkeypatch.setattr(adf, 'LEGACY_NOTEBOOKS', ["old.ipynb"])
    
    moves = adf.archive_legacy_notebooks(dry_run=True)
    assert len(moves) == 1
    assert "old.ipynb" in moves[0][1]

def test_archive_legacy_items(mock_env, monkeypatch):
    adf, workspace = mock_env
    (workspace / "legacy.py").write_text("old code")
    
    monkeypatch.setattr(adf, 'LEGACY_ITEMS', {"legacy.py": "old/legacy.py"})
    
    moves = adf.archive_legacy_items(dry_run=True)
    assert len(moves) == 1
    assert "old/legacy.py" in moves[0][1]

# ── execute_moves directory ──────────────────────────────────────────────

def test_execute_moves_directory(mock_env):
    adf, workspace = mock_env
    src_dir = workspace / "output" / "subdir"
    src_dir.mkdir()
    (src_dir / "file.txt").write_text("ok")
    
    dest_dir = workspace / "archive" / "output" / "subdir"
    
    moves = [(str(src_dir), str(dest_dir), "output")]
    total = adf.execute_moves(moves, dry_run=False)
    assert total == 1
    assert not src_dir.exists()
    assert dest_dir.exists()
    assert (dest_dir / "file.txt").exists()

# ── main ─────────────────────────────────────────────────────────────────

def test_main_dry_run(mock_env, capsys):
    adf, workspace = mock_env
    # Create an old file
    (workspace / "output" / "pred_2026-01-01.csv").write_text("old")
    # And records
    records = {"anchor_date": "2026-03-01"}
    with open(workspace / "latest_train_records.json", "w") as f:
        json.dump(records, f)
        
    with patch.object(sys, 'argv', ['script.py', '--dry-run']):
        adf.main()
    
    captured = capsys.readouterr()
    assert "DRY-RUN" in captured.out
    assert "pred_2026-01-01.csv" in captured.out

def test_main_real_all(mock_env, monkeypatch):
    adf, workspace = mock_env
    # Dated file
    (workspace / "output" / "pred_2026-01-01.csv").write_text("old")
    # Legacy nb
    nb_dir = workspace / "notebooks"
    nb_dir.mkdir()
    (nb_dir / "old.ipynb").write_text("{}")
    monkeypatch.setattr(adf, 'LEGACY_NOTEBOOKS', ["old.ipynb"])
    # Legacy item
    (workspace / "legacy.py").write_text("old")
    monkeypatch.setattr(adf, 'LEGACY_ITEMS', {"legacy.py": "old/legacy.py"})
    
    with patch.object(sys, 'argv', ['script.py', '--all', '--anchor-date', '2026-03-01']):
        adf.main()
        
    assert (workspace / "archive" / "output" / "pred_2026-01-01.csv").exists()
    assert (workspace / "archive" / "notebooks" / "old.ipynb").exists()
    assert (workspace / "archive" / "old" / "legacy.py").exists()

def test_main_skip_trade_data(mock_env):
    adf, workspace = mock_env
    (workspace / "output" / "buy_suggestion_2026-01-01.csv").write_text("trade")
    
    with patch.object(sys, 'argv', ['script.py', '--skip-trade-data', '--anchor-date', '2026-03-01']):
        adf.main()
    
    assert (workspace / "output" / "buy_suggestion_2026-01-01.csv").exists()
