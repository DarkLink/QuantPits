import os
import sys
import importlib
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    (workspace / "config").mkdir()
    data_dir = workspace / "data"
    data_dir.mkdir()
    (data_dir / "order_history").mkdir()
    output_dir = workspace / "output"
    output_dir.mkdir()

    scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'quantpits', 'scripts')
    scripts_dir = os.path.abspath(scripts_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    from quantpits.scripts import env
    importlib.reload(env)
    import env as bare_env
    importlib.reload(bare_env)

    from quantpits.scripts.analysis import trade_classifier as tc
    importlib.reload(tc)

    monkeypatch.setattr(tc, 'ROOT_DIR', str(workspace))

    yield tc, workspace, data_dir


# ── _add_prefix ──────────────────────────────────────────────────────────

def test_add_prefix(mock_env):
    tc, _, _ = mock_env
    assert tc._add_prefix("600000") == "SH600000"
    assert tc._add_prefix("000001") == "SZ000001"
    assert tc._add_prefix("300123") == "SZ300123"
    assert tc._add_prefix("800000") == "800000"  # Unknown prefix


# ── _normalize_instrument ────────────────────────────────────────────────

def test_normalize_instrument(mock_env):
    tc, _, _ = mock_env
    assert tc._normalize_instrument("SH601066") == "SH601066"
    assert tc._normalize_instrument("601066") == "SH601066"
    assert tc._normalize_instrument("000001") == "SZ000001"
    assert tc._normalize_instrument("SZ000001") == "SZ000001"


# ── save_classification / load_classification ────────────────────────────

def test_save_and_load_classification(mock_env):
    tc, workspace, data_dir = mock_env
    out_path = str(data_dir / "trade_classification.csv")

    df = pd.DataFrame({
        "trade_date": ["2026-03-01", "2026-03-02"],
        "instrument": ["SZ000001", "SH600000"],
        "trade_type": ["BUY", "SELL"],
        "trade_class": ["S", "M"],
        "suggestion_date": ["2026-02-28", None],
        "suggestion_rank": [1, None]
    })

    tc.save_classification(df, path=out_path)
    assert os.path.exists(out_path)

    loaded = tc.load_classification(path=out_path)
    assert len(loaded) == 2
    assert "trade_class" in loaded.columns


def test_load_classification_missing(mock_env):
    tc, _, data_dir = mock_env
    result = tc.load_classification(path=str(data_dir / "nonexistent.csv"))
    assert result.empty


# ── _print_summary ───────────────────────────────────────────────────────

def test_print_summary(mock_env, capsys):
    tc, _, _ = mock_env
    df = pd.DataFrame({
        "trade_date": ["2026-03-01", "2026-03-01", "2026-03-02"],
        "instrument": ["SZ000001", "SH600000", "SZ000002"],
        "trade_type": ["BUY", "SELL", "BUY"],
        "trade_class": ["S", "M", "D"],
        "suggestion_date": ["2026-02-28", None, None],
        "suggestion_rank": [1, None, None]
    })
    tc._print_summary(df)
    captured = capsys.readouterr()
    assert "SIGNAL" in captured.out or "S" in captured.out
