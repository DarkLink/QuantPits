import os
import sys
import json
import importlib
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    config_dir = workspace / "config"
    config_dir.mkdir()
    data_dir = workspace / "data"
    data_dir.mkdir()

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

    from quantpits.scripts.analysis import utils
    importlib.reload(utils)

    monkeypatch.setattr(utils, 'ROOT_DIR', str(workspace))
    monkeypatch.setattr(utils, 'CONFIG_FILE', str(config_dir / "prod_config.json"))
    monkeypatch.setattr(utils, 'MODEL_CONFIG_FILE', str(config_dir / "model_config.json"))
    monkeypatch.setattr(utils, 'PREDICTION_DIR', str(workspace / "output" / "predictions"))

    yield utils, workspace, config_dir, data_dir


# ── load_market_config ───────────────────────────────────────────────────

def test_load_market_config(mock_env):
    utils, workspace, config_dir, _ = mock_env
    model_config = {"market": "csi500", "benchmark": "SH000905"}
    with open(config_dir / "model_config.json", "w") as f:
        json.dump(model_config, f)

    market, benchmark = utils.load_market_config()
    assert market == "csi500"
    assert benchmark == "SH000905"


def test_load_market_config_missing(mock_env):
    utils, _, _, _ = mock_env
    market, benchmark = utils.load_market_config()
    assert market == "csi300"  # DEFAULT_MARKET
    assert benchmark == "SH000300"  # DEFAULT_BENCHMARK


def test_load_market_config_bad_json(mock_env):
    utils, _, config_dir, _ = mock_env
    with open(config_dir / "model_config.json", "w") as f:
        f.write("not valid json{{{")

    market, benchmark = utils.load_market_config()
    assert market == "csi300"  # Should fallback gracefully


# ── load_trade_log ───────────────────────────────────────────────────────

def test_load_trade_log(mock_env):
    utils, workspace, _, data_dir = mock_env
    df = pd.DataFrame({
        "成交日期": ["2026-03-01", "2026-03-02"],
        "证券代码": ["000001", "600000"],
        "成交金额": [1000.0, 2000.0]
    })
    df.to_csv(data_dir / "trade_log_full.csv", index=False)

    result = utils.load_trade_log()
    assert len(result) == 2
    assert pd.api.types.is_datetime64_any_dtype(result["成交日期"])


def test_load_trade_log_missing(mock_env, capsys):
    utils, _, _, _ = mock_env
    result = utils.load_trade_log()
    assert result.empty
    captured = capsys.readouterr()
    assert "Warning" in captured.out


# ── load_daily_amount ────────────────────────────────────────────────────

def test_load_daily_amount(mock_env):
    utils, workspace, _, data_dir = mock_env
    df = pd.DataFrame({
        "成交日期": ["2026-03-01"],
        "收盘价值": [1000000.0],
    })
    df.to_csv(data_dir / "daily_amount_log_full.csv", index=False)

    result = utils.load_daily_amount()
    assert len(result) == 1


def test_load_daily_amount_missing(mock_env, capsys):
    utils, _, _, _ = mock_env
    result = utils.load_daily_amount()
    assert result.empty


# ── load_holding_log ─────────────────────────────────────────────────────

def test_load_holding_log(mock_env):
    utils, workspace, _, data_dir = mock_env
    df = pd.DataFrame({
        "成交日期": ["2026-03-01"],
        "证券代码": ["SZ000001"],
        "持仓数量": [100]
    })
    df.to_csv(data_dir / "holding_log_full.csv", index=False)

    result = utils.load_holding_log()
    assert len(result) == 1


# ── load_model_predictions ───────────────────────────────────────────────

def test_load_model_predictions(mock_env):
    utils, workspace, _, _ = mock_env
    pred_dir = workspace / "output" / "predictions"
    pred_dir.mkdir(parents=True)

    df = pd.DataFrame({
        "datetime": ["2026-03-01", "2026-03-01"],
        "instrument": ["SZ000001", "SZ000002"],
        "score": [0.5, 0.3]
    })
    df.to_csv(pred_dir / "GATs_2026-03-01.csv", index=False)

    result = utils.load_model_predictions("GATs")
    assert not result.empty
    assert "score" in result.columns


def test_load_model_predictions_no_files(mock_env):
    utils, workspace, _, _ = mock_env
    pred_dir = workspace / "output" / "predictions"
    pred_dir.mkdir(parents=True)

    result = utils.load_model_predictions("Nonexistent")
    assert result.empty


def test_load_model_predictions_rename_columns(mock_env):
    utils, workspace, _, _ = mock_env
    pred_dir = workspace / "output" / "predictions"
    pred_dir.mkdir(parents=True)

    # CSV with column '0' instead of 'score'
    df = pd.DataFrame({
        "datetime": ["2026-03-01"],
        "instrument": ["SZ000001"],
        "0": [0.5]
    })
    df.to_csv(pred_dir / "test_model_2026-03-01.csv", index=False)

    result = utils.load_model_predictions("test_model")
    assert "score" in result.columns
