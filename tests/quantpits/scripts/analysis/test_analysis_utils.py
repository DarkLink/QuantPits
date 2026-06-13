import os
import sys
import json
import importlib
import pytest
import pandas as pd
import numpy as np
import qlib.workflow
from unittest.mock import patch, MagicMock, mock_open


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

    from quantpits.utils import env
    importlib.reload(env)
    

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

@patch('builtins.open', new_callable=mock_open, read_data='{"models": {"GATs": "fake_record_id"}, "experiment_name": "fake_exp"}')
@patch('quantpits.scripts.analysis.utils.os.path.exists')
def test_load_model_predictions(mock_exists, mock_file, mock_env):
    utils, _, _, _ = mock_env
    # Ensure our mocked path returns True
    mock_exists.return_value = True
    
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-03-01", "2026-03-01"]),
        "instrument": ["SZ000001", "SZ000002"],
        "score": [0.5, 0.3]
    }).set_index(["datetime", "instrument"])
    
    with patch('qlib.workflow.R') as mock_R:
        mock_rec = MagicMock()
        mock_rec.load_object.return_value = df
        mock_R.get_recorder.return_value = mock_rec
        
        result = utils.load_model_predictions("GATs")
        assert not result.empty
        assert "score" in result.columns


def test_load_model_predictions_no_files(mock_env):
    utils, workspace, _, _ = mock_env
    # latest_train_records.json doesn't exist
    result = utils.load_model_predictions("Nonexistent")
    assert result.empty


@patch('builtins.open', new_callable=mock_open, read_data='{"models": {"test_model": "fake_record_id"}, "experiment_name": "fake_exp"}')
@patch('quantpits.scripts.analysis.utils.os.path.exists')
def test_load_model_predictions_rename_columns(mock_exists, mock_file, mock_env):
    utils, _, _, _ = mock_env
    mock_exists.return_value = True
    
    # DataFrame with column '0' instead of 'score'
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-03-01"]),
        "instrument": ["SZ000001"],
        "0": [0.5]
    }).set_index(["datetime", "instrument"])
    
    with patch('qlib.workflow.R') as mock_R:
        mock_rec = MagicMock()
        mock_rec.load_object.return_value = df
        mock_R.get_recorder.return_value = mock_rec
        
        result = utils.load_model_predictions("test_model")
        assert "score" in result.columns


# ── load_model_predictions — additional edge cases ────────────────────────

@patch('builtins.open', new_callable=mock_open, read_data='INVALID JSON {{{')
@patch('quantpits.scripts.analysis.utils.os.path.exists')
def test_load_model_predictions_corrupt_records_json(mock_exists, mock_file, mock_env):
    """Lines 154-155: corrupt latest_train_records.json → exception swallowed."""
    utils, _, _, _ = mock_env
    mock_exists.return_value = True
    result = utils.load_model_predictions("GATs")
    assert result.empty


@patch('builtins.open', new_callable=mock_open, read_data='{"models": {"GATs": "rec_1"}, "experiment_name": "exp"}')
@patch('quantpits.scripts.analysis.utils.os.path.exists')
def test_load_model_predictions_recorder_exception(mock_exists, mock_file, mock_env):
    """Lines 164-166: recorder.load_object raises exception → empty DataFrame."""
    utils, _, _, _ = mock_env
    mock_exists.return_value = True
    with patch('qlib.workflow.R') as mock_R:
        mock_rec = MagicMock()
        mock_rec.load_object.side_effect = Exception("Bad pred.pkl")
        mock_R.get_recorder.return_value = mock_rec
        result = utils.load_model_predictions("GATs")
        assert result.empty


@patch('builtins.open', new_callable=mock_open, read_data='{"models": {"GATs": "rec_1"}, "experiment_name": "exp"}')
@patch('quantpits.scripts.analysis.utils.os.path.exists')
def test_load_model_predictions_series_to_dataframe(mock_exists, mock_file, mock_env):
    """Line 169: prediction is a Series → converted to DataFrame with 'score' column."""
    utils, _, _, _ = mock_env
    mock_exists.return_value = True
    s = pd.Series([0.5, 0.3], name="pred")
    with patch('qlib.workflow.R') as mock_R:
        mock_rec = MagicMock()
        mock_rec.load_object.return_value = s
        mock_R.get_recorder.return_value = mock_rec
        result = utils.load_model_predictions("GATs")
        assert "score" in result.columns


@patch('builtins.open', new_callable=mock_open, read_data='{"models": {"GATs": "rec_1"}, "experiment_name": "exp"}')
@patch('quantpits.scripts.analysis.utils.os.path.exists')
def test_load_model_predictions_datetime_as_column(mock_exists, mock_file, mock_env):
    """Lines 180-181: datetime is a column (not in index) → set_index."""
    utils, _, _, _ = mock_env
    mock_exists.return_value = True
    df = pd.DataFrame({
        "datetime": ["2026-03-01", "2026-03-01"],
        "instrument": ["SZ000001", "SZ000002"],
        "score": [0.5, 0.3],
    })
    with patch('qlib.workflow.R') as mock_R:
        mock_rec = MagicMock()
        mock_rec.load_object.return_value = df
        mock_R.get_recorder.return_value = mock_rec
        result = utils.load_model_predictions("GATs")
        assert not result.empty
        assert "datetime" in result.index.names


@patch('builtins.open', new_callable=mock_open, read_data='{"models": {"GATs": "rec_1"}, "experiment_name": "exp"}')
@patch('quantpits.scripts.analysis.utils.os.path.exists')
def test_load_model_predictions_numeric_column_auto_rename(mock_exists, mock_file, mock_env):
    """Line 176: no '0' column, no 'score' column, but numeric cols exist → rename first."""
    utils, _, _, _ = mock_env
    mock_exists.return_value = True
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-03-01"]),
        "instrument": ["SZ000001"],
        "prediction": [0.5],  # numeric column, not named '0' or 'score'
    }).set_index(["datetime", "instrument"])
    with patch('qlib.workflow.R') as mock_R:
        mock_rec = MagicMock()
        mock_rec.load_object.return_value = df
        mock_R.get_recorder.return_value = mock_rec
        result = utils.load_model_predictions("GATs")
        assert "score" in result.columns


# ── get_daily_features / get_forward_returns with market=None ─────────

def test_get_daily_features_market_default(mock_env):
    """Line 61,63: market=None triggers load_market_config() fallback + default features."""
    utils, _, config_dir, _ = mock_env
    with open(config_dir / "model_config.json", "w") as f:
        json.dump({"market": "csi500", "benchmark": "SH000905"}, f)
    with patch('qlib.data.D') as mock_D:
        mock_instruments = MagicMock()
        mock_D.instruments.return_value = mock_instruments
        # Need 8 columns to match default features dict
        mock_D.features.return_value = pd.DataFrame(
            np.zeros((1, 8)),
            columns=range(8),
        )
        utils.get_daily_features("2026-01-01", "2026-01-05", market=None)
        mock_D.instruments.assert_called_once_with(market="csi500")


def test_get_forward_returns_market_default(mock_env):
    """Line 90: market=None triggers load_market_config() fallback."""
    utils, _, config_dir, _ = mock_env
    with open(config_dir / "model_config.json", "w") as f:
        json.dump({"market": "csi500", "benchmark": "SH000905"}, f)
    with patch('qlib.data.D') as mock_D:
        mock_instruments = MagicMock()
        mock_D.instruments.return_value = mock_instruments
        mock_D.features.return_value = pd.DataFrame({"return_1d": [0.01]})
        utils.get_forward_returns("2026-01-01", "2026-01-05", market=None)
        mock_D.instruments.assert_called_once_with(market="csi500")
