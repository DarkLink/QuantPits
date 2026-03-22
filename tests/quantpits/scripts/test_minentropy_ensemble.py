import pytest
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from quantpits.scripts import minentropy_ensemble
from quantpits.utils import strategy

@pytest.fixture
def mock_is_norm_df():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    tuples = [(d, "A") for d in dates]
    idx = pd.MultiIndex.from_tuples(tuples, names=["datetime", "instrument"])
    df = pd.DataFrame({
        "m1": [0.5, 0.6, 0.7], 
        "m2": [0.1, 0.2, 0.3],
        "m3": [0.9, 0.8, 0.7]
    }, index=idx)
    return df

def test_minentropy_backtest_logic(mock_is_norm_df, tmp_path):
    # Ensure index names are correct
    mock_is_norm_df.index.names = ["datetime", "instrument"]
    
    with patch('quantpits.utils.strategy.load_strategy_config', return_value={}):
        with patch('quantpits.utils.strategy.get_backtest_config', return_value={"account": 100, "exchange_kwargs": {"freq": "day"}}):
            with patch('quantpits.scripts.minentropy_ensemble.Exchange'):
                def mock_run_single(models, *args, **kwargs):
                    # Distinguish between single models and combos
                    if len(models) == 1:
                        mname = models[0]
                    else:
                        mname = ",".join(models)
                    return {
                        "models": mname,
                        "n_models": len(models),
                        "Ann_Excess": 0.1,
                        "Ann_Ret": 0.05,
                        "Max_DD": -0.1,
                        "Total_Ret": 1.2,
                        "Final_NAV": 1.2,
                        "Calmar": 1.0
                    }
                
                with patch('quantpits.scripts.minentropy_ensemble.run_single_backtest', side_effect=mock_run_single):
                    # Do NOT patch _append_results_to_csv, as brute_force_backtest needs it
                    res_df = minentropy_ensemble.minentropy_backtest(
                        mock_is_norm_df.copy(), 22, 3, "SH000300", "day",
                        2, str(tmp_path), "2020-01-01"
                    )
                    assert not res_df.empty

def test_minentropy_backtest_empty():
    empty_df = pd.DataFrame()
    res = minentropy_ensemble.minentropy_backtest(
        empty_df, 1, 0, "B", "day", 2, "dir", "date"
    )
    assert res.empty

def test_main_flow(mock_is_norm_df, tmp_path):
    mock_is_norm_df.index.names = ["datetime", "instrument"]
    empty_oos = mock_is_norm_df.iloc[:0].copy()
    
    with patch('quantpits.scripts.minentropy_ensemble.init_qlib'):
        with patch('quantpits.scripts.minentropy_ensemble.load_config', return_value=({}, {})):
            with patch('quantpits.scripts.minentropy_ensemble.load_predictions', return_value=(mock_is_norm_df.copy(), None)):
                with patch('quantpits.scripts.minentropy_ensemble.split_is_oos_by_args', return_value=(mock_is_norm_df.copy(), empty_oos)):
                    with patch('quantpits.scripts.minentropy_ensemble.correlation_analysis'):
                        with patch('quantpits.scripts.minentropy_ensemble.minentropy_backtest', return_value=pd.DataFrame({"Ann_Excess": [0.1]})):
                            with patch('quantpits.scripts.minentropy_ensemble.env.safeguard'):
                                import sys
                                with patch.object(sys, 'argv', ['script.py', '--output-dir', str(tmp_path)]):
                                    with patch('builtins.open', create=True):
                                        minentropy_ensemble.main()

def test_main_training_mode_filter(mock_is_norm_df, tmp_path):
    mock_is_norm_df.index.names = ["datetime", "instrument"]
    train_records = {"models": {"m1": {}, "m2": {}}, "anchor_date": "2020"}
    empty_oos = mock_is_norm_df.iloc[:0].copy()
    
    with patch('quantpits.scripts.minentropy_ensemble.init_qlib'):
        with patch('quantpits.scripts.minentropy_ensemble.load_config', return_value=(train_records, {})):
            with patch('quantpits.utils.train_utils.filter_models_by_mode', return_value={"m1": {}}) as mock_filter:
                with patch('quantpits.scripts.minentropy_ensemble.load_predictions', return_value=(mock_is_norm_df.copy(), None)):
                    with patch('quantpits.scripts.minentropy_ensemble.split_is_oos_by_args', return_value=(mock_is_norm_df.copy(), empty_oos)):
                        with patch('quantpits.scripts.minentropy_ensemble.minentropy_backtest'):
                            with patch('quantpits.scripts.minentropy_ensemble.env.safeguard'):
                                import sys
                                with patch.object(sys, 'argv', ['script.py', '--training-mode', 'static']):
                                    minentropy_ensemble.main()
                                    assert mock_filter.called
