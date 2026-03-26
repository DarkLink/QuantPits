import os
import json
import yaml
import pytest
from pathlib import Path
from quantpits.utils.config_loader import load_workspace_config, load_rolling_config

@pytest.fixture
def mock_workspace(tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    config_dir = workspace / "config"
    config_dir.mkdir()
    return workspace, config_dir

def test_load_workspace_config_full(mock_workspace):
    workspace, config_dir = mock_workspace
    
    # 1. model_config.json
    with open(config_dir / "model_config.json", "w") as f:
        json.dump({"market": "csi300", "TopK": 50}, f)
        
    # 2. strategy_config.yaml
    strat_yaml = {
        "strategy": {"params": {"topk": 20, "n_drop": 3}},
        "backtest": {"account": 1000000}
    }
    with open(config_dir / "strategy_config.yaml", "w") as f:
        yaml.dump(strat_yaml, f)
        
    # 3. prod_config.json
    with open(config_dir / "prod_config.json", "w") as f:
        json.dump({"current_cash": 500000, "market": "wrong_market"}, f)
        
    config = load_workspace_config(workspace)
    
    # Assertions
    assert config["market"] == "csi300"  # From model_config
    assert config["topk"] == 20          # From strategy_config
    assert config["TopK"] == 20          # Promoted/Override
    assert config["current_cash"] == 500000 # From prod_config
    assert config["backtest"]["account"] == 1000000

def test_load_rolling_config_formats(mock_workspace):
    workspace, config_dir = mock_workspace
    
    # Test 3M
    with open(config_dir / "rolling_config.yaml", "w") as f:
        f.write("rolling_start: 2020-01-01\ntrain_years: 5\nvalid_years: 1\ntest_step: 3M")
    
    cfg = load_rolling_config(workspace)
    assert cfg["test_step_months"] == 3
    
    # Test 1Y
    with open(config_dir / "rolling_config.yaml", "w") as f:
        f.write("rolling_start: 2020-01-01\ntrain_years: 5\nvalid_years: 1\ntest_step: 1Y")
        
    cfg = load_rolling_config(workspace)
    assert cfg["test_step_months"] == 12
    
    # Test Empty
    with open(config_dir / "rolling_config.yaml", "w") as f:
        f.write("")
    assert load_rolling_config(workspace) is None

def test_load_rolling_config_invalid(mock_workspace):
    workspace, config_dir = mock_workspace
    
    with open(config_dir / "rolling_config.yaml", "w") as f:
        f.write("test_step: 10D")
        
    with pytest.raises(ValueError, match="Invalid test_step format"):
        load_rolling_config(workspace)

def test_load_rolling_config_missing(tmp_path):
    assert load_rolling_config(tmp_path) is None
