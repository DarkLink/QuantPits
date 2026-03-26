import os
import sys
import yaml
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    
    config_dir = workspace / "config"
    config_dir.mkdir()
    
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.utils import env
    from quantpits.tools import check_workflow_yaml
    import importlib
    importlib.reload(env)
    importlib.reload(check_workflow_yaml)
    
    monkeypatch.setattr(check_workflow_yaml, 'ROOT_DIR', str(workspace))
    monkeypatch.setattr(check_workflow_yaml, 'CONFIG_DIR', str(config_dir))
    
    yield check_workflow_yaml, workspace

# ── check_yamls ──────────────────────────────────────────────────────────
def test_check_yamls_valid_week(mock_env):
    cw, workspace = mock_env
    valid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -6) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 52
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "week"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(valid_yaml)
    anomalies = cw.check_yamls(freq="week")
    assert anomalies == {}

def test_check_yamls_wrong_label(mock_env):
    cw, workspace = mock_env
    invalid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 52
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "week"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(invalid_yaml)
    anomalies = cw.check_yamls(freq="week")
    assert "workflow_config_m1.yaml" in anomalies
    assert any("LABEL:" in val for val in anomalies["workflow_config_m1.yaml"])

def test_check_yamls_wrong_time_per_step(mock_env):
    cw, workspace = mock_env
    invalid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -6) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 52
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "day"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(invalid_yaml)
    anomalies = cw.check_yamls(freq="week")
    assert any("TIME_PER_STEP:" in val for val in anomalies["workflow_config_m1.yaml"])

def test_check_yamls_wrong_ann_scaler(mock_env):
    cw, workspace = mock_env
    invalid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -6) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 252
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "week"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(invalid_yaml)
    anomalies = cw.check_yamls(freq="week")
    assert any("ANN_SCALER:" in val for val in anomalies["workflow_config_m1.yaml"])

def test_check_yamls_day_freq(mock_env):
    cw, workspace = mock_env
    valid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 252
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "day"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(valid_yaml)
    anomalies = cw.check_yamls(freq="day")
    assert anomalies == {}

# ── fix_yamls ────────────────────────────────────────────────────────────
def test_fix_yamls_label(mock_env):
    cw, workspace = mock_env
    target_yaml = workspace / "config" / "workflow_config_m1.yaml"
    invalid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
"""
    target_yaml.write_text(invalid_yaml)
    cw.fix_yamls(freq="week")
    
    fixed_yaml = target_yaml.read_text()
    assert 'label: ["Ref($close, -6) / Ref($close, -1) - 1"]' in fixed_yaml

def test_fix_yamls_ann_scaler(mock_env):
    cw, workspace = mock_env
    target_yaml = workspace / "config" / "workflow_config_m1.yaml"
    invalid_yaml = """
task:
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 252
"""
    target_yaml.write_text(invalid_yaml)
    cw.fix_yamls(freq="week")
    
    fixed_yaml = target_yaml.read_text()
    assert 'ann_scaler: 52' in fixed_yaml

def test_fix_yamls_lr_scientific(mock_env):
    cw, workspace = mock_env
    target_yaml = workspace / "config" / "workflow_config_m1.yaml"
    invalid_yaml = """
task:
  model:
    kwargs:
      lr: 1e-4
"""
    target_yaml.write_text(invalid_yaml)
    
    # Needs to report it first
    anomalies = cw.check_yamls(freq="week")
    assert any("LR" in val for val in anomalies.get("workflow_config_m1.yaml", []))
    
    cw.fix_yamls(freq="week")
    
    fixed_yaml = target_yaml.read_text()
    assert 'lr: 0.0001' in fixed_yaml

def test_fix_yamls_insert_executor(mock_env):
    cw, workspace = mock_env
    target_yaml = workspace / "config" / "workflow_config_m1.yaml"
    invalid_yaml = """
port_analysis_config:
  backtest:
    kwargs: {}
"""
    target_yaml.write_text(invalid_yaml)
    cw.fix_yamls(freq="week")
    
    fixed_yaml = target_yaml.read_text()
    assert 'executor:' in fixed_yaml
    assert 'time_per_step: "week"' in fixed_yaml

def test_check_yamls_tcts(mock_env):
    cw, workspace = mock_env
    # TCTS should have multiple labels and valid PA/SigAna configs
    tcts_yaml = """
data_handler_config:
    label: ["Ref($close, -6) / Ref($close, -1) - 1", "another"]
task:
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 52
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "week"
"""
    (workspace / "config" / "workflow_config_tcts.yaml").write_text(tcts_yaml)
    anomalies = cw.check_yamls(freq="week")
    assert "workflow_config_tcts.yaml" not in anomalies

def test_fix_yamls_tcts(mock_env):
    cw, workspace = mock_env
    target_yaml = workspace / "config" / "workflow_config_tcts.yaml"
    # Old day freq TCTS
    invalid_yaml = 'label: ["Ref($close, -2) / Ref($close, -1) - 1", "Ref($close, -3) / Ref($close, -1) - 1"]'
    target_yaml.write_text(invalid_yaml)
    cw.fix_yamls(freq="week")
    
    fixed_yaml = target_yaml.read_text()
    assert 'Ref($close, -6)' in fixed_yaml
    assert 'Ref($close, -11)' in fixed_yaml

# ── main ─────────────────────────────────────────────────────────────────

def test_main_no_fix(mock_env, capsys):
    cw, workspace = mock_env
    (workspace / "config" / "model_config.json").write_text('{"freq": "week"}')
    # Valid yaml
    valid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -6) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 52
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "week"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(valid_yaml)
    
    with patch.object(sys, 'argv', ['script.py']):
        cw.main()
    
    captured = capsys.readouterr()
    assert "完美符合 week 频次要求" in captured.out

def test_main_with_fix(mock_env, capsys):
    cw, workspace = mock_env
    (workspace / "config" / "model_config.json").write_text('{"freq": "week"}')
    # Invalid yaml (wrong label)
    invalid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -2) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 52
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "week"
"""
    (workspace / "config" / "workflow_config_m1.yaml").write_text(invalid_yaml)
    
    with patch.object(sys, 'argv', ['script.py', '--fix']):
        cw.main()
    
    captured = capsys.readouterr()
    assert "开始执行自动修复 (week)..." in captured.out
    assert "完美符合 week 频次要求" in captured.out


def test_fix_yamls_day_freq(mock_env):
    cw, workspace = mock_env
    target_yaml = workspace / "config" / "workflow_config_m1.yaml"
    # Week freq yaml
    invalid_yaml = """
task:
  dataset:
    kwargs:
      handler:
        kwargs:
          label: ["Ref($close, -6) / Ref($close, -1) - 1"]
  record:
    - class: SigAnaRecord
      kwargs:
        ann_scaler: 52
port_analysis_config:
  executor:
    kwargs:
      time_per_step: "week"
"""
    target_yaml.write_text(invalid_yaml)
    cw.fix_yamls(freq="day")
    
    fixed_yaml = target_yaml.read_text()
    assert 'label: ["Ref($close, -2) / Ref($close, -1) - 1"]' in fixed_yaml
    assert 'ann_scaler: 252' in fixed_yaml
    assert 'time_per_step: "day"' in fixed_yaml
