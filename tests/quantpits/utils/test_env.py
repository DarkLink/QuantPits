import os
import sys
import importlib
import pytest
import time

def test_env_workspace_arg(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace_Arg"
    workspace.mkdir()
    
    original_argv = sys.argv[:]
    sys.argv.clear()
    sys.argv.extend(['script.py', '--workspace', str(workspace), '--other-arg'])
    monkeypatch.delenv("QLIB_WORKSPACE_DIR", raising=False)
    
    from quantpits.utils import env
    importlib.reload(env)
    
    assert env.ROOT_DIR == str(workspace)
    assert os.environ["QLIB_WORKSPACE_DIR"] == str(workspace)
    assert sys.argv == ['script.py', '--other-arg']
    
    sys.argv.clear()
    sys.argv.extend(original_argv)

def test_env_workspace_env(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace_Env"
    workspace.mkdir()
    
    original_argv = sys.argv[:]
    sys.argv.clear()
    sys.argv.extend(['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.utils import env
    importlib.reload(env)
    
    assert env.ROOT_DIR == str(workspace)
    assert env._workspace_arg is None
    assert "mlruns" in os.environ["MLFLOW_TRACKING_URI"]
    
    sys.argv.clear()
    sys.argv.extend(original_argv)

def test_env_no_workspace(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.delenv("QLIB_WORKSPACE_DIR", raising=False)
    
    with pytest.raises(RuntimeError, match="Please source a workspace run_env.sh first!"):
        from quantpits.utils import env
        importlib.reload(env)

def test_init_qlib(monkeypatch, tmp_path):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.utils import env
    
    class MockQlib:
        def __init__(self):
            self.provider_uri = None
            self.region = None
            
        def init(self, provider_uri, region):
            self.provider_uri = provider_uri
            self.region = region
    
    mock_qlib = MockQlib()
    monkeypatch.setitem(sys.modules, "qlib", mock_qlib)
    
    class MockConstant:
        REG_CN = "cn_constant"
        REG_US = "us_constant"
        
    monkeypatch.setitem(sys.modules, "qlib.constant", MockConstant())
    
    monkeypatch.setenv("QLIB_DATA_DIR", "/mock/qlib/data")
    monkeypatch.setenv("QLIB_REGION", "cn")
    
    importlib.reload(env)
    env.init_qlib()
    
    assert mock_qlib.provider_uri == "/mock/qlib/data"
    assert mock_qlib.region == MockConstant.REG_CN

# ── set_root_dir() tests ──────────────────────────────────────

def test_set_root_dir_updates_env(monkeypatch, tmp_path):
    """set_root_dir() should update env.ROOT_DIR and os.environ."""
    ws1 = tmp_path / "Workspace1"
    ws1.mkdir()
    ws2 = tmp_path / "Workspace2"
    ws2.mkdir()
    (ws2 / "mlruns").mkdir()

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(ws1))

    from quantpits.utils import env
    importlib.reload(env)

    assert env.ROOT_DIR == str(ws1)

    env.set_root_dir(str(ws2))

    assert env.ROOT_DIR == str(ws2)
    assert os.environ["QLIB_WORKSPACE_DIR"] == str(ws2)
    assert "Workspace2" in os.environ["MLFLOW_TRACKING_URI"]


def test_set_root_dir_patches_train_utils(monkeypatch, tmp_path):
    """set_root_dir() should update all train_utils module-level path constants."""
    ws1 = tmp_path / "Workspace1"
    ws1.mkdir()
    ws2 = tmp_path / "Workspace2"
    ws2.mkdir()
    (ws2 / "mlruns").mkdir()

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(ws1))

    from quantpits.utils import env
    importlib.reload(env)

    # Force train_utils into sys.modules so set_root_dir patches it
    from quantpits.utils import train_utils as tu
    original_root = tu.ROOT_DIR

    env.set_root_dir(str(ws2))

    # All train_utils path constants should now point to ws2
    assert tu.ROOT_DIR == str(ws2)
    assert str(ws2) in tu.REGISTRY_FILE
    assert str(ws2) in tu.MODEL_CONFIG_FILE
    assert str(ws2) in tu.PROD_CONFIG_FILE
    assert str(ws2) in tu.RECORD_OUTPUT_FILE
    assert str(ws2) in tu.PREDICTION_OUTPUT_DIR
    assert str(ws2) in tu.ROLLING_PREDICTION_DIR
    assert str(ws2) in tu.HISTORY_DIR
    assert str(ws2) in tu.RUN_STATE_FILE
    assert str(ws2) in tu.ROLLING_STATE_FILE
    assert str(ws2) in tu.LEGACY_ROLLING_RECORD_FILE
    assert str(ws2) in tu.PRETRAINED_DIR


def test_set_root_dir_without_train_utils(monkeypatch, tmp_path):
    """set_root_dir() should not crash when train_utils is not in sys.modules."""
    ws1 = tmp_path / "Workspace1"
    ws1.mkdir()
    ws2 = tmp_path / "Workspace2"
    ws2.mkdir()

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(ws1))

    from quantpits.utils import env
    importlib.reload(env)

    # Ensure train_utils is NOT in sys.modules
    monkeypatch.delitem(sys.modules, "quantpits.utils.train_utils", raising=False)

    # Should not raise
    env.set_root_dir(str(ws2))
    assert env.ROOT_DIR == str(ws2)


def test_set_root_dir_roundtrip(monkeypatch, tmp_path):
    """set_root_dir() should support switching back and forth between workspaces."""
    ws1 = tmp_path / "Workspace1"
    ws1.mkdir()
    ws2 = tmp_path / "Workspace2"
    ws2.mkdir()

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(ws1))

    from quantpits.utils import env
    importlib.reload(env)

    assert env.ROOT_DIR == str(ws1)

    env.set_root_dir(str(ws2))
    assert env.ROOT_DIR == str(ws2)

    env.set_root_dir(str(ws1))
    assert env.ROOT_DIR == str(ws1)


def test_safeguard(monkeypatch, tmp_path, capsys):
    workspace = tmp_path / "MockWorkspace"
    workspace.mkdir()
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    from quantpits.utils import env
    importlib.reload(env)
    
    # Mock time.sleep to avoid waiting 3 seconds during tests
    monkeypatch.setattr(time, 'sleep', lambda x: None)
    
    env.safeguard("TestScript")
    
    captured = capsys.readouterr()
    assert "SAFEGUARD ACTIVATED" in captured.out
    assert "TestScript" in captured.out
    assert "MockWorkspace" in captured.out
