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
