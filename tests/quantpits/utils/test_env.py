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
    # Backend is sqlite:// (new clean workspace) or file:// (legacy data detected)
    assert "MLFLOW_TRACKING_URI" in os.environ
    uri = os.environ["MLFLOW_TRACKING_URI"]
    assert "mlruns" in uri or "mlflow.db" in uri
    
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
            self.exp_manager = None
            
        def init(self, provider_uri, region, exp_manager):
            self.provider_uri = provider_uri
            self.region = region
            self.exp_manager = exp_manager
    
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
    assert mock_qlib.exp_manager == {
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {
            "uri": env.mlflow_backend,
            "default_exp_name": "Experiment",
        },
    }

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

    env._qlib_initialized = True
    env.set_root_dir(str(ws2))

    assert env.ROOT_DIR == str(ws2)
    assert os.environ["QLIB_WORKSPACE_DIR"] == str(ws2)
    assert env._qlib_initialized is False
    # URI must reference the new workspace; accept either backend.
    uri = os.environ["MLFLOW_TRACKING_URI"]
    assert "Workspace2" in uri or str(ws2) in uri


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


# ── _resolve_mlflow_backend() tests ──────────────────────────────────────────

def test_env_sqlite_default(monkeypatch, tmp_path):
    """Clean workspace (no mlruns/) should default to sqlite:// backend."""
    workspace = tmp_path / "CleanWorkspace"
    workspace.mkdir()

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    from quantpits.utils import env
    importlib.reload(env)

    uri = os.environ["MLFLOW_TRACKING_URI"]
    assert uri.startswith("sqlite:///"), f"Expected sqlite:// URI, got: {uri}"
    assert "mlflow.db" in uri
    assert str(workspace) in uri


def test_env_legacy_mlruns_detection(monkeypatch, tmp_path):
    """Workspace with non-empty mlruns/ should fall back to file:// backend."""
    workspace = tmp_path / "LegacyWorkspace"
    workspace.mkdir()
    # Simulate legacy data: a numeric experiment directory inside mlruns/
    (workspace / "mlruns" / "0").mkdir(parents=True)

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    from quantpits.utils import env
    importlib.reload(env)

    uri = os.environ["MLFLOW_TRACKING_URI"]
    assert uri.startswith("file://"), f"Expected file:// URI for legacy workspace, got: {uri}"
    assert "mlruns" in uri
    # MLFLOW_ALLOW_FILE_STORE must be set so mlflow ≥ 3.0 does not block
    assert os.environ.get("MLFLOW_ALLOW_FILE_STORE", "").lower() == "true"


def test_env_respects_existing_tracking_uri(monkeypatch, tmp_path):
    """User-supplied MLFLOW_TRACKING_URI must not be overwritten by env.py."""
    workspace = tmp_path / "UserWorkspace"
    workspace.mkdir()
    custom_uri = f"sqlite:///{tmp_path}/custom.db"

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", custom_uri)

    from quantpits.utils import env
    importlib.reload(env)

    assert os.environ["MLFLOW_TRACKING_URI"] == custom_uri, (
        "env.py must not overwrite a user-supplied MLFLOW_TRACKING_URI"
    )


def test_env_mlflow3_allow_file_store(monkeypatch, tmp_path):
    """When the resolved backend is file://, MLFLOW_ALLOW_FILE_STORE must be set."""
    workspace = tmp_path / "FileStoreWorkspace"
    workspace.mkdir()
    # Provide a file:// URI directly via the environment (simulates user override)
    mlruns_abs = str(workspace / "mlruns")
    os.makedirs(mlruns_abs, exist_ok=True)
    file_uri = f"file://{mlruns_abs}"

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", file_uri)
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)

    from quantpits.utils import env
    importlib.reload(env)

    assert os.environ["MLFLOW_TRACKING_URI"] == file_uri
    assert os.environ.get("MLFLOW_ALLOW_FILE_STORE", "").lower() == "true", (
        "MLFLOW_ALLOW_FILE_STORE must be 'true' when backend is file://"
    )
