"""Root conftest for the QuantPits test suite."""

import atexit as _atexit
import os
import shutil as _shutil
import sys
import tempfile as _tempfile

# ── Session-wide test workspace bootstrap ────────────────────────────
# env.py enforces QLIB_WORKSPACE_DIR at import time (line 29).  Pytest
# imports test modules during collection *before* any fixture runs, so we
# must satisfy this requirement with module-level code here.
#
# - Create one temporary workspace for the entire test session.
# - os.environ.setdefault preserves an externally-supplied value (e.g.
#   user running tests inside a sourced workspace).
# - Cleaned up via atexit; typical /tmp tmpwatch also handles orphans.

_TEST_WORKSPACE = _tempfile.mkdtemp(prefix="qp_test_ws_")
os.makedirs(os.path.join(_TEST_WORKSPACE, "config"), exist_ok=True)
os.makedirs(os.path.join(_TEST_WORKSPACE, "data", "history"), exist_ok=True)
os.makedirs(os.path.join(_TEST_WORKSPACE, "output", "predictions"), exist_ok=True)
os.environ.setdefault("QLIB_WORKSPACE_DIR", _TEST_WORKSPACE)


@_atexit.register
def _cleanup_test_workspace() -> None:
    _shutil.rmtree(_TEST_WORKSPACE, ignore_errors=True)


import pytest

@pytest.fixture(autouse=True)
def prevent_mlruns(monkeypatch, tmp_path):
    """
    Globally redirect MLflowExpManager's URI to a temporary SQLite database so
    that unmocked instances of Qlib workflow recorder won't create artefacts in
    the project root.  Uses SQLite (not file://) so the fixture works on both
    mlflow < 3.0 and mlflow ≥ 3.0 without requiring MLFLOW_ALLOW_FILE_STORE.

    Setting MLFLOW_ALLOW_FILE_STORE=true is also applied as a belt-and-braces
    guard: some tests reload env.py which may detect legacy mlruns/ data and
    set a file:// URI; the env var prevents mlflow ≥ 3.0 from blocking it.
    """
    # Guard against mlflow ≥ 3.0 file-store gate for any file:// URIs that
    # might be set by tests that reload env.py with legacy mlruns/ data.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    import qlib.workflow.expm
    original_init = qlib.workflow.expm.MLflowExpManager.__init__

    def mocked_init(self, uri, default_exp_name, *args, **kwargs):
        # Force the URI to a per-test SQLite DB to avoid writing to the project root.
        safe_uri = f"sqlite:///{tmp_path / 'mock_mlflow.db'}"

        # Log the caller so we can clean up the tests later
        import traceback
        with open("unmocked_mlruns.log", "a") as f:
            f.write("=" * 60 + "\\n")
            f.write(f"WARNING: Unmocked MLflowExpManager created for {default_exp_name}!\\n")
            f.write("This means a test failed to properly mock qlib.workflow.R.\\n")
            # Only print the last 15 stack frames to avoid huge logs
            traceback.print_stack(limit=15, file=f)

        original_init(self, safe_uri, default_exp_name, *args, **kwargs)

    monkeypatch.setattr(qlib.workflow.expm.MLflowExpManager, "__init__", mocked_init)


# Add scripts directory to sys.path so bare `import env` and other script module imports work
_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'quantpits', 'scripts'))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

@pytest.fixture
def mock_qlib_workspace(tmp_path):
    """Fixture providing a temporary workspace directory with Qlib structure."""
    workspace = tmp_path / "MockWorkspace"
    os.makedirs(workspace / "config", exist_ok=True)
    os.makedirs(workspace / "data" / "history", exist_ok=True)
    os.makedirs(workspace / "output" / "predictions", exist_ok=True)
    return workspace

@pytest.fixture
def mock_env_constants(monkeypatch, mock_qlib_workspace):
    """
    Shared fixture to mock Qlib workspace environment for all utils tests.
    Sets up a temporary workspace and patches env/train_utils variables.
    """
    workspace = mock_qlib_workspace
    import importlib
    
    # Ensure sys.argv is clean for env.py parsing
    monkeypatch.setattr(sys, 'argv', ['pytest'])
    # satisfy quantpits.utils.env check
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))
    
    # Reload env and train_utils to apply the new environment
    from quantpits.utils import env, train_utils
    importlib.reload(env)
    importlib.reload(train_utils)
    
    # Patch module-level constants in train_utils to use temporary paths
    monkeypatch.setattr(train_utils, 'ROOT_DIR', str(workspace))
    monkeypatch.setattr(train_utils, 'RECORD_OUTPUT_FILE', str(workspace / "latest_train_records.json"))
    monkeypatch.setattr(train_utils, 'HISTORY_DIR', str(workspace / "data" / "history"))
    monkeypatch.setattr(train_utils, 'RUN_STATE_FILE', str(workspace / "data" / "run_state.json"))
    monkeypatch.setattr(train_utils, 'PRETRAINED_DIR', str(workspace / "data" / "pretrain"))
    monkeypatch.setattr(train_utils, 'PREDICTION_OUTPUT_DIR', str(workspace / "output" / "predictions"))
    
    # Create required directories for constants
    os.makedirs(train_utils.PRETRAINED_DIR, exist_ok=True)
    
    yield train_utils, workspace

@pytest.fixture
def mock_script_context(monkeypatch, mock_qlib_workspace):
    """
    Factory fixture to create a script-specific mock context.
    Usage:
        def test_something(mock_script_context):
            import my_script
            ctx = mock_script_context(my_script, ["quantpits.utils.env", "quantpits.utils.strategy"])
            # Now my_script.ROOT_DIR is patched, and env/strategy are reloaded.
    """
    def _create_context(script_module, reload_modules=None):
        import importlib
        import sys
        
        # 1. Clean up argv
        monkeypatch.setattr(sys, 'argv', ['script.py'])
        
        # 2. Set Env
        monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(mock_qlib_workspace))
        
        # 3. Reload requested modules
        if reload_modules:
            for mod_name in reload_modules:
                if mod_name in sys.modules:
                    importlib.reload(sys.modules[mod_name])
        
        # 4. Patch script's ROOT_DIR and other common paths if they exist
        if hasattr(script_module, 'ROOT_DIR'):
            monkeypatch.setattr(script_module, 'ROOT_DIR', str(mock_qlib_workspace))
        
        # Many scripts use these, so we patch them if present
        for path_name in ['CONFIG_FILE', 'CASHFLOW_FILE', 'ENSEMBLE_CONFIG_FILE']:
            if hasattr(script_module, path_name):
                # We assume these are based on ROOT_DIR or we can just point to mock_qlib_workspace
                orig_path = getattr(script_module, path_name)
                # If it's a relative path or inside a different ROOT, we force it to mock_qlib_workspace
                filename = os.path.basename(orig_path)
                monkeypatch.setattr(script_module, path_name, os.path.join(mock_qlib_workspace, "config", filename))
                
        return mock_qlib_workspace
        
    return _create_context

# conftest.py can be used for globally shared fixtures
