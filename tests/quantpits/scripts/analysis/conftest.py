"""
conftest for analysis tests.

Set up sys.path and QLIB_WORKSPACE_DIR *before* any analysis module
is imported, because the analysis source files do `import env` at
module level.

Integration test files (*_integration.py) are excluded from collection
when the workspace data is not available, so they never show as SKIPPED.
"""
import os
import sys
import importlib

# Add scripts dir to sys.path so bare `import env` resolves
_scripts_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'quantpits', 'scripts')
)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Ensure QLIB_WORKSPACE_DIR is set (env.py needs it at import time)
if "QLIB_WORKSPACE_DIR" not in os.environ:
    os.environ["QLIB_WORKSPACE_DIR"] = "/tmp/MockWorkspace_conftest"
    os.makedirs("/tmp/MockWorkspace_conftest/config", exist_ok=True)


def _has_workspace_data():
    """Check whether the configured workspace has real production data."""
    ws = os.environ.get("QLIB_WORKSPACE_DIR", "")
    return ws and os.path.isfile(os.path.join(ws, "data", "daily_amount_log_full.csv"))


# Completely exclude integration test files from collection when data unavailable.
# This is cleaner than pytest.mark.skipif — tests won't show as SKIPPED in CI.
collect_ignore_glob = []
if not _has_workspace_data():
    collect_ignore_glob.append("*_integration.py")
