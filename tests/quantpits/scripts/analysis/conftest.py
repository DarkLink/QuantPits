"""
conftest for analysis tests.

Set up sys.path and QLIB_WORKSPACE_DIR *before* any analysis module
is imported, because the analysis source files do `import env` at
module level.
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
