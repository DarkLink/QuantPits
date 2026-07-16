"""
conftest for analysis tests.

Set up sys.path and QLIB_WORKSPACE_DIR *before* any analysis module
is imported, because the analysis source files do `import env` at
module level.

Real-workspace tests remain visible at collection and explicitly skip when
their opt-in data precondition is absent.
"""
import os
import sys

# Add scripts dir to sys.path so bare `import env` resolves
_scripts_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'quantpits', 'scripts')
)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
