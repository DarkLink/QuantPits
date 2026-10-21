"""A fresh project parent must not import legacy workspace globals."""
import os
from pathlib import Path
import subprocess
import sys


def test_cycle_and_fusion_preparation_import_without_workspace(tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    for name in ("QLIB_WORKSPACE_DIR", "MLFLOW_TRACKING_URI", "QLIB_DATA_DIR", "QLIB_REGION"):
        environment.pop(name, None)
    source = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    names = args[2] if len(args) > 2 else kwargs.get('fromlist', ())
    if name == 'quantpits.utils.env' or (name == 'quantpits.utils' and 'env' in (names or ())):
        raise AssertionError('legacy env imported by clean project parent')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from quantpits.scripts.run_cycle import main
from quantpits.ensemble.service import prepare_ensemble_run
from quantpits.utils.ensemble_plan import resolve_ensemble_combos
from types import SimpleNamespace
result = resolve_ensemble_combos(args=SimpleNamespace(models='selected', method='equal'),
    train_records={'models': {'selected@static': 'fixed-source'}}, ensemble_config={})
assert result[0].models == ('selected@static',)
"""
    completed = subprocess.run([sys.executable, "-c", source], cwd=tmp_path, env=environment,
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
