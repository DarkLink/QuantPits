import importlib.metadata
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

def _read_pyproject_version():
    """Read the canonical version from pyproject.toml."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    for line in pyproject.read_text().splitlines():
        if line.startswith("version"):
            return line.split('"')[1]
    raise RuntimeError("version not found in pyproject.toml")

def test_version_installed():
    import quantpits
    assert quantpits.__version__ is not None

def test_version_package_not_found():
    # Reload package with mocked version function to raise PackageNotFoundError
    expected = _read_pyproject_version()
    # Save quantpits and all its submodules so we can restore them after the test;
    # otherwise subsequent tests fail with ImportError.
    _saved = {k: v for k, v in sys.modules.items()
              if k == "quantpits" or k.startswith("quantpits.")}
    for k in _saved:
        del sys.modules[k]
    try:
        with patch('importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError):
            import quantpits
            assert quantpits.__version__ == expected
    finally:
        sys.modules.update(_saved)
