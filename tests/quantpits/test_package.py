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
    with patch('importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError):
        if 'quantpits' in sys.modules:
            del sys.modules['quantpits']
        import quantpits
        assert quantpits.__version__ == expected
