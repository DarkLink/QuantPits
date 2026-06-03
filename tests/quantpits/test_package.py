import importlib.metadata
import sys
from unittest.mock import patch
import pytest

def test_version_installed():
    import quantpits
    assert quantpits.__version__ is not None

def test_version_package_not_found():
    # Reload package with mocked version function to raise PackageNotFoundError
    with patch('importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError):
        if 'quantpits' in sys.modules:
            del sys.modules['quantpits']
        import quantpits
        assert quantpits.__version__ == "0.4.2-alpha"
