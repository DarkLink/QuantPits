"""QuantPits — An advanced, production-ready quantitative trading system built on top of Microsoft Qlib."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("quantpits")
except PackageNotFoundError:
    # Package is not installed (running from source without pip install -e .)
    __version__ = "0.4.1-alpha"
