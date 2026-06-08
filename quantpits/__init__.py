"""QuantPits — An advanced, production-ready quantitative trading system built on top of Microsoft Qlib."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("quantpits")
except PackageNotFoundError:
    # Package is not installed (running from source without pip install -e .)
    __version__ = "0.4.3-alpha"

# Explicitly import subpackages so that `quantpits.utils` and `quantpits.scripts`
# are accessible as attributes on Python 3.8–3.10.
# On Python 3.11+ the interpreter resolves these automatically via sys.modules,
# but older versions require the parent package to hold a direct reference.
from quantpits import scripts, utils, tools  # noqa: E402
