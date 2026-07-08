"""Built-in pipeline stages. Import triggers ``@register_stage`` registration.

Import order is irrelevant — the StageRunner executes stages in
topological order derived from each stage's ``depends_on`` declaration.
"""

from .discover import run_discover               # noqa: F401
from .agents import run_agents                   # noqa: F401
from .synthesis import run_synthesis             # noqa: F401
from .window_analysis import run_window_analysis  # noqa: F401
from .signals import run_signals                 # noqa: F401
from .critic_pipeline import run_critic          # noqa: F401
from .report import run_report                   # noqa: F401
