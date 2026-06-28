"""
Rolling Training core module (compatibility re-exports).

Functions have moved to:
  - orchestration.py:   run_model_windows (shared)
  - strategy_slide.py:  train_window_model, train_window_model_isolated (slide)

This module remains for backward compatibility with existing imports.
"""

# Re-export from strategy_slide for backward compatibility
from quantpits.scripts.rolling.strategy_slide import (
    train_window_model,
    train_window_model_isolated,
)

# Re-export from orchestration for backward compatibility
from quantpits.scripts.rolling.orchestration import (
    run_model_windows,
)
