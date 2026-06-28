"""
Rolling Windows generation module (compatibility re-exports).

Functions have moved to strategy_slide.py and strategy_cpcv.py.
This module remains for backward compatibility with existing imports.
"""

# Re-export from strategy_slide for backward compatibility
from quantpits.scripts.rolling.strategy_slide import (
    generate_rolling_windows,
    generate_windows,
    parse_step_to_relativedelta,
)
