"""
Rolling Prediction stitching & saving module (compatibility re-exports).

Functions have moved to:
  - orchestration.py:   concatenate_rolling_predictions, _filter_pred_to_test_segment,
                         save_rolling_records (shared)
  - strategy_slide.py:  predict_with_latest_model, _repair_truncated_prediction (slide)

This module remains for backward compatibility with existing imports.
"""

# Re-export from orchestration
from quantpits.scripts.rolling.orchestration import (
    concatenate_rolling_predictions,
    save_rolling_records,
    _filter_pred_to_test_segment,
)

# Re-export from strategy_slide (using backward-compat aliases)
from quantpits.scripts.rolling.strategy_slide import (
    predict_with_latest_model,
    _repair_truncated_prediction,
)
