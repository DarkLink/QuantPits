from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantpits.order.command import ResolvedOrderSource
from quantpits.order.execution import (
    InvalidPredictionDataError,
    OrderSourceUnavailableError,
    load_resolved_prediction,
    normalize_prediction_data,
)


def _source(record_id="rid"):
    return ResolvedOrderSource("ensemble", None, "demo", record_id, "Ensemble_Fusion", "ensemble")


def test_load_resolved_prediction_uses_prepared_record():
    index = pd.MultiIndex.from_tuples([("A", pd.Timestamp("2026-01-01"))], names=["instrument", "datetime"])
    recorder = MagicMock()
    recorder.load_object.return_value = pd.Series([0.5], index=index)
    runtime = MagicMock()
    runtime.get_recorder.return_value = recorder
    with patch.dict("sys.modules", {"qlib.workflow": MagicMock(R=runtime)}):
        loaded = load_resolved_prediction(_source())
    runtime.get_recorder.assert_called_once_with(recorder_id="rid", experiment_name="Ensemble_Fusion")
    assert loaded.data.columns.tolist() == ["score"]


def test_missing_prepared_record_is_typed_failure():
    with pytest.raises(OrderSourceUnavailableError):
        load_resolved_prediction(_source(None))


def test_invalid_prediction_is_typed_failure():
    with pytest.raises(InvalidPredictionDataError):
        normalize_prediction_data(pd.DataFrame({"label": ["bad"]}))
