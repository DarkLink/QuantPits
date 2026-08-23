from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantpits.order.command import ResolvedOrderSource
from quantpits.order.execution import (
    InvalidPredictionDataError,
    OrderSourceUnavailableError,
    UniverseObservationError,
    UniverseSnapshot,
    load_resolved_prediction,
    normalize_prediction_data,
    resolve_exact_universe,
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


def test_universe_snapshot_canonicalizes_and_fingerprints_members():
    first = UniverseSnapshot.observe("demo", "2026-01-01", ["B", "A"])
    second = UniverseSnapshot.observe("demo", "2026-01-01", ["A", "B"])
    assert first.instruments == ("A", "B")
    assert first.instrument_count == 2
    assert first.fingerprint_algorithm == "sha256"
    assert first.fingerprint == second.fingerprint
    with pytest.raises(TypeError):
        UniverseSnapshot("demo", "2026-01-01", ("A",), 1, "sha256", "forged")


@pytest.mark.parametrize("members", [[], [""], [" A"], ["A", "A"], [1]])
def test_universe_snapshot_rejects_invalid_members(members):
    with pytest.raises(UniverseObservationError):
        UniverseSnapshot.observe("demo", "2026-01-01", members)


def test_exact_universe_resolver_uses_frozen_qlib_query():
    provider = MagicMock()
    provider.instruments.return_value = "universe-expression"
    provider.list_instruments.return_value = ["B", "A"]
    with patch("qlib.data.D", provider):
        snapshot = resolve_exact_universe("demo", "2026-01-01")
    provider.instruments.assert_called_once_with(market="demo")
    provider.list_instruments.assert_called_once_with(
        "universe-expression", start_time="2026-01-01", end_time="2026-01-01",
        freq="day", as_list=True,
    )
    assert snapshot.instruments == ("A", "B")
