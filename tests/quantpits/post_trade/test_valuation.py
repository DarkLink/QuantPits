from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quantpits.post_trade.contracts import ValuationMissingError, ValuationSchemaError
from quantpits.post_trade.valuation import QlibValuationProvider


def test_valuation_requests_only_exact_instruments(monkeypatch):
    calls = []
    index = pd.MultiIndex.from_tuples([("SH000001", pd.Timestamp("2026-01-02")), ("SH000300", pd.Timestamp("2026-01-02"))], names=["instrument", "datetime"])
    frame = pd.DataFrame({
        "$close": [20.0, 40.0], "$factor": [2.0, 2.0],
        "Div($close,$factor)": [10.0, 20.0],
    }, index=index)
    class D:
        @staticmethod
        def features(instruments, fields, start_time, end_time):
            calls.append(tuple(instruments)); return frame.loc[list(instruments)]
    import qlib.data
    monkeypatch.setattr(qlib.data, "D", D())
    result = QlibValuationProvider().snapshot("2026-01-02", ("SH000001",), "SH000300")
    assert calls == [("SH000001",), ("SH000300",)]
    assert result.close_map()["SH000001"] == Decimal("10.0")
    assert result.quote_evidence[0].raw_close == Decimal("20.0")


def test_valuation_missing_instrument_fails(monkeypatch):
    import qlib.data
    monkeypatch.setattr(qlib.data, "D", type("D", (), {"features": staticmethod(lambda **kwargs: pd.DataFrame())})())
    with pytest.raises(ValuationMissingError):
        QlibValuationProvider().snapshot("2026-01-02", ("SH000001",), "SH000300")


def test_valuation_accepts_qlib_float_quantization_within_one_cent():
    value = QlibValuationProvider._validated_derived_quote(
        Decimal("25.372746"), Decimal("0.6673526"), Decimal("38.019997"),
        "SH600036",
    )
    assert value == Decimal("38.02")


def test_valuation_rejects_close_factor_mismatch_across_cent_boundary():
    with pytest.raises(ValuationSchemaError, match="SH600036"):
        QlibValuationProvider._validated_derived_quote(
            Decimal("20"), Decimal("2"), Decimal("10.011"), "SH600036",
        )


@pytest.mark.parametrize("selected", ["SH600001", "SH000300"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_valuation_preserves_float32_values_at_cent_boundary(monkeypatch, selected, corrupt):
    index = pd.MultiIndex.from_tuples(
        [("SH600001", pd.Timestamp("2024-05-06")), ("SH000300", pd.Timestamp("2024-05-06"))],
        names=["instrument", "datetime"],
    )
    frame = pd.DataFrame({
        "$close": np.array([20, 20], dtype="float32"),
        "$factor": np.array([2, 2], dtype="float32"),
        "Div($close,$factor)": np.array([10, 10], dtype="float32"),
    }, index=index)
    raw, factor = np.float32("0.003945"), np.float32("0.003")
    derived = np.float32("1.33") if corrupt else np.float32(raw / factor)
    frame.loc[selected, ["$close", "$factor", "Div($close,$factor)"]] = [raw, factor, derived]

    class Data:
        @staticmethod
        def features(instruments, fields, start_time, end_time):
            return frame.loc[list(instruments)]

    import qlib.data
    monkeypatch.setattr(qlib.data, "D", Data())
    if corrupt:
        with pytest.raises(ValuationSchemaError, match=selected):
            QlibValuationProvider().snapshot("2024-05-06", ("SH600001",), "SH000300")
        return

    result = QlibValuationProvider().snapshot("2024-05-06", ("SH600001",), "SH000300")
    quote = result.quote_evidence[0] if selected == "SH600001" else result.benchmark_evidence
    assert quote.price == Decimal("1.31")
    assert quote.raw_close == Decimal(str(float(raw)))
    assert quote.factor == Decimal(str(float(factor)))
