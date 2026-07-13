from decimal import Decimal

import pandas as pd
import pytest

from quantpits.post_trade.contracts import ValuationMissingError
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
