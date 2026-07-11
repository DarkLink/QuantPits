"""Explicit valuation boundary for post-trade state calculation."""

from __future__ import annotations

from decimal import Decimal
from typing import Mapping, Protocol, Tuple

from quantpits.post_trade.contracts import ValuationMissingError, ValuationSchemaError
from quantpits.post_trade.state import ValuationSnapshot, decimal_value, normalize_instrument


class ValuationProvider(Protocol):
    def snapshot(self, trade_date: str, instruments: Tuple[str, ...], benchmark: str) -> ValuationSnapshot: ...


class MappingValuationProvider:
    def __init__(self, closes_by_date: Mapping[str, Mapping[str, object]], benchmarks: Mapping[str, object]):
        self._closes = closes_by_date; self._benchmarks = benchmarks

    def snapshot(self, trade_date, instruments, benchmark):
        available = self._closes.get(trade_date, {})
        missing = [item for item in instruments if item not in available]
        if missing:
            raise ValuationMissingError("Missing close prices for: %s" % ", ".join(missing))
        if trade_date not in self._benchmarks:
            raise ValuationMissingError("Missing benchmark close for %s" % trade_date)
        return ValuationSnapshot(trade_date, tuple(sorted((item, decimal_value(available[item], field="close")) for item in instruments)), decimal_value(self._benchmarks[trade_date], field="benchmark"))


class QlibValuationProvider:
    def snapshot(self, trade_date, instruments, benchmark):
        from qlib.data import D
        field = "Div($close,$factor)"
        closes = {}
        if instruments:
            frame = D.features(instruments=list(instruments), fields=[field], start_time=trade_date, end_time=trade_date)
            if frame is None:
                raise ValuationMissingError("No valuation data for %s" % trade_date)
            reset = frame.reset_index()
            value_col = field if field in reset.columns else reset.columns[-1]
            for instrument, group in reset.groupby("instrument"):
                if len(group) != 1:
                    raise ValuationSchemaError("Duplicate close rows for %s" % instrument)
                closes[normalize_instrument(instrument)] = decimal_value(group.iloc[0][value_col], field="close")
        missing = sorted(set(instruments) - set(closes))
        if missing:
            raise ValuationMissingError("Missing close prices for: %s" % ", ".join(missing))
        benchmark_frame = D.features(instruments=[benchmark], fields=[field], start_time=trade_date, end_time=trade_date)
        if benchmark_frame is None or len(benchmark_frame) != 1:
            raise ValuationMissingError("Missing benchmark close for %s" % trade_date)
        benchmark_value = decimal_value(benchmark_frame.reset_index().iloc[0][field], field="benchmark")
        return ValuationSnapshot(trade_date, tuple(sorted(closes.items())), benchmark_value)
