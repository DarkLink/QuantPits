"""Explicit valuation boundary for post-trade state calculation."""

from __future__ import annotations

from decimal import Decimal
from typing import Mapping, Protocol, Tuple

from quantpits.post_trade.contracts import ValuationMissingError, ValuationSchemaError
from quantpits.post_trade.state import ValuationSnapshot, decimal_value, normalize_instrument
from quantpits.post_trade.valuation_provenance import ValuationQuoteEvidence


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
        closes = tuple(sorted((item, decimal_value(available[item], field="close")) for item in instruments))
        evidence = tuple(ValuationQuoteEvidence(
            item, value, "account_valuation", trade_date, None, trade_date,
            "mapping", "mapping_close", "provided_close", "test_fixture", None, None,
        ) for item, value in closes)
        benchmark_value = decimal_value(self._benchmarks[trade_date], field="benchmark")
        benchmark_evidence = ValuationQuoteEvidence(
            normalize_instrument(benchmark), benchmark_value, "account_valuation", trade_date,
            None, trade_date, "mapping", "mapping_close", "provided_close", "test_fixture", None, None,
        )
        return ValuationSnapshot(trade_date, closes, benchmark_value, evidence, benchmark_evidence)


class QlibValuationProvider:
    def snapshot(self, trade_date, instruments, benchmark):
        from qlib.data import D
        fields = ["$close", "$factor", "Div($close,$factor)"]
        derived = fields[-1]
        closes, evidence = {}, {}
        if instruments:
            frame = D.features(instruments=list(instruments), fields=fields, start_time=trade_date, end_time=trade_date)
            if frame is None or frame.empty:
                raise ValuationMissingError("No valuation data for %s" % trade_date)
            reset = frame.reset_index()
            if "instrument" not in reset.columns:
                raise ValuationSchemaError("Valuation data has no instrument column")
            missing_columns = [name for name in fields if name not in reset.columns]
            if missing_columns:
                raise ValuationSchemaError("Valuation data is missing exact fields: %s" % missing_columns)
            for instrument, group in reset.groupby("instrument"):
                if len(group) != 1:
                    raise ValuationSchemaError("Duplicate close rows for %s" % instrument)
                normalized = normalize_instrument(instrument)
                raw = decimal_value(group.iloc[0]["$close"], field="raw close")
                factor = decimal_value(group.iloc[0]["$factor"], field="factor")
                value = decimal_value(group.iloc[0][derived], field="close")
                if factor == 0 or abs((raw / factor) - value) > Decimal("0.000001"):
                    raise ValuationSchemaError("Raw close/factor does not match derived close for %s" % normalized)
                closes[normalized] = value
                evidence[normalized] = ValuationQuoteEvidence(
                    normalized, value, "account_valuation", trade_date, None, trade_date,
                    "qlib_local", derived, "unadjusted_close_from_close_div_factor",
                    "formula_verified", raw, factor,
                )
        missing = sorted(set(instruments) - set(closes))
        if missing:
            raise ValuationMissingError("Missing close prices for: %s" % ", ".join(missing))
        benchmark_frame = D.features(instruments=[benchmark], fields=fields, start_time=trade_date, end_time=trade_date)
        if benchmark_frame is None or len(benchmark_frame) != 1:
            raise ValuationMissingError("Missing benchmark close for %s" % trade_date)
        benchmark_reset = benchmark_frame.reset_index()
        if any(name not in benchmark_reset.columns for name in fields):
            raise ValuationSchemaError("Benchmark valuation data is missing exact fields")
        benchmark_raw = decimal_value(benchmark_reset.iloc[0]["$close"], field="benchmark raw close")
        benchmark_factor = decimal_value(benchmark_reset.iloc[0]["$factor"], field="benchmark factor")
        benchmark_value = decimal_value(benchmark_reset.iloc[0][derived], field="benchmark")
        if benchmark_factor == 0:
            raise ValuationSchemaError("Benchmark factor must be nonzero")
        benchmark_evidence = ValuationQuoteEvidence(
            normalize_instrument(benchmark), benchmark_value, "account_valuation", trade_date,
            None, trade_date, "qlib_local", derived,
            "provider_derived_index_close", "provider_reported", benchmark_raw, benchmark_factor,
        )
        return ValuationSnapshot(
            trade_date, tuple(sorted(closes.items())), benchmark_value,
            tuple(evidence[key] for key in sorted(evidence)), benchmark_evidence,
        )
