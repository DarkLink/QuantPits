"""Workspace-bound discovery and strict preflight for broker evidence."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from quantpits.post_trade.contracts import (
    ExecutionEvidenceGapError, ParsedPostTradeInput, PostTradeInputCatalog,
    PostTradeIntakeIssue, PostTradeSourceRef, SourceChangedError,
)
from quantpits.utils.workspace import WorkspaceContext, fingerprint_file

_PATTERNS = {
    "settlement": re.compile(r"^(\d{4}-\d{2}-\d{2})-table\.xlsx$"),
    "order": re.compile(r"^(\d{4}-\d{2}-\d{2})-order\.xlsx$"),
    "trade": re.compile(r"^(\d{4}-\d{2}-\d{2})-trade\.xlsx$"),
}


def _display_path(ctx: WorkspaceContext, path: Path) -> str:
    try:
        return path.relative_to(ctx.root).as_posix()
    except ValueError:
        return path.as_posix()


def load_ingestion_receipts(path: Path) -> Mapping[str, Mapping[str, object]]:
    if not path.exists():
        return {}
    import json
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    sources = payload.get("sources", {})
    return sources if isinstance(sources, dict) else {}


def discover_inputs(
    ctx: WorkspaceContext, date_from: str, date_to: str, *,
    receipts: Optional[Mapping[str, Mapping[str, object]]] = None,
    stream_date_from: Optional[Mapping[str, str]] = None,
) -> PostTradeInputCatalog:
    receipts = receipts if receipts is not None else load_ingestion_receipts(ctx.data_path("post_trade_ingestion_state.json"))
    grouped = {"settlement": [], "order": [], "trade": []}
    issues = []
    for stream, pattern in _PATTERNS.items():
        lower_bound = (stream_date_from or {}).get(stream, date_from)
        suffix = {"settlement": "table", "order": "order", "trade": "trade"}[stream]
        for path in sorted(ctx.data_dir.glob("????-??-??-%s.xlsx" % suffix)):
            match = pattern.match(path.name)
            if not match:
                continue
            trade_date = match.group(1)
            try:
                datetime.strptime(trade_date, "%Y-%m-%d")
            except ValueError:
                issues.append(PostTradeIntakeIssue(
                    code="invalid_source_filename", severity="warning",
                    message="Broker source filename contains an invalid calendar date.",
                    stream=stream, trade_date=trade_date,
                ))
                continue
            if not lower_bound <= trade_date <= date_to:
                continue
            digest = fingerprint_file(path)
            display = _display_path(ctx, path)
            key = "%s:%s" % (stream, display)
            receipt = receipts.get(key)
            if receipt and receipt.get("sha256") == digest:
                status = "already_ingested"
            elif receipt:
                status = "changed"
            else:
                status = "present"
            grouped[stream].append(PostTradeSourceRef(
                stream=stream, trade_date=trade_date, path=path.resolve(),
                display_path=display, status=status, fingerprint=digest,
                size_bytes=path.stat().st_size,
            ))
            if status == "changed":
                issues.append(PostTradeIntakeIssue(
                    code="source_changed", severity="error",
                    message="Previously ingested source fingerprint changed.",
                    stream=stream, trade_date=trade_date,
                ))
    issues.append(PostTradeIntakeIssue(
        code="calendar_resolution_deferred", severity="info",
        message="Exact trading-calendar completeness is resolved during strict preflight.",
    ))
    if any(receipts.values()):
        issues.append(PostTradeIntakeIssue(
            code="legacy_cursor_ignored_for_discovery", severity="info",
            message="Source receipts, not the legacy max-date cursor, control execution discovery.",
        ))
    return PostTradeInputCatalog(
        date_from=date_from, date_to=date_to,
        settlement_sources=tuple(grouped["settlement"]),
        order_sources=tuple(grouped["order"]), trade_sources=tuple(grouped["trade"]),
        issues=tuple(issues),
    )


def verify_source(source: PostTradeSourceRef) -> None:
    if not source.path.exists() or not source.fingerprint:
        raise SourceChangedError("Source disappeared before execution: %s" % source.display_path)
    if fingerprint_file(source.path) != source.fingerprint:
        raise SourceChangedError("Source changed after planning: %s" % source.display_path)


def verify_parsed_sources(parsed: Mapping[Tuple[str, str], ParsedPostTradeInput]) -> None:
    """Recheck every physical source immediately before the first writer."""
    for item in parsed.values():
        if item.source.status != "assumed_empty":
            verify_source(item.source)


def parse_pending_sources(catalog: PostTradeInputCatalog, adapter: object, *, reparse_execution_dates=()) -> Dict[Tuple[str, str], ParsedPostTradeInput]:
    parsed: Dict[Tuple[str, str], ParsedPostTradeInput] = {}
    methods = {"settlement": "parse_settlement", "order": "parse_orders", "trade": "parse_trades"}
    for stream in ("settlement", "order", "trade"):
        for source in catalog.sources_for(stream):
            if source.status == "assumed_empty":
                frame = pd.DataFrame()
                parsed[(stream, source.trade_date)] = ParsedPostTradeInput(source, frame, 0)
                continue
            if source.status == "already_ingested" and stream != "settlement" and source.trade_date not in set(reparse_execution_dates):
                # The receipt is authoritative for presence; raw rows remain in
                # cumulative logs and must not be reparsed or re-ingested.
                parsed[(stream, source.trade_date)] = ParsedPostTradeInput(source, pd.DataFrame(), 0)
                continue
            if source.status == "changed":
                raise SourceChangedError("Previously ingested source changed: %s" % source.display_path)
            verify_source(source)
            frame = getattr(adapter, methods[stream])(source.path)
            parsed[(stream, source.trade_date)] = ParsedPostTradeInput(source, frame, len(frame))
    return parsed


def _has_trade_activity(frame: pd.DataFrame) -> bool:
    if frame.empty or "交易类别" not in frame.columns:
        return False
    from quantpits.scripts.brokers.base import BUY_TYPES, SELL_TYPES
    return frame["交易类别"].isin(BUY_TYPES + SELL_TYPES).any()


def _has_fills(frame: pd.DataFrame) -> bool:
    if frame.empty or "成交数量" not in frame.columns:
        return False
    return pd.to_numeric(frame["成交数量"], errors="coerce").fillna(0).gt(0).any()


def validate_cross_stream(
    parsed: Mapping[Tuple[str, str], ParsedPostTradeInput], *, scope: str = "all",
    settlement_required_dates: Optional[Sequence[str]] = None,
) -> None:
    if scope == "state":
        return
    dates = {date for _, date in parsed}
    settlement_required = set(settlement_required_dates or dates)
    for date in sorted(dates):
        settlement = parsed.get(("settlement", date))
        order = parsed.get(("order", date))
        trade = parsed.get(("trade", date))
        if trade is not None and order is None:
            raise ExecutionEvidenceGapError("Trade evidence exists without order evidence for %s" % date)
        if order is not None and trade is None and _has_fills(order.dataframe):
            raise ExecutionEvidenceGapError("Filled orders exist without trade evidence for %s" % date)
        if scope == "all" and settlement is not None and _has_trade_activity(settlement.dataframe) and (order is None or trade is None):
            raise ExecutionEvidenceGapError("Settlement trades lack order/trade evidence for %s" % date)
        settlement_missing = settlement is None or settlement.source.status == "assumed_empty"
        if scope == "all" and date in settlement_required and trade is not None and _has_fills(trade.dataframe) and settlement_missing:
            raise ExecutionEvidenceGapError("Trade fills exist without settlement evidence for %s" % date)
