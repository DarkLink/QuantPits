"""Workspace-bound discovery and strict preflight for broker evidence."""

from __future__ import annotations

import numbers
import os
import re
import stat
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd

from quantpits.post_trade.contracts import (
    ExecutionEvidenceGapError, ParsedPostTradeInput, PostTradeInputCatalog,
    PostTradeInputError, PostTradeIntakeIssue, PostTradePlanError,
    PostTradeSourceRef, SettlementBundleRef, SourceChangedError,
)
from quantpits.post_trade.contracts import PostTradeReceiptLedgerSchemaError
from quantpits.utils.workspace import WorkspaceContext, fingerprint_file
from quantpits.utils.workspace import fingerprint_value

_PATTERNS = {
    "settlement": re.compile(r"^(\d{4}-\d{2}-\d{2})-table\.xlsx$"),
    "order": re.compile(r"^(\d{4}-\d{2}-\d{2})-order\.xlsx$"),
    "trade": re.compile(r"^(\d{4}-\d{2}-\d{2})-trade\.xlsx$"),
}
_BUNDLE_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})-table\.xlsx$"
)


def _display_path(ctx: WorkspaceContext, path: Path) -> str:
    try:
        return path.relative_to(ctx.root).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_iso_date(value: str, *, label: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise PostTradePlanError("Invalid %s date: %s" % (label, value)) from exc
    return parsed.strftime("%Y-%m-%d")


def prepare_settlement_bundle(
    ctx: WorkspaceContext, value: str,
) -> SettlementBundleRef:
    """Resolve and identity-bind one explicit workspace-contained bundle."""
    if not isinstance(value, str) or not value.strip():
        raise PostTradePlanError("--settlement-bundle requires a non-empty path")
    raw = Path(value.strip()).expanduser()
    if raw.is_absolute() or ".." in raw.parts:
        raise PostTradePlanError("--settlement-bundle must be a workspace-relative path")
    public_path = ctx.root / raw
    public_path = public_path.absolute()
    match = _BUNDLE_PATTERN.fullmatch(public_path.name)
    if not match:
        raise PostTradePlanError(
            "Settlement bundle filename must be YYYY-MM-DD-YYYY-MM-DD-table.xlsx"
        )
    coverage_start = _parse_iso_date(match.group(1), label="bundle start")
    coverage_end = _parse_iso_date(match.group(2), label="bundle end")
    if coverage_start > coverage_end:
        raise PostTradePlanError("Settlement bundle start date must not be after end date")
    # Coverage is checked against the resolved trading calendar during strict
    # preflight.  The raw state window can begin on a weekend or holiday and is
    # therefore not an authoritative coverage boundary.
    root = ctx.root.resolve(strict=True)
    try:
        display_path = public_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PostTradePlanError("Settlement bundle path is outside the selected workspace") from exc
    try:
        resolved = public_path.resolve(strict=True)
    except OSError as exc:
        raise PostTradePlanError("Settlement bundle is not an existing regular file") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PostTradePlanError("Settlement bundle resolves outside the selected workspace") from exc
    entry_info = public_path.lstat()
    info = public_path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise PostTradePlanError("Settlement bundle must resolve to a regular file")
    parent = public_path.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise PostTradePlanError("Settlement bundle parent resolves outside the selected workspace") from exc
    root_info, parent_info = root.stat(), parent.stat()
    return SettlementBundleRef(
        path=public_path, resolved_path=resolved,
        display_path=display_path,
        coverage_start=coverage_start, coverage_end=coverage_end,
        fingerprint=fingerprint_file(public_path), size_bytes=info.st_size,
        mtime_ns=info.st_mtime_ns, device=info.st_dev, inode=info.st_ino,
        entry_device=entry_info.st_dev, entry_inode=entry_info.st_ino,
        entry_mtime_ns=entry_info.st_mtime_ns, entry_mode=entry_info.st_mode,
        root_device=root_info.st_dev, root_inode=root_info.st_ino,
        parent_path=parent, parent_device=parent_info.st_dev,
        parent_inode=parent_info.st_ino,
    )


def verify_settlement_bundle(ctx: WorkspaceContext, bundle: SettlementBundleRef) -> None:
    """Re-establish public-name, containment, namespace, and byte identity."""
    try:
        root = ctx.root.resolve(strict=True)
        root_info = root.stat()
        parent = bundle.path.parent.resolve(strict=True)
        parent_info = parent.stat()
        resolved = bundle.path.resolve(strict=True)
        entry_info = bundle.path.lstat()
        info = bundle.path.stat()
        resolved.relative_to(root)
        parent.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SourceChangedError("Settlement bundle namespace changed after planning") from exc
    if (
        (root_info.st_dev, root_info.st_ino) != (bundle.root_device, bundle.root_inode)
        or parent != bundle.parent_path
        or (parent_info.st_dev, parent_info.st_ino) != (bundle.parent_device, bundle.parent_inode)
        or resolved != bundle.resolved_path
        or not stat.S_ISREG(info.st_mode)
        or (entry_info.st_dev, entry_info.st_ino, entry_info.st_mtime_ns, entry_info.st_mode)
        != (
            bundle.entry_device, bundle.entry_inode,
            bundle.entry_mtime_ns, bundle.entry_mode,
        )
        or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        != (bundle.device, bundle.inode, bundle.size_bytes, bundle.mtime_ns)
        or fingerprint_file(bundle.path) != bundle.fingerprint
    ):
        raise SourceChangedError("Settlement bundle changed after planning: %s" % bundle.display_path)


def load_ingestion_receipts(path: Path, *, strict: bool = False) -> Mapping[str, Mapping[str, object]]:
    if not path.exists():
        return {}
    import json
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if strict: raise PostTradeReceiptLedgerSchemaError("Malformed ingestion receipt ledger") from exc
        return {}
    if not isinstance(payload, dict) or payload.get("schema_version", 1) != 1:
        if strict: raise PostTradeReceiptLedgerSchemaError("Unsupported ingestion receipt ledger")
        return {}
    sources = payload.get("sources", {})
    if not isinstance(sources, dict):
        if strict: raise PostTradeReceiptLedgerSchemaError("Receipt sources must be an object")
        return {}
    if strict:
        for key, receipt in sources.items():
            if not isinstance(receipt, dict) or receipt.get("stream") not in {"order", "trade"}:
                raise PostTradeReceiptLedgerSchemaError("Invalid ingestion receipt: %s" % key)
            required = ("trade_date", "path", "sha256", "row_count", "run_id")
            if any(name not in receipt for name in required) or len(str(receipt["sha256"])) != 64:
                raise PostTradeReceiptLedgerSchemaError("Incomplete ingestion receipt: %s" % key)
            expected = "%s:%s" % (receipt["stream"], receipt["path"])
            if key != expected or Path(str(receipt["path"])).is_absolute() or ".." in Path(str(receipt["path"])).parts:
                raise PostTradeReceiptLedgerSchemaError("Receipt key/path mismatch: %s" % key)
    return sources


def discover_inputs(
    ctx: WorkspaceContext, date_from: str, date_to: str, *,
    receipts: Optional[Mapping[str, Mapping[str, object]]] = None,
    stream_date_from: Optional[Mapping[str, str]] = None,
    settlement_bundle: Optional[str] = None,
) -> PostTradeInputCatalog:
    receipts = receipts if receipts is not None else load_ingestion_receipts(ctx.data_path("post_trade_ingestion_state.json"))
    grouped = {"settlement": [], "order": [], "trade": []}
    issues = []
    for stream, pattern in _PATTERNS.items():
        if stream == "settlement" and settlement_bundle is not None:
            continue
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
    bundle = None
    if settlement_bundle is not None:
        bundle = prepare_settlement_bundle(ctx, settlement_bundle)
    return PostTradeInputCatalog(
        date_from=date_from, date_to=date_to,
        settlement_sources=tuple(grouped["settlement"]),
        order_sources=tuple(grouped["order"]), trade_sources=tuple(grouped["trade"]),
        issues=tuple(issues),
        settlement_bundle=bundle,
    )


def verify_source(source: PostTradeSourceRef) -> None:
    if not source.path.exists() or not source.fingerprint:
        raise SourceChangedError("Source disappeared before execution: %s" % source.display_path)
    if fingerprint_file(source.path) != source.fingerprint:
        raise SourceChangedError("Source changed after planning: %s" % source.display_path)


def verify_parsed_sources(parsed: Mapping[Tuple[str, str], ParsedPostTradeInput]) -> None:
    """Recheck every physical source immediately before the first writer."""
    for item in parsed.values():
        if item.source.status != "assumed_empty" and item.source.source_kind != "bundle_partition":
            verify_source(item.source)


def _logical_frame_fingerprint(frame: pd.DataFrame) -> str:
    canonical = frame.reset_index(drop=True).sort_index(axis=1).to_json(
        orient="split", date_format="iso", date_unit="ns", force_ascii=True,
    )
    return fingerprint_value({"schema": 1, "frame": canonical})


def _partition_settlement_bundle(
    catalog: PostTradeInputCatalog, frame: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], ParsedPostTradeInput], Tuple[PostTradeSourceRef, ...]]:
    bundle = catalog.settlement_bundle
    if bundle is None:
        return {}, ()
    if not isinstance(frame, pd.DataFrame):
        raise PostTradeInputError("Settlement bundle parser must return a DataFrame")
    if list(frame.columns).count("交收日期") != 1:
        raise PostTradeInputError("Settlement bundle schema is missing 交收日期")
    from quantpits.post_trade.state import normalize_date
    normalized_dates = []
    for value in frame["交收日期"].tolist():
        if isinstance(value, bool) or not pd.api.types.is_scalar(value):
            raise PostTradeInputError("Settlement bundle contains an invalid settlement date")
        if isinstance(value, numbers.Real) and not float(value).is_integer():
            raise PostTradeInputError("Settlement bundle contains an invalid settlement date")
        if isinstance(value, str) and re.fullmatch(r"\d{8}\.\d+", value.strip()):
            fraction = value.strip().split(".", 1)[1]
            if set(fraction) != {"0"}:
                raise PostTradeInputError("Settlement bundle contains an invalid settlement date")
        if value is None or pd.isna(value) or not str(value).strip():
            raise PostTradeInputError("Settlement bundle contains a missing settlement date")
        try:
            date = normalize_date(value)
            date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
        except Exception as exc:
            raise PostTradeInputError("Settlement bundle contains an invalid settlement date") from exc
        if not bundle.coverage_start <= date <= bundle.coverage_end:
            raise PostTradeInputError("Settlement bundle row falls outside its filename coverage")
        if not catalog.date_from <= date <= catalog.date_to:
            raise PostTradeInputError("Settlement bundle row falls outside the requested state window")
        normalized_dates.append(date)
    working = frame.copy()
    working["__quantpits_bundle_date"] = normalized_dates
    parsed, sources = {}, []
    for date in sorted(set(normalized_dates)):
        logical = working.loc[working["__quantpits_bundle_date"] == date].drop(
            columns=["__quantpits_bundle_date"]
        ).reset_index(drop=True)
        digest = _logical_frame_fingerprint(logical)
        source = PostTradeSourceRef(
            stream="settlement", trade_date=date, path=bundle.path,
            display_path=bundle.display_path, status="present",
            fingerprint=digest, size_bytes=None,
            source_kind="bundle_partition", row_count=len(logical),
        )
        sources.append(source)
        parsed[("settlement", date)] = ParsedPostTradeInput(source, logical, len(logical))
    return parsed, tuple(sources)


def parse_pending_sources_with_catalog(
    catalog: PostTradeInputCatalog, adapter: object, *, reparse_execution_dates=(),
) -> Tuple[Dict[Tuple[str, str], ParsedPostTradeInput], PostTradeInputCatalog]:
    parsed: Dict[Tuple[str, str], ParsedPostTradeInput] = {}
    if catalog.settlement_bundle is not None:
        verify_settlement_bundle_for_parse = catalog.settlement_bundle
        # The physical file is verified once immediately before its only read.
        # The command boundary performs the workspace-aware recheck before calling
        # this helper; the byte fingerprint here still closes direct callers.
        if fingerprint_file(verify_settlement_bundle_for_parse.path) != verify_settlement_bundle_for_parse.fingerprint:
            raise SourceChangedError(
                "Settlement bundle changed after planning: %s"
                % verify_settlement_bundle_for_parse.display_path
            )
        # Keep the planned physical object alive across the adapter call.  An
        # unlink-and-recreate can otherwise reuse the just-freed inode quickly
        # enough to defeat a same-byte namespace identity check.  Pinning the
        # original inode guarantees that any replacement receives a distinct
        # physical identity for the command boundary's post-parse recheck.
        try:
            identity_guard = verify_settlement_bundle_for_parse.path.open("rb")
        except OSError as exc:
            raise SourceChangedError(
                "Settlement bundle namespace changed after planning"
            ) from exc
        try:
            try:
                guard_info = os.fstat(identity_guard.fileno())
            except OSError as exc:
                raise SourceChangedError(
                    "Settlement bundle namespace changed after planning"
                ) from exc
            if (guard_info.st_dev, guard_info.st_ino) != (
                verify_settlement_bundle_for_parse.device,
                verify_settlement_bundle_for_parse.inode,
            ):
                raise SourceChangedError(
                    "Settlement bundle changed after planning: %s"
                    % verify_settlement_bundle_for_parse.display_path
                )
            frame = adapter.parse_settlement(verify_settlement_bundle_for_parse.path)
        finally:
            identity_guard.close()
        bundle_parsed, logical_sources = _partition_settlement_bundle(catalog, frame)
        parsed.update(bundle_parsed)
        catalog = replace(catalog, settlement_sources=logical_sources)
    methods = {"settlement": "parse_settlement", "order": "parse_orders", "trade": "parse_trades"}
    for stream in ("settlement", "order", "trade"):
        if stream == "settlement" and catalog.settlement_bundle is not None:
            continue
        for source in catalog.sources_for(stream):
            if source.status == "assumed_empty":
                frame = pd.DataFrame()
                parsed[(stream, source.trade_date)] = ParsedPostTradeInput(source, frame, 0)
                continue
            if source.status == "already_ingested" and stream != "settlement" and source.trade_date not in set(reparse_execution_dates):
                parsed[(stream, source.trade_date)] = ParsedPostTradeInput(source, pd.DataFrame(), 0)
                continue
            if source.status == "changed":
                raise SourceChangedError("Previously ingested source changed: %s" % source.display_path)
            verify_source(source)
            frame = getattr(adapter, methods[stream])(source.path)
            parsed[(stream, source.trade_date)] = ParsedPostTradeInput(source, frame, len(frame))
    return parsed, catalog


def parse_pending_sources(catalog: PostTradeInputCatalog, adapter: object, *, reparse_execution_dates=()) -> Dict[Tuple[str, str], ParsedPostTradeInput]:
    parsed, _ = parse_pending_sources_with_catalog(
        catalog, adapter, reparse_execution_dates=reparse_execution_dates,
    )
    return parsed


def _has_trade_activity(frame: pd.DataFrame) -> bool:
    if frame.empty or "交易类别" not in frame.columns:
        return False
    from quantpits.scripts.brokers.base import BUY_TYPES, SELL_TYPES
    return frame["交易类别"].isin(BUY_TYPES + SELL_TYPES).any()


def _has_fills(frame: pd.DataFrame) -> bool:
    if frame.empty or "成交数量" not in frame.columns:
        return False
    from quantpits.scripts.brokers.base import BUY_TYPES, SELL_TYPES
    execution_rows = frame["交易类别"].isin(BUY_TYPES + SELL_TYPES) if "交易类别" in frame.columns else True
    return (pd.to_numeric(frame["成交数量"], errors="coerce").fillna(0).gt(0) & execution_rows).any()


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
        if trade is not None and order is None and _has_fills(trade.dataframe):
            raise ExecutionEvidenceGapError("Trade evidence exists without order evidence for %s" % date)
        if order is not None and trade is None and _has_fills(order.dataframe):
            raise ExecutionEvidenceGapError("Filled orders exist without trade evidence for %s" % date)
        if scope == "all" and settlement is not None and _has_trade_activity(settlement.dataframe) and (order is None or trade is None):
            raise ExecutionEvidenceGapError("Settlement trades lack order/trade evidence for %s" % date)
        settlement_missing = settlement is None or settlement.source.status == "assumed_empty"
        if scope == "all" and date in settlement_required and trade is not None and _has_fills(trade.dataframe) and settlement_missing:
            raise ExecutionEvidenceGapError("Trade fills exist without settlement evidence for %s" % date)
