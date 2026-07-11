import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantpits.post_trade.contracts import ExecutionEvidenceGapError, ParsedPostTradeInput, PostTradeSourceRef, SourceChangedError
from quantpits.post_trade.intake import discover_inputs, parse_pending_sources, validate_cross_stream, verify_parsed_sources
from quantpits.utils.workspace import WorkspaceContext


def test_discovery_is_sorted_fingerprinted_and_receipt_aware(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    ctx.data_dir.mkdir()
    late = ctx.data_path("2026-01-02-order.xlsx"); late.write_bytes(b"late")
    early = ctx.data_path("2026-01-01-order.xlsx"); early.write_bytes(b"early")
    first = discover_inputs(ctx, "2026-01-01", "2026-01-03")
    assert [x.trade_date for x in first.order_sources] == ["2026-01-01", "2026-01-02"]
    receipt = {"order:data/2026-01-01-order.xlsx": {"sha256": first.order_sources[0].fingerprint}}
    second = discover_inputs(ctx, "2026-01-01", "2026-01-03", receipts=receipt)
    assert second.order_sources[0].status == "already_ingested"
    early.write_bytes(b"changed")
    assert discover_inputs(ctx, "2026-01-01", "2026-01-03", receipts=receipt).order_sources[0].status == "changed"


def test_invalid_calendar_date_filename_is_reported_and_ignored(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path); ctx.data_dir.mkdir()
    ctx.data_path("2026-99-99-order.xlsx").write_bytes(b"invalid")
    catalog = discover_inputs(ctx, "2026-01-01", "2026-99-99")
    assert not catalog.order_sources
    assert "invalid_source_filename" in {issue.code for issue in catalog.issues}


def test_changed_source_fails_before_parser(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path); ctx.data_dir.mkdir()
    path = ctx.data_path("2026-01-01-order.xlsx"); path.write_bytes(b"one")
    catalog = discover_inputs(ctx, "2026-01-01", "2026-01-01")
    path.write_bytes(b"two")
    with pytest.raises(SourceChangedError):
        parse_pending_sources(catalog, MagicMock())


def test_cross_stream_rejects_trade_without_order(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path); ctx.data_dir.mkdir()
    path = ctx.data_path("2026-01-01-trade.xlsx"); path.write_bytes(b"x")
    catalog = discover_inputs(ctx, "2026-01-01", "2026-01-01")
    adapter = MagicMock()
    adapter.parse_trades.return_value = pd.DataFrame({"成交数量": [1]})
    parsed = parse_pending_sources(catalog, adapter)
    with pytest.raises(ExecutionEvidenceGapError):
        validate_cross_stream(parsed)


def test_assumed_empty_settlement_cannot_hide_fill(tmp_path):
    settlement_source = PostTradeSourceRef("settlement", "2026-01-01", tmp_path / "missing.xlsx", "data/missing.xlsx", "assumed_empty")
    order_source = PostTradeSourceRef("order", "2026-01-01", tmp_path / "order.xlsx", "data/order.xlsx", "present")
    trade_source = PostTradeSourceRef("trade", "2026-01-01", tmp_path / "trade.xlsx", "data/trade.xlsx", "present")
    parsed = {
        ("settlement", "2026-01-01"): ParsedPostTradeInput(settlement_source, pd.DataFrame(), 0),
        ("order", "2026-01-01"): ParsedPostTradeInput(order_source, pd.DataFrame({"成交数量": [1]}), 1),
        ("trade", "2026-01-01"): ParsedPostTradeInput(trade_source, pd.DataFrame({"成交数量": [1]}), 1),
    }
    with pytest.raises(ExecutionEvidenceGapError):
        validate_cross_stream(parsed)


def test_historical_execution_bootstrap_does_not_require_settlement(tmp_path):
    order_source = PostTradeSourceRef("order", "2020-01-01", tmp_path / "order.xlsx", "data/order.xlsx", "present")
    trade_source = PostTradeSourceRef("trade", "2020-01-01", tmp_path / "trade.xlsx", "data/trade.xlsx", "present")
    parsed = {
        ("order", "2020-01-01"): ParsedPostTradeInput(order_source, pd.DataFrame({"成交数量": [1]}), 1),
        ("trade", "2020-01-01"): ParsedPostTradeInput(trade_source, pd.DataFrame({"成交数量": [1]}), 1),
    }
    validate_cross_stream(parsed, settlement_required_dates=("2026-01-02",))


def test_zero_fill_order_without_trade_is_valid():
    source = PostTradeSourceRef("order", "2026-01-01", Path("order.xlsx"), "data/order.xlsx", "present")
    parsed = {
        ("order", "2026-01-01"): ParsedPostTradeInput(source, pd.DataFrame({"成交数量": [0]}), 1),
    }
    validate_cross_stream(parsed)


def test_execution_scope_still_rejects_trade_without_order(tmp_path):
    source = PostTradeSourceRef("trade", "2026-01-01", tmp_path / "trade.xlsx", "data/trade.xlsx", "present")
    parsed = {
        ("trade", "2026-01-01"): ParsedPostTradeInput(source, pd.DataFrame({"成交数量": [1]}), 1),
    }
    with pytest.raises(ExecutionEvidenceGapError):
        validate_cross_stream(parsed, scope="execution")


def test_settlement_interest_only_does_not_require_execution_evidence(tmp_path):
    source = PostTradeSourceRef("settlement", "2026-01-01", tmp_path / "table.xlsx", "data/table.xlsx", "present")
    parsed = {
        ("settlement", "2026-01-01"): ParsedPostTradeInput(
            source, pd.DataFrame({"交易类别": ["利息归本"]}), 1
        ),
    }
    validate_cross_stream(parsed, scope="all")


def test_settlement_trade_requires_order_and_trade_evidence(tmp_path):
    source = PostTradeSourceRef("settlement", "2026-01-01", tmp_path / "table.xlsx", "data/table.xlsx", "present")
    parsed = {
        ("settlement", "2026-01-01"): ParsedPostTradeInput(
            source, pd.DataFrame({"交易类别": ["上海A股普通股票竞价买入"]}), 1
        ),
    }
    with pytest.raises(ExecutionEvidenceGapError):
        validate_cross_stream(parsed, scope="all")


def test_pre_writer_verification_detects_change_after_parse(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path); ctx.data_dir.mkdir()
    path = ctx.data_path("2026-01-01-order.xlsx"); path.write_bytes(b"one")
    catalog = discover_inputs(ctx, "2026-01-01", "2026-01-01")
    adapter = MagicMock()
    def parse_and_mutate(_):
        path.write_bytes(b"two")
        return pd.DataFrame()
    adapter.parse_orders.side_effect = parse_and_mutate
    parsed = parse_pending_sources(catalog, adapter)
    with pytest.raises(SourceChangedError):
        verify_parsed_sources(parsed)
