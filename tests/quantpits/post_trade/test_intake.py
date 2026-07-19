import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantpits.post_trade.contracts import ExecutionEvidenceGapError, ParsedPostTradeInput, PostTradeInputError, PostTradePlanError, PostTradeSourceRef, SourceChangedError
from quantpits.post_trade.intake import discover_inputs, parse_pending_sources, parse_pending_sources_with_catalog, validate_cross_stream, verify_parsed_sources
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


def test_bonus_share_trade_does_not_require_order(tmp_path):
    source = PostTradeSourceRef("trade", "2026-07-09", tmp_path / "trade.xlsx", "data/trade.xlsx", "present")
    parsed = {
        ("trade", "2026-07-09"): ParsedPostTradeInput(
            source,
            pd.DataFrame({"交易类别": ["上海A股红股上市入账"], "成交数量": [180]}),
            1,
        ),
    }
    validate_cross_stream(parsed, scope="all")


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


def _bundle_ctx(tmp_path, name="2026-01-02-2026-01-05-table.xlsx"):
    ctx = WorkspaceContext.from_root(tmp_path)
    ctx.data_dir.mkdir()
    path = ctx.data_path(name)
    path.write_bytes(b"bundle")
    return ctx, path


def test_explicit_bundle_is_only_physical_settlement_and_is_parsed_once(tmp_path):
    ctx, bundle = _bundle_ctx(tmp_path)
    ctx.data_path("2026-01-02-table.xlsx").write_bytes(b"daily-must-be-ignored")
    catalog = discover_inputs(
        ctx, "2026-01-02", "2026-01-05",
        settlement_bundle="data/2026-01-02-2026-01-05-table.xlsx",
    )
    assert not catalog.settlement_sources
    assert catalog.settlement_bundle.display_path == bundle.relative_to(ctx.root).as_posix()
    adapter = MagicMock()
    adapter.parse_settlement.return_value = pd.DataFrame({
        "交收日期": [20260102, 20260102, 20260105],
        "交易类别": ["利息归本", "利息归本", "利息归本"],
    })
    parsed, resolved = parse_pending_sources_with_catalog(catalog, adapter)
    adapter.parse_settlement.assert_called_once_with(catalog.settlement_bundle.path)
    assert [item.trade_date for item in resolved.settlement_sources] == ["2026-01-02", "2026-01-05"]
    assert [item.row_count for item in resolved.settlement_sources] == [2, 1]
    assert all(item.source_kind == "bundle_partition" for item in resolved.settlement_sources)
    assert parsed[("settlement", "2026-01-02")].row_count == 2


def test_bundle_partitions_reuse_the_strict_daily_accounting_contract(tmp_path):
    from quantpits.post_trade.state import normalize_settlement_frame

    ctx, _ = _bundle_ctx(tmp_path)
    catalog = discover_inputs(
        ctx, "2026-01-02", "2026-01-05",
        settlement_bundle="data/2026-01-02-2026-01-05-table.xlsx",
    )
    frame = pd.DataFrame({
        "证券代码": ["000001", "000002"],
        "交易类别": ["深圳A股普通股票竞价买入", "深圳A股普通股票竞价买入"],
        "成交价格": [10, 20], "成交数量": [100, 100],
        "成交金额": [1000, 2000], "资金发生数": [-1000, -2000],
        "交收日期": [20260102, 20260105],
    })
    adapter = MagicMock(); adapter.parse_settlement.return_value = frame
    parsed, _ = parse_pending_sources_with_catalog(catalog, adapter)
    for date in ("2026-01-02", "2026-01-05"):
        logical = parsed[("settlement", date)].dataframe
        events, warnings = normalize_settlement_frame(logical, date)
        assert len(events) == 1 and warnings == ()


def test_unselected_bundle_is_ignored_and_daily_discovery_is_unchanged(tmp_path):
    ctx, _ = _bundle_ctx(tmp_path)
    ctx.data_path("2026-01-02-table.xlsx").write_bytes(b"daily")
    catalog = discover_inputs(ctx, "2026-01-02", "2026-01-05")
    assert catalog.settlement_bundle is None
    assert [item.trade_date for item in catalog.settlement_sources] == ["2026-01-02"]


@pytest.mark.parametrize("bad_date", [None, "", True, [], 20260102.5, "20260102.5", "NaT", "not-a-date", "2026-01-06"])
def test_bundle_rejects_missing_invalid_or_out_of_range_row_dates(tmp_path, bad_date):
    ctx, _ = _bundle_ctx(tmp_path)
    catalog = discover_inputs(
        ctx, "2026-01-02", "2026-01-05",
        settlement_bundle="data/2026-01-02-2026-01-05-table.xlsx",
    )
    adapter = MagicMock()
    adapter.parse_settlement.return_value = pd.DataFrame({"交收日期": [bad_date]})
    with pytest.raises(PostTradeInputError):
        parse_pending_sources_with_catalog(catalog, adapter)


@pytest.mark.parametrize("bad_frame", [None, "not-a-frame"])
def test_bundle_rejects_non_dataframe_parser_results(tmp_path, bad_frame):
    ctx, _ = _bundle_ctx(tmp_path)
    catalog = discover_inputs(
        ctx, "2026-01-02", "2026-01-05",
        settlement_bundle="data/2026-01-02-2026-01-05-table.xlsx",
    )
    adapter = MagicMock(); adapter.parse_settlement.return_value = bad_frame
    with pytest.raises(PostTradeInputError, match="DataFrame"):
        parse_pending_sources_with_catalog(catalog, adapter)


def test_bundle_rejects_duplicate_date_columns(tmp_path):
    ctx, _ = _bundle_ctx(tmp_path)
    catalog = discover_inputs(
        ctx, "2026-01-02", "2026-01-05",
        settlement_bundle="data/2026-01-02-2026-01-05-table.xlsx",
    )
    adapter = MagicMock()
    adapter.parse_settlement.return_value = pd.DataFrame(
        [[20260102, 20260102]], columns=["交收日期", "交收日期"],
    )
    with pytest.raises(PostTradeInputError, match="交收日期"):
        parse_pending_sources_with_catalog(catalog, adapter)


@pytest.mark.parametrize("name", [
    "bundle.xlsx",
    "2026-01-06-2026-01-02-table.xlsx",
])
def test_bundle_filename_is_strict(tmp_path, name):
    ctx, _ = _bundle_ctx(tmp_path, name=name)
    with pytest.raises(PostTradePlanError):
        discover_inputs(
            ctx, "2026-01-02", "2026-01-05",
            settlement_bundle="data/%s" % name,
        )


def test_bundle_rejects_symlink_escape(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path / "workspace")
    ctx.data_dir.mkdir(parents=True)
    outside = tmp_path / "2026-01-02-2026-01-05-table.xlsx"
    outside.write_bytes(b"private")
    ctx.data_path(outside.name).symlink_to(outside)
    with pytest.raises(PostTradePlanError, match="outside"):
        discover_inputs(
            ctx, "2026-01-02", "2026-01-05",
            settlement_bundle="data/%s" % outside.name,
        )
