import json
import pandas as pd
import pytest

from quantpits.post_trade.command import (
    PostTradeRunOptions, execute_prepared, prepare_post_trade_run, render_prepared,
)
from quantpits.post_trade.contracts import PostTradePlanError, SourceChangedError
from quantpits.utils.workspace import WorkspaceContext


def _ctx(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    ctx.config_dir.mkdir(); ctx.data_dir.mkdir()
    ctx.config_path("prod_config.json").write_text(json.dumps({
        "current_date": "2026-01-01", "last_processed_date": "2026-01-01",
        "current_cash": 123456, "current_holding": [{"instrument": "SECRET"}], "broker": "gtja",
    }))
    ctx.config_path("cashflow.json").write_text(json.dumps({"cashflows": {"2026-01-02": 99}}))
    return ctx


def test_plan_is_light_redacted_and_execution_scans_late_files(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.data_path("2020-01-01-order.xlsx").write_bytes(b"late")
    prepared = prepare_post_trade_run(ctx, PostTradeRunOptions(scope="execution", json_plan=True, run_id="volatile"))
    public = render_prepared(prepared)
    assert "2020-01-01-order.xlsx" in public
    assert "123456" not in public and "SECRET" not in public
    assert prepared.catalog.date_from == "1900-01-01"


def test_plan_fingerprint_ignores_run_id(tmp_path):
    ctx = _ctx(tmp_path)
    one = prepare_post_trade_run(ctx, PostTradeRunOptions(scope="all", run_id="one"))
    two = prepare_post_trade_run(ctx, PostTradeRunOptions(scope="all", run_id="two"))
    assert one.plan_fingerprint == two.plan_fingerprint

    one_cli = prepare_post_trade_run(ctx, PostTradeRunOptions(scope="all", run_id="one"), cli_args=("--run-id", "one"))
    two_cli = prepare_post_trade_run(ctx, PostTradeRunOptions(scope="all", run_id="two"), cli_args=("--run-id", "two"))
    assert one_cli.plan_fingerprint == two_cli.plan_fingerprint


def test_all_scope_bootstraps_execution_but_bounds_settlement_to_state_cursor(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.data_path("2020-01-01-order.xlsx").write_bytes(b"old-order")
    ctx.data_path("2020-01-01-trade.xlsx").write_bytes(b"old-trade")
    ctx.data_path("2020-01-01-table.xlsx").write_bytes(b"old-settlement")
    ctx.data_path("2026-01-02-table.xlsx").write_bytes(b"new-settlement")

    prepared = prepare_post_trade_run(
        ctx, PostTradeRunOptions(scope="all", end_date="2026-01-02")
    )

    assert [item.trade_date for item in prepared.catalog.settlement_sources] == ["2026-01-02"]
    assert [item.trade_date for item in prepared.catalog.order_sources] == ["2020-01-01"]
    assert prepared.catalog.date_from == "2026-01-02"
    assert prepared.execution_options.start_date == "2026-01-02"


def test_state_scope_rejects_historical_replay(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(PostTradePlanError, match="historical state replay"):
        prepare_post_trade_run(
            ctx,
            PostTradeRunOptions(
                scope="all", start_date="2025-12-31", end_date="2026-01-02"
            ),
        )


def test_plan_exposes_stable_source_status_counts(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.data_path("2020-01-01-order.xlsx").write_bytes(b"order")
    prepared = prepare_post_trade_run(ctx, PostTradeRunOptions(scope="execution"))
    counts = prepared.plan.metadata["source_status_counts"]
    assert counts["order"]["present"] == 1
    assert set(counts["order"]) == {
        "present", "missing", "assumed_empty", "already_ingested", "changed"
    }


def test_scope_catalogs_only_include_owned_streams(tmp_path):
    ctx = _ctx(tmp_path)
    for suffix in ("table", "order", "trade"):
        ctx.data_path("2026-01-02-%s.xlsx" % suffix).write_bytes(suffix.encode())
    state = prepare_post_trade_run(
        ctx, PostTradeRunOptions(scope="state", end_date="2026-01-02")
    )
    execution = prepare_post_trade_run(
        ctx, PostTradeRunOptions(scope="execution", end_date="2026-01-02")
    )
    assert state.catalog.settlement_sources
    assert not state.catalog.order_sources and not state.catalog.trade_sources
    assert not execution.catalog.settlement_sources
    assert execution.catalog.order_sources and execution.catalog.trade_sources


def test_allow_missing_settlement_changes_semantic_fingerprint(tmp_path):
    ctx = _ctx(tmp_path)
    strict = prepare_post_trade_run(ctx, PostTradeRunOptions(scope="all"))
    acknowledged = prepare_post_trade_run(
        ctx, PostTradeRunOptions(scope="all", allow_missing_settlement=True)
    )
    assert strict.plan_fingerprint != acknowledged.plan_fingerprint


def test_dry_run_is_byte_stable_and_detects_pre_writer_source_drift(tmp_path):
    ctx = _ctx(tmp_path)
    order = ctx.data_path("2026-01-02-order.xlsx"); order.write_bytes(b"order")
    trade = ctx.data_path("2026-01-02-trade.xlsx"); trade.write_bytes(b"trade")
    sentinel = ctx.data_path("raw_order_log_full.csv")
    sentinel.write_bytes(b"sentinel")

    class StableAdapter:
        def parse_orders(self, _):
            return pd.DataFrame({"成交数量": [0]})
        def parse_trades(self, _):
            return pd.DataFrame({"成交数量": []})

    prepared = prepare_post_trade_run(
        ctx,
        PostTradeRunOptions(
            scope="execution", start_date="2026-01-02",
            end_date="2026-01-02", dry_run=True,
        ),
    )
    before = sentinel.read_bytes()
    execute_prepared(prepared, StableAdapter())
    assert sentinel.read_bytes() == before
    assert not ctx.data_path("post_trade_ingestion_state.json").exists()

    class MutatingAdapter(StableAdapter):
        def parse_orders(self, _):
            frame = super().parse_orders(_)
            order.write_bytes(b"changed-after-parse")
            return frame

    real = prepare_post_trade_run(
        ctx,
        PostTradeRunOptions(
            scope="execution", start_date="2026-01-02", end_date="2026-01-02"
        ),
    )
    with pytest.raises(SourceChangedError):
        execute_prepared(real, MutatingAdapter())
    assert sentinel.read_bytes() == before
