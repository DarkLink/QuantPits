import json
import pandas as pd
import pytest

from quantpits.post_trade.command import (
    PostTradeRunOptions, execute_prepared, prepare_post_trade_run, render_prepared,
)
from quantpits.post_trade.contracts import (
    PostTradeInputError, PostTradeInputMissingError, PostTradePartialExecutionError,
    PostTradePlanError, SourceChangedError,
)
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
    assert any(state.path == "data/.post_trade.lock" for state in prepared.plan.states)


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


def test_execution_scope_rejects_settlement_bundle(tmp_path):
    ctx = _ctx(tmp_path)
    with pytest.raises(PostTradePlanError, match="state or all"):
        prepare_post_trade_run(ctx, PostTradeRunOptions(
            scope="execution",
            settlement_bundle="data/2026-01-02-2026-01-02-table.xlsx",
        ))


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


def test_state_failure_preserves_committed_ingestion_ledger(tmp_path):
    ctx = _ctx(tmp_path)
    for suffix in ("table", "order", "trade"):
        ctx.data_path("2026-01-02-%s.xlsx" % suffix).write_bytes(suffix.encode())

    class Adapter:
        def parse_settlement(self, _):
            return pd.DataFrame()

        def parse_orders(self, _):
            return pd.DataFrame({"成交数量": [0]})

        def parse_trades(self, _):
            return pd.DataFrame({"成交数量": [0]})

    prepared = prepare_post_trade_run(
        ctx,
        PostTradeRunOptions(scope="all", start_date="2026-01-02", end_date="2026-01-02"),
    )

    with pytest.raises(PostTradePartialExecutionError) as caught:
        execute_prepared(
            prepared, Adapter(), init_qlib=lambda: None,
            resolve_trade_dates=lambda *_: ("2026-01-02",),
            state_callback=lambda *_: (_ for _ in ()).throw(RuntimeError("state failed")),
        )

    summary = caught.value.summary
    assert summary.ingestion is not None
    assert summary.ingestion.ingested_sources
    assert ctx.data_path("post_trade_ingestion_state.json") in summary.ingestion.outputs


def _bundle_frame(*dates):
    return pd.DataFrame({
        "交收日期": [date.replace("-", "") for date in dates],
        "交易类别": ["利息归本"] * len(dates),
    })


def test_bundle_coverage_uses_resolved_trading_dates_not_cursor_calendar_days(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.config_path("prod_config.json").write_text(json.dumps({
        "current_date": "2026-07-10", "last_processed_date": "2026-07-10",
        "current_cash": 123456, "current_holding": [], "broker": "gtja",
    }))
    ctx.data_path("2026-07-13-2026-07-17-table.xlsx").write_bytes(b"bundle")

    class Adapter:
        def parse_settlement(self, _):
            return _bundle_frame(
                "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16",
            )

    prepared = prepare_post_trade_run(ctx, PostTradeRunOptions(
        scope="state", end_date="2026-07-17", allow_missing_settlement=True,
        settlement_bundle="data/2026-07-13-2026-07-17-table.xlsx",
    ))
    summary = execute_prepared(
        prepared, Adapter(), init_qlib=lambda: None,
        resolve_trade_dates=lambda *_: (
            "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17",
        ),
    )
    assert summary.prepared.catalog.date_from == "2026-07-11"
    assert summary.prepared.catalog.source_for_date(
        "settlement", "2026-07-17",
    ).status == "assumed_empty"


@pytest.mark.parametrize("name", [
    "2026-01-03-2026-01-05-table.xlsx",
    "2026-01-02-2026-01-04-table.xlsx",
])
def test_bundle_coverage_must_cover_resolved_trading_dates(tmp_path, name):
    ctx = _ctx(tmp_path)
    ctx.data_path(name).write_bytes(b"bundle")
    prepared = prepare_post_trade_run(ctx, PostTradeRunOptions(
        scope="state", end_date="2026-01-05",
        settlement_bundle="data/%s" % name,
    ))

    with pytest.raises(PostTradeInputError, match="resolved trading dates"):
        execute_prepared(
            prepared, object(), init_qlib=lambda: None,
            resolve_trade_dates=lambda *_: ("2026-01-02", "2026-01-05"),
        )


def test_bundle_plan_records_one_physical_input_and_strict_logical_evidence(tmp_path):
    ctx = _ctx(tmp_path)
    bundle = ctx.data_path("2026-01-02-2026-01-05-table.xlsx")
    bundle.write_bytes(b"bundle")
    ctx.data_path("2026-01-02-table.xlsx").write_bytes(b"daily-ignored")

    class Adapter:
        calls = 0
        def parse_settlement(self, _):
            self.calls += 1
            return _bundle_frame("2026-01-02", "2026-01-02", "2026-01-05")

    prepared = prepare_post_trade_run(ctx, PostTradeRunOptions(
        scope="state", end_date="2026-01-05", dry_run=True,
        allow_missing_settlement=True,
        settlement_bundle="data/2026-01-02-2026-01-05-table.xlsx",
    ))
    physical = [item for item in prepared.plan.inputs if item.description == "settlement_bundle"]
    assert len(physical) == 1
    assert json.dumps(prepared.plan.to_public_dict()).count(physical[0].fingerprint) == 1
    assert not [item for item in prepared.plan.inputs if item.description == "settlement"]
    before = {path.relative_to(ctx.root): path.read_bytes() for path in ctx.root.rglob("*") if path.is_file()}
    adapter = Adapter()
    summary = execute_prepared(
        prepared, adapter, init_qlib=lambda: None,
        resolve_trade_dates=lambda *_: ("2026-01-02", "2026-01-05"),
    )
    assert adapter.calls == 1
    logical = summary.prepared.plan.metadata["settlement_logical_partitions"]
    assert logical["2026-01-02"]["row_count"] == 2
    assert logical["2026-01-05"]["row_count"] == 1
    assert all(item["status"] == "observed" for item in logical.values())
    assert summary.prepared.plan.metadata["source_status_counts"]["settlement"]["present"] == 1
    assert not ctx.data_path("2026-01-05-table.xlsx").exists()
    after = {path.relative_to(ctx.root): path.read_bytes() for path in ctx.root.rglob("*") if path.is_file()}
    assert after == before


def test_bundle_missing_trading_day_requires_explicit_acknowledgement(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.data_path("2026-01-02-2026-01-05-table.xlsx").write_bytes(b"bundle")

    class Adapter:
        def parse_settlement(self, _):
            return _bundle_frame("2026-01-02")

    prepared = prepare_post_trade_run(ctx, PostTradeRunOptions(
        scope="state", end_date="2026-01-05",
        settlement_bundle="data/2026-01-02-2026-01-05-table.xlsx",
    ))
    with pytest.raises(PostTradeInputMissingError, match="2026-01-05"):
        execute_prepared(
            prepared, Adapter(), init_qlib=lambda: None,
            resolve_trade_dates=lambda *_: ("2026-01-02", "2026-01-05"),
        )

    acknowledged = prepare_post_trade_run(ctx, PostTradeRunOptions(
        scope="state", end_date="2026-01-05", allow_missing_settlement=True,
        settlement_bundle="data/2026-01-02-2026-01-05-table.xlsx",
    ))
    summary = execute_prepared(
        acknowledged, Adapter(), init_qlib=lambda: None,
        resolve_trade_dates=lambda *_: ("2026-01-02", "2026-01-05"),
    )
    missing = summary.prepared.catalog.source_for_date("settlement", "2026-01-05")
    assert missing.status == "assumed_empty"
    assert missing.row_count == 0
    logical = summary.prepared.plan.metadata["settlement_logical_partitions"]
    assert logical["2026-01-05"] == {
        "status": "assumed_empty", "row_count": 0, "fingerprint": None,
    }


def test_bundle_rejects_non_trading_logical_partition(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.data_path("2026-01-02-2026-01-05-table.xlsx").write_bytes(b"bundle")

    class Adapter:
        def parse_settlement(self, _):
            return _bundle_frame("2026-01-03")

    prepared = prepare_post_trade_run(ctx, PostTradeRunOptions(
        scope="state", end_date="2026-01-05", allow_missing_settlement=True,
        settlement_bundle="data/2026-01-02-2026-01-05-table.xlsx",
    ))
    with pytest.raises(PostTradeInputError, match="trading calendar"):
        execute_prepared(
            prepared, Adapter(), init_qlib=lambda: None,
            resolve_trade_dates=lambda *_: ("2026-01-02", "2026-01-05"),
        )


def test_bundle_drift_after_single_parse_fails_before_callback(tmp_path):
    ctx = _ctx(tmp_path)
    bundle = ctx.data_path("2026-01-02-2026-01-02-table.xlsx")
    bundle.write_bytes(b"bundle")

    class MutatingAdapter:
        def parse_settlement(self, _):
            bundle.write_bytes(b"replacement")
            return _bundle_frame("2026-01-02")

    prepared = prepare_post_trade_run(ctx, PostTradeRunOptions(
        scope="state", end_date="2026-01-02",
        settlement_bundle="data/2026-01-02-2026-01-02-table.xlsx",
    ))
    called = []
    with pytest.raises(SourceChangedError):
        execute_prepared(
            prepared, MutatingAdapter(), init_qlib=lambda: None,
            resolve_trade_dates=lambda *_: ("2026-01-02",),
            state_callback=lambda *_: called.append(True),
        )
    assert called == []


def test_bundle_same_bytes_namespace_replacement_is_source_drift(tmp_path):
    ctx = _ctx(tmp_path)
    bundle = ctx.data_path("2026-01-02-2026-01-02-table.xlsx")
    bundle.write_bytes(b"bundle")

    class ReplacingAdapter:
        def parse_settlement(self, _):
            bundle.unlink()
            bundle.write_bytes(b"bundle")
            return _bundle_frame("2026-01-02")

    prepared = prepare_post_trade_run(ctx, PostTradeRunOptions(
        scope="state", end_date="2026-01-02",
        settlement_bundle="data/2026-01-02-2026-01-02-table.xlsx",
    ))
    with pytest.raises(SourceChangedError):
        execute_prepared(
            prepared, ReplacingAdapter(), init_qlib=lambda: None,
            resolve_trade_dates=lambda *_: ("2026-01-02",),
        )
