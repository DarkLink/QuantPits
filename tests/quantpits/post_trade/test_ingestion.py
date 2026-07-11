import json

import pandas as pd
import pytest

from quantpits.post_trade.contracts import ParsedPostTradeInput
from quantpits.post_trade import ingestion
from quantpits.post_trade.contracts import IngestionPersistenceError
from quantpits.post_trade.ingestion import ingest_execution_evidence
from quantpits.post_trade.intake import discover_inputs, parse_pending_sources
from quantpits.utils.workspace import WorkspaceContext


class Adapter:
    def parse_orders(self, path):
        return pd.DataFrame({"证券代码": ["000001"], "委托数量": [1]})
    def parse_trades(self, path):
        return pd.DataFrame({"证券代码": ["000001"], "成交数量": [1]})


def test_ingestion_uses_source_receipts_and_is_idempotent(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path); ctx.data_dir.mkdir()
    ctx.data_path("2020-01-01-order.xlsx").write_bytes(b"old-order")
    ctx.data_path("2020-01-01-trade.xlsx").write_bytes(b"old-trade")
    catalog = discover_inputs(ctx, "1900-01-01", "2026-01-01")
    parsed = parse_pending_sources(catalog, Adapter())
    result = ingest_execution_evidence(ctx, parsed, run_id="test")
    assert result.max_trade_date == "2020-01-01"
    ledger = json.loads(ctx.data_path("post_trade_ingestion_state.json").read_text())
    assert len(ledger["sources"]) == 2
    assert pd.read_csv(ctx.data_path("raw_order_log_full.csv"), dtype={"证券代码": str})["证券代码"].tolist() == ["000001"]
    second_catalog = discover_inputs(ctx, "1900-01-01", "2026-01-01")
    assert {x.status for x in second_catalog.order_sources + second_catalog.trade_sources} == {"already_ingested"}


def test_partial_raw_write_keeps_ledger_uncommitted_and_rerun_converges(tmp_path, monkeypatch):
    ctx = WorkspaceContext.from_root(tmp_path); ctx.data_dir.mkdir()
    ctx.data_path("2020-01-01-order.xlsx").write_bytes(b"old-order")
    ctx.data_path("2020-01-01-trade.xlsx").write_bytes(b"old-trade")
    parsed = parse_pending_sources(discover_inputs(ctx, "1900-01-01", "2026-01-01"), Adapter())
    original = ingestion._atomic_bytes
    calls = {"raw": 0}

    def fail_second_raw(path, payload):
        if path.suffix == ".csv":
            calls["raw"] += 1
            if calls["raw"] == 2:
                raise OSError("second raw writer failed")
        return original(path, payload)

    monkeypatch.setattr(ingestion, "_atomic_bytes", fail_second_raw)
    with pytest.raises(IngestionPersistenceError):
        ingest_execution_evidence(ctx, parsed, run_id="failed")
    assert not ctx.data_path("post_trade_ingestion_state.json").exists()

    monkeypatch.setattr(ingestion, "_atomic_bytes", original)
    ingest_execution_evidence(ctx, parsed, run_id="retry")
    orders = pd.read_csv(ctx.data_path("raw_order_log_full.csv"), dtype={"证券代码": str})
    assert len(orders) == 1


def test_legacy_cursor_failure_does_not_rollback_authoritative_ledger(tmp_path, monkeypatch):
    ctx = WorkspaceContext.from_root(tmp_path); ctx.data_dir.mkdir()
    ctx.data_path("2020-01-01-order.xlsx").write_bytes(b"old-order")
    parsed = parse_pending_sources(discover_inputs(ctx, "1900-01-01", "2026-01-01"), Adapter())
    original = ingestion._atomic_bytes

    def fail_legacy(path, payload):
        if path.name == ".order_trade_state.json":
            raise OSError("legacy mirror unavailable")
        return original(path, payload)

    monkeypatch.setattr(ingestion, "_atomic_bytes", fail_legacy)
    with pytest.warns(RuntimeWarning, match="legacy cursor mirror"):
        result = ingest_execution_evidence(ctx, parsed, run_id="test")
    assert ctx.data_path("post_trade_ingestion_state.json") in result.outputs
