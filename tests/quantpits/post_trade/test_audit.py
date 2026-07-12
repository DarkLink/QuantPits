import json

from quantpits.post_trade.audit import write_post_trade_manifest
from quantpits.post_trade.command import (
    PostTradeRunOptions, PostTradeRunSummary, prepare_post_trade_run,
)
from quantpits.post_trade.transaction import PostTradeTransactionManager
from quantpits.utils.workspace import WorkspaceContext


def _prepared(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path / "Demo_Workspace")
    ctx.config_dir.mkdir(parents=True)
    ctx.data_dir.mkdir()
    ctx.config_path("prod_config.json").write_text(json.dumps({
        "current_date": "2026-01-01", "current_cash": 123456,
        "current_holding": [{"instrument": "PRIVATE"}], "broker": "gtja",
    }))
    ctx.config_path("cashflow.json").write_text("{}")
    prepared = prepare_post_trade_run(
        ctx, PostTradeRunOptions(scope="state", run_id="audit-run"),
    )
    return ctx, prepared


def test_failure_manifest_reports_only_verified_transaction_targets(tmp_path):
    ctx, prepared = _prepared(tmp_path)
    old = ctx.data_path("trade_log_full.csv")
    old.write_bytes(b"old")
    manager = PostTradeTransactionManager(ctx)
    journal = manager.prepare(
        transaction_id="audit-run", run_id="audit-run", scope="state",
        light_fingerprint=prepared.plan_fingerprint,
        resolved_fingerprint="resolved", cursor_before="2026-01-01",
        cursor_after="2026-01-02", processed_dates=("2026-01-02",),
        consumed_cashflow_dates=(), payloads=(
            (100, "trade_log", old, b"new"),
            (500, "prod_config_cursor", ctx.config_path("prod_config.json"), b"{}\n"),
        ),
    )
    first = journal.artifacts[0]
    (ctx.root / first.path).write_bytes((ctx.root / first.staged_path).read_bytes())
    actual = manager.verified_target_paths(journal)

    manifest_path, _ = write_post_trade_manifest(
        prepared, PostTradeRunSummary(prepared), started_at="2026-01-02T00:00:00",
        status="failed", error={"type": "InjectedFailure", "message": "stopped"},
        journal=journal, actual_state_paths=actual,
    )
    payload = json.loads(manifest_path.read_text())
    output_paths = {item["path"] for item in payload["outputs"]}
    assert "data/trade_log_full.csv" in output_paths
    assert "config/prod_config.json" not in output_paths
    rendered = json.dumps(payload)
    assert "123456" not in rendered
    assert "PRIVATE" not in rendered
