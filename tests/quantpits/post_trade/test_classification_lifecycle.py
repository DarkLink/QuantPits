from quantpits.post_trade.transaction import PostTradeTransactionManager
from quantpits.utils.workspace import WorkspaceContext


def _committed(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    manager = PostTradeTransactionManager(WorkspaceContext.from_root(root))
    journal = manager.prepare(
        transaction_id="classification-run", run_id="classification-run",
        scope="state", light_fingerprint="light", resolved_fingerprint="resolved",
        cursor_before="2026-01-01", cursor_after="2026-01-02",
        processed_dates=("2026-01-02",), consumed_cashflow_dates=(),
        payloads=((500, "prod_config_cursor", root / "config/prod_config.json", b"{}\n"),),
    )
    return manager, manager.commit(journal)


def test_stale_running_classification_remains_recoverable(tmp_path):
    manager, journal = _committed(tmp_path)
    running = manager.set_classification(journal, status="running")
    assert running.classification.attempts == 1
    assert manager.active()[0].transaction_id == running.transaction_id
    recovered = manager.recover(running.transaction_id)
    assert recovered.classification.status == "running"
    assert recovered.classification.attempts == 1


def test_failed_classification_can_close_authoritative_state(tmp_path):
    manager, journal = _committed(tmp_path)
    running = manager.set_classification(journal, status="running")
    failed = manager.set_classification(running, status="failed", error="redacted")
    completed = manager.complete(failed)
    assert completed.status == "completed"
    assert completed.classification.status == "failed"
    assert manager.active() == ()


def test_real_suggestion_conflict_fails_derived_step_after_account_commit(tmp_path):
    from quantpits.order.persistence import OrderPersistenceRequest, persist_order_artifacts
    from quantpits.scripts.prod_post_trade import _run_classification
    manager, journal = _committed(tmp_path)
    ctx = manager.ctx
    account_bytes = ctx.config_path("prod_config.json").read_bytes()
    ctx.data_path("trade_log_full.csv").write_text(
        "成交日期,证券代码,交易类别,成交数量\n"
        "2026-01-02,SZ000002,深圳A股普通股票竞价买入,100\n")
    classification = ctx.data_path("trade_classification.csv")
    # An existing manual edit must survive an unresolved new classification.
    classification.write_text("trade_date,instrument,trade_class\n2026-01-01,SZ000099,M\n")
    saved_classification = classification.read_bytes()
    ledger = persist_order_artifacts(OrderPersistenceRequest(
        ctx=ctx, output_dir=ctx.root / "output/cycles/p/week/orders",
        trade_date="2026-01-02", source_label="ensemble", opinions=None,
        buy_orders=({"instrument": "SZ000001"}, {"instrument": "SZ000002"}),
        sell_orders=(),
    ))
    other = ctx.root / "output/buy_suggestion_other_20260102.csv"
    other.write_text("instrument\nSZ000002\n")
    failed, warning = _run_classification(ctx, manager, journal)
    assert "SuggestionConflictError" in warning
    assert failed.status == "state_committed"
    assert failed.classification.status == "failed"
    assert failed.classification.output_paths == ()
    assert classification.read_bytes() == saved_classification
    assert ctx.config_path("prod_config.json").read_bytes() == account_bytes
    assert (ctx.root / ledger.buy_csv).is_file()
    manager.verify_committed(failed)
    completed = manager.complete(failed)
    assert completed.status == "completed"
    assert completed.classification.status == "failed"
    assert manager.active() == ()
