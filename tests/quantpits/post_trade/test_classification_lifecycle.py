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
