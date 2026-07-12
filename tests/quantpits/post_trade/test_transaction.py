import json
from dataclasses import replace

import pytest

from quantpits.post_trade.contracts import (
    PostTradeConcurrentRunError, PostTradeTransactionConflictError,
)
from quantpits.post_trade.transaction import PostTradeTransactionManager, sha256_file
from quantpits.utils.workspace import WorkspaceContext


def _manager(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    return PostTradeTransactionManager(WorkspaceContext.from_root(root))


def _prepare(manager):
    root = manager.ctx.root
    (root / "data" / "a.csv").write_bytes(b"old")
    return manager.prepare(
        transaction_id="run-1", run_id="run-1", scope="state", light_fingerprint="light",
        resolved_fingerprint="resolved", cursor_before="2026-01-01", cursor_after="2026-01-02",
        processed_dates=("2026-01-02",), consumed_cashflow_dates=(),
        payloads=((100, "trade_log", root / "data" / "a.csv", b"new"),
                  (500, "prod_config_cursor", root / "config" / "prod_config.json", b"{}\n")),
    )


def test_commit_and_recovery_recognize_already_replaced_target(tmp_path):
    manager = _manager(tmp_path); journal = _prepare(manager)
    first = journal.artifacts[0]
    (manager.ctx.root / first.path).write_bytes((manager.ctx.root / first.staged_path).read_bytes())
    recovered = manager.recover(journal.transaction_id)
    assert recovered.status == "state_committed"
    assert (manager.ctx.root / "data/a.csv").read_bytes() == b"new"
    assert (manager.ctx.root / "config/prod_config.json").read_bytes() == b"{}\n"


def test_third_version_is_never_overwritten(tmp_path):
    manager = _manager(tmp_path); journal = _prepare(manager)
    (manager.ctx.root / "data/a.csv").write_bytes(b"operator edit")
    with pytest.raises(PostTradeTransactionConflictError): manager.commit(journal)
    assert (manager.ctx.root / "data/a.csv").read_bytes() == b"operator edit"
    assert manager.load("run-1").status == "conflicted"


def test_status_summary_is_read_only_and_redacted(tmp_path):
    manager = _manager(tmp_path); _prepare(manager)
    before = {p: p.read_bytes() for p in manager.ctx.root.rglob("*") if p.is_file()}
    summary = manager.status_summary()
    after = {p: p.read_bytes() for p in manager.ctx.root.rglob("*") if p.is_file()}
    assert summary[0]["transaction_id"] == "run-1"
    assert "cursor_before" not in summary[0]
    assert before == after


def test_state_committed_pending_classification_remains_active(tmp_path):
    manager = _manager(tmp_path)
    committed = manager.commit(_prepare(manager))
    assert committed.status == "state_committed"
    assert [item.transaction_id for item in manager.active()] == ["run-1"]

    completed = manager.complete(committed)
    assert completed.status == "completed"
    assert manager.active() == ()


def test_recover_committed_transaction_verifies_all_targets(tmp_path):
    manager = _manager(tmp_path)
    committed = manager.commit(_prepare(manager))
    (manager.ctx.root / "data/a.csv").write_bytes(b"third-party-change")

    with pytest.raises(PostTradeTransactionConflictError, match="artifact changed"):
        manager.recover(committed.transaction_id)


def test_workspace_lock_rejects_concurrent_real_run(tmp_path):
    manager = _manager(tmp_path)
    with manager.lock():
        with pytest.raises(PostTradeConcurrentRunError):
            with manager.lock():
                pass


def test_classification_output_fingerprint_round_trips(tmp_path):
    manager = _manager(tmp_path)
    journal = manager.commit(_prepare(manager))
    output = manager.ctx.data_path("trade_classification.csv")
    output.write_bytes(b"classification\n")
    display = output.relative_to(manager.ctx.root).as_posix()
    updated = manager.set_classification(
        journal, status="success", output_paths=(display,),
        output_fingerprints=((display, sha256_file(output)),),
    )
    loaded = manager.load(updated.transaction_id)
    assert loaded.classification.output_fingerprints == ((display, sha256_file(output)),)


def test_verified_target_paths_report_only_actual_target_versions(tmp_path):
    manager = _manager(tmp_path)
    journal = _prepare(manager)
    first = journal.artifacts[0]
    (manager.ctx.root / first.path).write_bytes(
        (manager.ctx.root / first.staged_path).read_bytes()
    )
    assert manager.verified_target_paths(journal) == (first.path,)
