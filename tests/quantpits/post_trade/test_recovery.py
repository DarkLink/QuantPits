import pytest

from quantpits.post_trade.transaction import PostTradeTransactionManager
from quantpits.utils.workspace import WorkspaceContext


def test_recovery_converges_after_replace_before_journal_update(tmp_path, monkeypatch):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    target = root / "data/state.csv"
    target.write_bytes(b"baseline")
    manager = PostTradeTransactionManager(WorkspaceContext.from_root(root))
    journal = manager.prepare(
        transaction_id="crash-run", run_id="crash-run", scope="state",
        light_fingerprint="light", resolved_fingerprint="resolved",
        cursor_before="2026-01-01", cursor_after="2026-01-02",
        processed_dates=("2026-01-02",), consumed_cashflow_dates=(),
        payloads=(
            (100, "trade_log", target, b"target"),
            (500, "prod_config_cursor", root / "config/prod_config.json", b"{}\n"),
        ),
    )

    original_write = manager.write_journal
    calls = {"count": 0}

    def crash_after_first_replace(value):
        calls["count"] += 1
        # commit writes `committing`, replaces the first target, then attempts
        # to record it. Simulate process death at that journal boundary.
        if calls["count"] == 2:
            raise OSError("injected journal interruption")
        return original_write(value)

    monkeypatch.setattr(manager, "write_journal", crash_after_first_replace)
    with pytest.raises(OSError, match="injected journal interruption"):
        manager.commit(journal)
    assert target.read_bytes() == b"target"
    assert manager.load("crash-run").committed_artifacts == ()

    monkeypatch.setattr(manager, "write_journal", original_write)
    recovered = manager.recover("crash-run")
    assert recovered.status == "state_committed"
    assert target.read_bytes() == b"target"
    assert (root / "config/prod_config.json").read_bytes() == b"{}\n"


@pytest.mark.parametrize("crash_index", range(7))
def test_every_state_artifact_recovers_after_replace_before_journal_update(tmp_path, monkeypatch, crash_index):
    root = tmp_path / ("Demo_Workspace_%s" % crash_index)
    (root / "config").mkdir(parents=True); (root / "data").mkdir()
    manager = PostTradeTransactionManager(WorkspaceContext.from_root(root))
    specs = (
        (10, "trade_detail", root / "data/detail.csv", b"detail"),
        (100, "trade_log", root / "data/trade.csv", b"trade"),
        (200, "holding_log", root / "data/holding.csv", b"holding"),
        (300, "daily_log", root / "data/daily.csv", b"daily"),
        (350, "valuation_evidence", root / "data/valuation_evidence.jsonl", b"evidence"),
        (400, "cashflow_config", root / "config/cashflow.json", b"cashflow"),
        (500, "prod_config_cursor", root / "config/prod_config.json", b"cursor"),
    )
    journal = manager.prepare(
        transaction_id="crash-run", run_id="crash-run", scope="state",
        light_fingerprint="light", resolved_fingerprint="resolved",
        cursor_before="2026-01-01", cursor_after="2026-01-02",
        processed_dates=("2026-01-02",), consumed_cashflow_dates=(), payloads=specs,
    )
    original_write = manager.write_journal; calls = {"count": 0}
    def injected(value):
        calls["count"] += 1
        if calls["count"] == 2 + crash_index:
            raise OSError("injected artifact boundary")
        return original_write(value)
    monkeypatch.setattr(manager, "write_journal", injected)
    with pytest.raises(OSError, match="artifact boundary"):
        manager.commit(journal)
    monkeypatch.setattr(manager, "write_journal", original_write)
    recovered = manager.recover("crash-run")
    assert recovered.status == "state_committed"
    for _, _, path, payload in specs:
        assert path.read_bytes() == payload
