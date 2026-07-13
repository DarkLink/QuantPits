from decimal import Decimal

from quantpits.post_trade.account_snapshot import BrokerAccountSnapshot, BrokerPositionObservation
from quantpits.post_trade.state import AccountState, Position, ValuationSnapshot
from quantpits.tools import reconcile_post_trade_account as tool


def test_reconciliation_cli_default_is_stdout_only(tmp_path, monkeypatch, capsys):
    root = tmp_path / "Demo_Workspace"
    (root / "data").mkdir(parents=True)
    snapshot_file = root / "data" / "example-asset.xlsx"
    snapshot_file.write_bytes(b"synthetic")
    state = AccountState("2026-01-02", Decimal("10"), (Position("SH000001", Decimal("2"), Decimal("10")),))
    valuation = ValuationSnapshot("2026-01-02", (("SH000001", Decimal("10")),), Decimal("20"))
    snapshot = BrokerAccountSnapshot(
        "gtja", snapshot_file.name, "a" * 64, "2026-01-03T00:00:00", "2026-01-02", "2026-01-02",
        Decimal("10"), Decimal("20"), Decimal("30"),
        (BrokerPositionObservation("SH000001", Decimal("2"), None, Decimal("10"), Decimal("20")),),
    )
    monkeypatch.setattr(tool, "_load_state", lambda ctx, date: state)
    monkeypatch.setattr(tool, "_load_valuation", lambda ctx, date: valuation)
    monkeypatch.setattr(tool.GtjaAdapter, "parse_account_snapshot", lambda self, path, **kwargs: snapshot)
    before = tuple(root.rglob("*"))
    code = tool.run(tool.build_parser().parse_args([
        "--workspace", str(root), "--broker", "gtja", "--snapshot-file", "data/example-asset.xlsx",
        "--account-date", "2026-01-02", "--snapshot-effective-date", "2026-01-02",
        "--snapshot-market-date", "2026-01-02", "--json",
    ]))
    assert code == 0
    assert '"analytics_eligibility": "eligible"' in capsys.readouterr().out
    assert tuple(root.rglob("*")) == before


def test_snapshot_path_must_be_workspace_contained(tmp_path):
    root = tmp_path / "Demo_Workspace"; root.mkdir()
    ctx = tool.WorkspaceContext.from_root(root)
    try:
        tool._contained(ctx, tmp_path / "external.xlsx")
    except ValueError:
        pass
    else:
        raise AssertionError("external snapshot path was accepted")


def test_reconciliation_dates_and_run_id_are_strict():
    import pytest
    with pytest.raises(ValueError):
        tool._validated_date("2026-02-30", "account_date", required=True)
    with pytest.raises(ValueError):
        tool._validated_date("2026-1-02", "account_date", required=True)
    with pytest.raises(ValueError):
        tool._validated_run_id("../escape")
