from decimal import Decimal

import pytest

from quantpits.post_trade.account_snapshot import (
    BrokerAccountSnapshot, BrokerPositionObservation, validate_snapshot,
)
from quantpits.post_trade.contracts import BrokerAccountSnapshotError
from quantpits.scripts.brokers.gtja import GtjaAdapter


def test_snapshot_internal_totals_are_validated_without_private_fields():
    snapshot = BrokerAccountSnapshot(
        "gtja", "asset.xlsx", "a" * 64, "2026-01-03T00:00:00", None, None,
        Decimal("10"), Decimal("20"), Decimal("30"),
        (BrokerPositionObservation("SH000001", Decimal("2"), None, Decimal("10"), Decimal("20")),),
    )
    assert validate_snapshot(snapshot) is snapshot
    assert "account" not in str(snapshot.to_public_dict()).lower()


def test_snapshot_rejects_inconsistent_totals():
    snapshot = BrokerAccountSnapshot(
        "gtja", "asset.xlsx", "a" * 64, "2026-01-03T00:00:00", None, None,
        Decimal("10"), Decimal("20"), Decimal("31"), (),
    )
    with pytest.raises(BrokerAccountSnapshotError): validate_snapshot(snapshot)


def test_gtja_snapshot_parser_discards_private_headers(tmp_path):
    import pandas as pd
    path = tmp_path / "synthetic-asset.xlsx"
    rows = [
        ["客户姓名", "TEST ONLY"], ["资金账号", "000000"],
        ["查询时间", "2026-01-03 10:00:00"], ["可用资金", 10],
        ["证券市值", 20], ["总资产", 30],
        ["证券代码", "证券数量", "可用数量", "现价", "市值", "备注"],
        ["600001", 2, 2, 10, 20, ""],
    ]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    snapshot = GtjaAdapter().parse_account_snapshot(
        path, effective_date="2026-01-02", market_date="2026-01-02",
    )
    rendered = str(snapshot.to_public_dict())
    assert snapshot.positions[0].instrument == "SH600001"
    assert "TEST ONLY" not in rendered and "000000" not in rendered
