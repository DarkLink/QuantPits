from decimal import Decimal

from quantpits.post_trade.account_snapshot import BrokerAccountSnapshot, BrokerPositionObservation
from quantpits.post_trade.state import AccountState, Position, ValuationSnapshot
from quantpits.post_trade.valuation_reconciliation import reconcile_account


def _inputs(status="none", market_date="2026-01-02"):
    state = AccountState("2026-01-02", Decimal("100"), (Position("SH000001", Decimal("10"), Decimal("80")),))
    valuation = ValuationSnapshot("2026-01-02", (("SH000001", Decimal("10")),), Decimal("20"))
    snapshot = BrokerAccountSnapshot(
        "gtja", "asset.xlsx", "a" * 64, "2026-01-03T10:00:00", "2026-01-02",
        market_date, Decimal("100"), Decimal("100"), Decimal("200"),
        (BrokerPositionObservation("SH000001", Decimal("10"), Decimal("10"), Decimal("10"), Decimal("100"), status),),
    )
    return state, valuation, snapshot


def test_comparable_account_reconciles():
    report = reconcile_account(*_inputs())
    assert report.analytics_eligibility == "eligible"
    assert report.nav.status == "matched"


def test_adjusted_in_advance_blocks_aggregate_nav():
    report = reconcile_account(*_inputs(status="adjusted_in_advance"))
    assert report.price_comparability == "not_comparable"
    assert report.nav.status == "not_comparable"
    assert report.analytics_eligibility == "review_required"


def test_missing_market_date_is_not_a_numeric_mismatch():
    report = reconcile_account(*_inputs(market_date=None))
    assert report.prices[0].reason_code == "missing_market_date"
    assert report.nav.difference is None
