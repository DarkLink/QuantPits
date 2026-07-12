import pytest

from quantpits.post_trade.cashflow import build_cashflow_commit
from quantpits.post_trade.contracts import PostTradeCashflowConflictError


def test_consumes_only_processed_active_dates_and_preserves_unknown_fields():
    raw = {"cashflows": {"2026-01-02": 0, "2026-01-03": -2}, "processed": {}, "notes": "keep"}
    result = build_cashflow_commit(raw, ("2026-01-02",))
    assert result.consumed_dates == ("2026-01-02",)
    assert result.next_config == {"cashflows": {"2026-01-03": -2}, "processed": {"2026-01-02": 0}, "notes": "keep"}
    assert raw["cashflows"]["2026-01-02"] == 0


def test_processed_value_conflict_fails():
    with pytest.raises(PostTradeCashflowConflictError):
        build_cashflow_commit({"cashflows": {"2026-01-02": 2}, "processed": {"2026-01-02": 1}}, ("2026-01-02",))


def test_legacy_value_applies_to_first_date_and_resets():
    result = build_cashflow_commit({"cash_flow_today": 3, "note": "x"}, ("2026-01-02", "2026-01-03"))
    assert result.consumed_dates == ("2026-01-02",)
    assert result.next_config == {"cash_flow_today": 0, "note": "x"}
