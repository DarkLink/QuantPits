from decimal import Decimal

import pandas as pd

from quantpits.post_trade.state import AccountState, DailyStateTransition, PostTradeStateChangeSet, ValuationSnapshot
from quantpits.post_trade.state_outputs import build_state_output_payloads
from quantpits.utils.workspace import WorkspaceContext


def test_outputs_replace_same_date_and_include_cashflow_target(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path / "Demo_Workspace")
    ctx.data_dir.mkdir(parents=True); ctx.config_dir.mkdir()
    ctx.data_path("daily_amount_log_full.csv").write_text("成交日期,持仓成本\n2026-01-02,999\n", encoding="utf-8")
    state = AccountState("2026-01-01", Decimal("100"), ())
    valuation = ValuationSnapshot("2026-01-02", (), Decimal("20"))
    transition = DailyStateTransition("2026-01-02", state, state, (), Decimal("0"), Decimal("0"), Decimal("0"), valuation, ())
    change = PostTradeStateChangeSet(state, state, (transition,), ("2026-01-02",), (), {"last_processed_date": "2026-01-02"}, ())
    payloads = build_state_output_payloads(ctx, change, {"2026-01-02": pd.DataFrame()}, cashflow_config={"cashflows": {"2026-01-02": 0}})
    assert payloads.daily_log.decode("utf-8-sig").count("2026-01-02") == 1
    assert b'"processed"' in payloads.cashflow_config
    assert b'"market_date":"2026-01-02"' in payloads.valuation_evidence
