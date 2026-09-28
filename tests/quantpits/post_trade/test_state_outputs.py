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


def test_cash_interest_survives_state_and_output_without_stock_position(tmp_path):
    import io
    from quantpits.post_trade.state import build_change_set, normalize_settlement_frame
    ctx = WorkspaceContext.from_root(tmp_path / "interest")
    ctx.data_dir.mkdir(parents=True); ctx.config_dir.mkdir()
    frame = pd.DataFrame({"证券代码": [""], "交易类别": ["利息归本"],
        "成交价格": [0], "成交数量": [0], "成交金额": [0],
        "资金发生数": [4.05], "交收日期": ["2026-09-21"]})
    events, _ = normalize_settlement_frame(frame, "2026-09-21")
    initial = AccountState("2026-09-18", Decimal("100"), ())
    dates = ("2026-09-21", "2026-09-22")
    valuations = {d: ValuationSnapshot(d, (), Decimal("20")) for d in dates}
    change = build_change_set(initial, dates, {dates[0]: events}, {}, valuations, {})
    assert change.final_state.cash == Decimal("104.05")
    assert change.final_state.positions == ()
    payload = build_state_output_payloads(ctx, change, {dates[0]: frame})
    trades = pd.read_csv(io.BytesIO(payload.trade_log), keep_default_na=False)
    assert trades.iloc[0]["证券代码"] == ""
    assert trades.iloc[0]["资金发生数"] == 4.05
    holdings = pd.read_csv(io.BytesIO(payload.holding_log))
    assert list(holdings["证券代码"]) == ["CASH", "CASH"]
    assert list(holdings["收盘价值"]) == [104.05, 104.05]
