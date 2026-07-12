from types import SimpleNamespace

import pandas as pd
import pytest

from quantpits.post_trade.contracts import ExecutionReconciliationError
from quantpits.post_trade.service import PostTradeService


def test_all_scope_empty_execution_frames_still_reconcile_material_settlement(monkeypatch):
    service = PostTradeService(SimpleNamespace(snapshot=lambda *args: None))
    prepared = SimpleNamespace(
        options=SimpleNamespace(scope="all"),
        config=SimpleNamespace(raw_prod_config={"last_processed_date": "2026-01-01", "current_cash": "0", "current_holding": []}, runtime_config={}, cashflow_config={}),
    )
    settlement = pd.DataFrame([{"交易类别": "深圳A股普通股票竞价买入", "证券代码": "000001", "成交价格": 10, "成交数量": 10, "成交金额": 100, "资金发生数": -100, "交收日期": 20260102}])
    parsed = {
        ("settlement", "2026-01-02"): SimpleNamespace(dataframe=settlement),
        ("order", "2026-01-02"): SimpleNamespace(dataframe=pd.DataFrame()),
        ("trade", "2026-01-02"): SimpleNamespace(dataframe=pd.DataFrame()),
    }
    with pytest.raises(ExecutionReconciliationError):
        service.calculate(prepared, parsed, ("2026-01-02",))
