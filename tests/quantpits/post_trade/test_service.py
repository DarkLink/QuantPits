from types import SimpleNamespace
from decimal import Decimal

import pandas as pd
import pytest

from quantpits.post_trade.contracts import ExecutionReconciliationError, SourceChangedError
from quantpits.post_trade.service import PostTradeService
from quantpits.post_trade.state import ValuationSnapshot
from quantpits.utils.workspace import WorkspaceContext


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


def test_bundle_identity_is_rechecked_after_state_calculation(monkeypatch, tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    ctx.config_dir.mkdir(); ctx.data_dir.mkdir()
    prepared = SimpleNamespace(
        ctx=ctx,
        options=SimpleNamespace(scope="state", dry_run=True),
        catalog=SimpleNamespace(settlement_bundle=object()),
        config=SimpleNamespace(
            raw_prod_config={
                "last_processed_date": "2026-01-01",
                "current_cash": "0", "current_holding": [],
            },
            runtime_config={}, cashflow_config={},
        ),
    )
    parsed = {
        ("settlement", "2026-01-02"): SimpleNamespace(dataframe=pd.DataFrame()),
    }
    service = PostTradeService(SimpleNamespace(
        snapshot=lambda date, *_: ValuationSnapshot(date, (), Decimal("3000")),
    ))
    checked = []
    def reject_drift(*_):
        checked.append(True)
        raise SourceChangedError("drift")
    monkeypatch.setattr(
        "quantpits.post_trade.intake.verify_settlement_bundle", reject_drift,
    )
    with pytest.raises(SourceChangedError, match="drift"):
        service.run_state(prepared, parsed, ("2026-01-02",))
    assert checked == [True]
