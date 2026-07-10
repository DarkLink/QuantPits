import pandas as pd
import pytest
from unittest.mock import patch

from quantpits.order.opinions import ModelOpinionsResult
from quantpits.order.persistence import OrderPersistenceRequest, atomic_write_csv, persist_order_artifacts
from quantpits.utils.workspace import WorkspaceContext


def test_persistence_records_only_actual_non_empty_artifacts(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    opinions = ModelOpinionsResult(
        dataframe=pd.DataFrame({"order_basis": ["BUY (1)"]}, index=["A"]),
        combo_composition={}, model_to_combos={}, source_summaries=(), thresholds={},
    )
    ledger = persist_order_artifacts(OrderPersistenceRequest(
        ctx=ctx, output_dir=tmp_path / "output", trade_date="2026-01-02", source_label="ensemble",
        sell_orders=(), buy_orders=({"instrument": "A", "amount": 100},), opinions=opinions,
    ))
    assert ledger.sell_csv is None
    assert ledger.buy_csv == "output/buy_suggestion_ensemble_2026-01-02.csv"
    assert len(ledger.outputs) == 3
    assert not list((tmp_path / "output").glob("*.tmp"))


def test_empty_request_writes_nothing(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    ledger = persist_order_artifacts(OrderPersistenceRequest(
        ctx=ctx, output_dir=tmp_path / "output", trade_date="2026-01-02", source_label="ensemble",
        sell_orders=(), buy_orders=(), opinions=None,
    ))
    assert ledger.outputs == ()
    assert not (tmp_path / "output").exists()


def test_atomic_writer_cleans_temp_file_when_replace_fails(tmp_path):
    target = tmp_path / "output" / "orders.csv"
    with patch("quantpits.order.persistence.os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            atomic_write_csv(pd.DataFrame({"instrument": ["A"]}), target, index=False)
    assert not target.exists()
    assert not list(target.parent.glob("*.tmp"))
