import pandas as pd

from quantpits.order.command import OrderRunConfig
from quantpits.order.opinions import ModelOpinionsRequest, build_model_opinions


def _config():
    return OrderRunConfig(
        merged_config={}, cashflow_config={}, strategy_config={},
        ensemble_config={"combos": {"demo": {"models": ["m1"], "default": True}}},
        ensemble_records={"combos": {"demo": "combo-rid"}},
        train_records={"models": {"m1": "model-rid"}, "experiment_name": "Train"},
    )


def test_opinions_use_config_snapshot_and_report_optional_source_failure():
    sorted_frame = pd.DataFrame({"score": [0.9, 0.8]}, index=["A", "B"])
    calls = []

    def load(record_id, experiment):
        calls.append((record_id, experiment))
        if record_id == "model-rid":
            raise RuntimeError("unavailable")
        return sorted_frame

    result = build_model_opinions(ModelOpinionsRequest(
        focus_instruments=("A", "B"), current_holding_instruments=("B",),
        top_k=2, drop_n=1, buy_suggestion_factor=2,
        sorted_predictions=sorted_frame, trade_date="2026-01-02",
        run_config=_config(), load_prediction=load,
    ))
    assert result is not None
    assert calls == [("combo-rid", "Ensemble_Fusion"), ("model-rid", "Train")]
    assert result.dataframe.loc["A", "order_basis"].startswith("BUY")
    assert result.warnings and "model_m1" in result.warnings[0]


def test_all_optional_failures_remain_visible_with_order_basis():
    sorted_frame = pd.DataFrame({"score": [0.9]}, index=["A"])

    def fail(record_id, experiment):
        raise RuntimeError(f"unavailable: {record_id}")

    result = build_model_opinions(ModelOpinionsRequest(
        focus_instruments=("A",), current_holding_instruments=(),
        top_k=1, drop_n=1, buy_suggestion_factor=2,
        sorted_predictions=sorted_frame, trade_date="2026-01-02",
        run_config=_config(), load_prediction=fail,
    ))
    assert result is not None
    assert result.dataframe.columns.tolist() == ["order_basis"]
    assert len(result.warnings) == 2
