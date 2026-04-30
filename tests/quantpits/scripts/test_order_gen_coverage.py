"""
Supplemental tests targeting uncovered branches in order_gen.py.

Coverage targets:
- Lines 221-225: no score column → rename num col / return None
- Lines 235-236: datetime in columns variant
- Line 242: instrument not in index → set_index
- Lines 420, 426-427: BUY* and pool-outside non-holding
- Lines 484-485: dry-run file skip message
- Lines 709-714: verbose sell display
- Lines 726-728: single-date fallback
- Line 775: source label determination
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def order_gen_env(monkeypatch, tmp_path):
    """Shared fixture for order_gen tests."""
    workspace = tmp_path / "OrderGenWS"
    workspace.mkdir()
    (workspace / "config").mkdir()
    (workspace / "data").mkdir()
    (workspace / "output").mkdir()

    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    import importlib
    from quantpits.utils import env
    importlib.reload(env)

    from quantpits.scripts import order_gen
    from quantpits.utils import strategy
    importlib.reload(order_gen)
    importlib.reload(strategy)

    yield order_gen, strategy, workspace


def test_prepare_daily_df_no_score_column_rename(order_gen_env):
    """Lines 221-223: score not in columns → rename first numeric col."""
    order_gen, _, _ = order_gen_env

    dates = pd.to_datetime(["2020-01-01", "2020-01-01"])
    df = pd.DataFrame({
        "instrument": ["A", "B"],
        "datetime": dates,
        "prediction": [0.9, 0.8],  # numeric, not named 'score'
    }).set_index(["instrument", "datetime"])

    result = order_gen._load_pred_latest_day(df)
    assert result is not None
    assert "score" in result.columns
    assert result.loc["A", "score"] == 0.9


def test_prepare_daily_df_no_score_no_numeric(order_gen_env):
    """Lines 221-225: no score AND no numeric columns → return None."""
    order_gen, _, _ = order_gen_env

    dates = pd.to_datetime(["2020-01-01"])
    df = pd.DataFrame({
        "instrument": ["A"],
        "datetime": dates,
        "name": ["test"],  # non-numeric
    }).set_index(["instrument", "datetime"])

    result = order_gen._load_pred_latest_day(df)
    assert result is None


def test_prepare_daily_df_datetime_in_columns(order_gen_env):
    """Lines 235-236: datetime is a column, not in index."""
    order_gen, _, _ = order_gen_env

    df = pd.DataFrame({
        "instrument": ["A", "B"],
        "datetime": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "score": [0.9, 0.8],
    })

    result = order_gen._load_pred_latest_day(df)
    assert result is not None
    assert "A" in result.index
    assert result.loc["A", "score"] == 0.9


def test_prepare_daily_df_no_instrument_in_index(order_gen_env):
    """Line 242: instrument not in index → set_index."""
    order_gen, _, _ = order_gen_env

    df = pd.DataFrame({
        "instrument": ["A", "B"],
        "score": [0.9, 0.8],
    })

    result = order_gen._load_pred_latest_day(df)
    assert result is not None
    assert result.index.name == "instrument"


def test_prepare_daily_df_single_date(order_gen_env):
    """Lines 232-233: only one unique datetime → droplevel."""
    order_gen, _, _ = order_gen_env

    dates = pd.to_datetime(["2020-01-01", "2020-01-01"])
    df = pd.DataFrame({
        "instrument": ["A", "B"],
        "datetime": dates,
        "score": [0.9, 0.8],
    }).set_index(["instrument", "datetime"])

    result = order_gen._load_pred_latest_day(df)
    assert result is not None
    assert "datetime" not in (result.index.names or [])


def test_analyze_positions_buy_star_logic(order_gen_env):
    """Lines 420, 426-427: BUY* decisions and pool-outside non-holding."""
    order_gen, strategy, _ = order_gen_env

    og = strategy.TopkDropoutOrderGenerator(topk=2, n_drop=0, buy_suggestion_factor=3)

    # Create prediction data: 6 instruments
    pred_data = {
        "instrument": ["A", "B", "C", "D", "E", "F"],
        "datetime": ["2026-03-01"] * 6,
        "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    }
    pred_df = pd.DataFrame(pred_data).set_index(["instrument", "datetime"])

    price_data = {
        "instrument": ["A", "B", "C", "D", "E", "F"],
        "current_close": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "possible_max": [11.0, 22.0, 33.0, 44.0, 55.0, 66.0],
        "possible_min": [9.0, 18.0, 27.0, 36.0, 45.0, 54.0],
    }
    price_df = pd.DataFrame(price_data).set_index("instrument")

    # No holdings → all non-held
    current_holding = []

    # topk=2, n_drop=0, buy_suggestion_factor=3
    # pool_size = topk + n_drop * buy_suggestion_factor = 2
    # buy_count = topk - len(hold_final) = 2
    # buy_primary: first 2 non-held in pool → A, B → BUY
    # buy_backup: next (buy_count * factor - len(buy_primary)) = 2*3-2 = 4
    #   But pool only covers 2 instruments so backup is empty

    hold_final, sell_cand, buy_cand, merged_df, buy_count = og.analyze_positions(
        pred_df, price_df, current_holding
    )

    # With n_drop=0, pool_size = 2
    # A and B get BUY, C-F get -- (pool outside)
    assert "A" in merged_df.index
    assert "C" in merged_df.index
    # C should be '--' because it's pool-outside non-holding


def test_analyze_positions_buy_star_with_backups(order_gen_env):
    """Lines 414, 420: BUY* decisions when backup slots exist.

    The TopkDropoutOrderGenerator.analyze_positions returns:
    (hold_final, sell_cand, buy_cand, merged_df, buy_count).
    buy_cand includes buy_suggestion_factor * buy_count candidates.
    """
    order_gen, strategy, _ = order_gen_env

    # topk=3, n_drop=0, buy_suggestion_factor=2
    # pool_size = 3 + 0*2 = 3
    # buy_count = 3 - 0 = 3
    # buy_candidates = 3 * 2 = 6 (but only non-held in pool)
    og = strategy.TopkDropoutOrderGenerator(topk=3, n_drop=0, buy_suggestion_factor=2)

    pred_data = {
        "instrument": [f"I{i}" for i in range(1, 8)],
        "datetime": ["2026-03-01"] * 7,
        "score": list(reversed(range(7))),
    }
    pred_df = pd.DataFrame(pred_data).set_index(["instrument", "datetime"])

    price_data = {
        "instrument": [f"I{i}" for i in range(1, 8)],
        "current_close": [10.0] * 7,
        "possible_max": [11.0] * 7,
        "possible_min": [9.0] * 7,
    }
    price_df = pd.DataFrame(price_data).set_index("instrument")

    current_holding = []

    hold_final, sell_cand, buy_cand, merged_df, buy_count = og.analyze_positions(
        pred_df, price_df, current_holding
    )

    # With topk=3, n_drop=0: buy_count = 3
    assert buy_count == 3
    # buy_cand = non_held_in_pool[:buy_count * factor]
    # = first 6 of 3 non-held in pool = 3
    assert len(buy_cand) == 3
    # No holdings → no sells
    assert len(sell_cand) == 0
    assert len(hold_final) == 0


def test_save_orders_dry_run_no_files_written(order_gen_env, capsys):
    """Lines 484-485: dry_run=True → files not written, message printed."""
    order_gen, _, workspace = order_gen_env

    sell_orders = [{"instrument": "000001", "datetime": "2026-03-02", "value": 100}]
    buy_orders = [{"instrument": "000002", "datetime": "2026-03-02", "value": 200}]

    out_dir = workspace / "output"
    sell_file, buy_file = order_gen.save_orders(
        sell_orders, buy_orders,
        next_trade_date_string="2026-03-02",
        output_dir=str(out_dir),
        source_label="ensemble",
        dry_run=True,
    )

    captured = capsys.readouterr()
    assert "DRY-RUN" in captured.out
    assert not os.path.exists(sell_file)


@patch("quantpits.scripts.order_gen.init_qlib")
@patch("quantpits.scripts.order_gen.get_anchor_date")
@patch("quantpits.scripts.order_gen.load_configs")
@patch("quantpits.scripts.order_gen.load_predictions")
@patch("quantpits.scripts.order_gen.get_price_data")
@patch("qlib.data.D", create=True)
def test_main_verbose_sell_display(mock_D, mock_price, mock_pred, mock_configs,
                                     mock_anchor, mock_init, order_gen_env, capsys):
    """Lines 709-714: --verbose flag triggers sell candidate display."""
    order_gen, _, workspace = order_gen_env

    mock_anchor.return_value = "2020-01-01"
    mock_configs.return_value = (
        {
            "market": "csi300",
            "current_cash": 1000000,
            "current_holding": [{"instrument": "HOLD1", "value": 1000}],
        },
        {"cash_flow_today": 0},
    )

    idx = pd.MultiIndex.from_tuples([
        ("000001", pd.to_datetime("2020-01-01")),
        ("HOLD1", pd.to_datetime("2020-01-01")),
    ], names=["instrument", "datetime"])
    mock_pred.return_value = (
        pd.DataFrame({"score": [0.9, 0.8]}, index=idx),
        "Mock Source",
    )

    idx_p = pd.Index(["000001", "HOLD1"], name="instrument")
    mock_price.return_value = pd.DataFrame({
        "current_close": [10.0, 20.0],
        "possible_max": [11.0, 22.0],
        "possible_min": [9.0, 18.0],
    }, index=idx_p)

    mock_D.calendar.return_value = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]

    import sys
    with patch.object(sys, "argv", ["script.py", "--verbose", "--dry-run"]):
        order_gen.main()

    captured = capsys.readouterr()
    assert "卖出候选" in captured.out


@patch("quantpits.scripts.order_gen.init_qlib")
@patch("quantpits.scripts.order_gen.get_anchor_date")
@patch("quantpits.scripts.order_gen.load_configs")
@patch("quantpits.scripts.order_gen.load_predictions")
@patch("quantpits.scripts.order_gen.get_price_data")
@patch("qlib.data.D", create=True)
def test_main_single_date_no_calendar_next(mock_D, mock_price, mock_pred, mock_configs,
                                              mock_anchor, mock_init, order_gen_env, capsys):
    """Lines 726-728: single date in calendar → prints message."""
    order_gen, _, workspace = order_gen_env

    mock_anchor.return_value = "2020-01-01"
    mock_configs.return_value = (
        {"market": "csi300", "current_cash": 1000000,
         "current_holding": [{"instrument": "HOLD1", "value": 1000}]},
        {},
    )

    idx = pd.MultiIndex.from_tuples([
        ("000001", pd.to_datetime("2020-01-01")),
        ("HOLD1", pd.to_datetime("2020-01-01")),
    ], names=["instrument", "datetime"])
    mock_pred.return_value = (pd.DataFrame({"score": [0.9, 0.8]}, index=idx), "src")

    idx_p = pd.Index(["000001", "HOLD1"], name="instrument")
    mock_price.return_value = pd.DataFrame({
        "current_close": [10.0, 20.0],
        "possible_max": [11.0, 22.0],
        "possible_min": [9.0, 18.0],
    }, index=idx_p)

    # Only one date in calendar — the script should complete successfully
    mock_D.calendar.return_value = [pd.Timestamp("2020-01-01")]

    import sys
    with patch.object(sys, "argv", ["script.py", "--dry-run"]):
        order_gen.main()

    captured = capsys.readouterr()
    # Script completed successfully (no crash with single date)
    assert "订单生成完成" in captured.out


@patch("quantpits.scripts.order_gen.init_qlib")
@patch("quantpits.scripts.order_gen.get_anchor_date")
@patch("quantpits.scripts.order_gen.load_configs")
@patch("quantpits.scripts.order_gen.load_predictions")
@patch("quantpits.scripts.order_gen.get_price_data")
@patch("qlib.data.D", create=True)
def test_main_source_ensemble_label(mock_D, mock_price, mock_pred, mock_configs,
                                      mock_anchor, mock_init, order_gen_env, capsys):
    """Line 775: source_label='ensemble' when loading without explicit model."""
    order_gen, _, workspace = order_gen_env

    mock_anchor.return_value = "2020-01-01"
    mock_configs.return_value = (
        {"market": "csi300", "current_cash": 1000000,
         "current_holding": []},
        {},
    )

    idx = pd.MultiIndex.from_tuples([
        ("000001", pd.to_datetime("2020-01-01")),
    ], names=["instrument", "datetime"])
    mock_pred.return_value = (
        pd.DataFrame({"score": [0.9]}, index=idx),
        "Ensemble 融合 (test_combo)",
    )

    idx_p = pd.Index(["000001"], name="instrument")
    mock_price.return_value = pd.DataFrame({
        "current_close": [10.0],
        "possible_max": [11.0],
        "possible_min": [9.0],
    }, index=idx_p)

    mock_D.calendar.return_value = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]

    import sys
    with patch.object(sys, "argv", ["script.py", "--dry-run"]):
        order_gen.main()

    captured = capsys.readouterr()
    assert "来源" in captured.out
