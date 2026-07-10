from unittest.mock import MagicMock, patch

import pandas as pd

from quantpits.ensemble.comparison import (
    ComboComparisonRequest,
    build_base_combo_row,
    build_daily_amount_frame,
    combo_comparison_chart_path,
    combo_comparison_csv_path,
    run_combo_comparison,
)


def _report_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account": [100000.0, 105000.0],
            "return": [0.0, 0.05],
            "bench": [0.0, 0.02],
        },
        index=["2020-01-01", "2020-01-02"],
    )


def test_combo_comparison_paths_preserve_legacy_names(tmp_path):
    assert combo_comparison_csv_path(tmp_path, "2026-01-07") == (
        tmp_path / "combo_comparison_2026-01-07.csv"
    )
    assert combo_comparison_chart_path(tmp_path, "2026-01-07") == (
        tmp_path / "combo_comparison_2026-01-07.png"
    )


def test_build_base_combo_row_preserves_legacy_schema():
    row = build_base_combo_row(
        {
            "name": "combo_a",
            "models": ["M1", "M2"],
            "method": "equal",
            "is_default": True,
        }
    )

    assert row == {
        "combo": "combo_a",
        "models": "M1, M2",
        "method": "equal",
        "is_default": True,
    }


def test_run_combo_comparison_without_reports_writes_csv_only(tmp_path):
    combo_results = [
        {
            "name": "combo_no_bt",
            "models": ["M1", "M2"],
            "method": "equal",
            "is_default": False,
            "report_df": None,
        }
    ]

    with patch("quantpits.utils.strategy.load_strategy_config") as mock_load_strategy:
        result = run_combo_comparison(
            ComboComparisonRequest(
                combo_results=combo_results,
                anchor_date="2026-01-07",
                output_dir=tmp_path,
                freq="day",
            ),
            verbose=False,
        )

    mock_load_strategy.assert_not_called()
    assert result.csv_path == tmp_path / "combo_comparison_2026-01-07.csv"
    assert result.csv_path.exists()
    assert result.chart_path is None
    assert not (tmp_path / "combo_comparison_2026-01-07.png").exists()
    assert result.frame.to_dict("records") == [
        {
            "combo": "combo_no_bt",
            "models": "M1, M2",
            "method": "equal",
            "is_default": False,
        }
    ]


def test_build_daily_amount_frame_normalizes_legacy_columns():
    amount_frame = build_daily_amount_frame(_report_frame(), "SH000300")

    assert isinstance(amount_frame.index, pd.DatetimeIndex)
    assert amount_frame.index.name == "成交日期"
    assert amount_frame["收盘价值"].tolist() == [100000.0, 105000.0]
    assert amount_frame["SH000300"].round(4).tolist() == [1.0, 1.02]


@patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer")
@patch("quantpits.utils.strategy.load_strategy_config")
def test_run_combo_comparison_with_report_writes_metrics_and_chart(
    mock_load_strategy,
    mock_analyzer_cls,
    monkeypatch,
    tmp_path,
):
    import qlib.data

    mock_load_strategy.return_value = {"benchmark": "SH000300"}
    mock_d = MagicMock()
    mock_d.calendar.return_value = pd.date_range("2020-01-01", periods=2, freq="D")
    monkeypatch.setattr(qlib.data, "D", mock_d)
    analyzer = MagicMock()
    analyzer.calculate_traditional_metrics.return_value = {
        "Absolute_Return": 0.05,
        "Benchmark_Absolute_Return": 0.02,
        "CAGR_252": 0.12,
        "Excess_Return_CAGR_252": 0.08,
        "Max_Drawdown": -0.03,
        "Calmar": 1.23456,
    }
    mock_analyzer_cls.return_value = analyzer

    combo_results = [
        {
            "name": "combo1",
            "models": ["M1", "M2"],
            "method": "equal",
            "is_default": True,
            "report_df": _report_frame(),
        }
    ]

    result = run_combo_comparison(
        ComboComparisonRequest(
            combo_results=combo_results,
            anchor_date="2020-01-01",
            output_dir=tmp_path,
            freq="day",
        ),
        verbose=False,
    )

    row = result.frame.iloc[0]
    assert row["combo"] == "combo1"
    assert row["total_return"] == 5.0
    assert row["annualized_return"] == 12.0
    assert row["annualized_excess"] == 8.0
    assert row["max_drawdown"] == -3.0
    assert row["calmar_ratio"] == 1.2346
    assert row["excess_return"] == 3.0
    assert result.csv_path.exists()
    assert result.chart_path == tmp_path / "combo_comparison_2020-01-01.png"
    assert result.chart_path.exists()


def test_script_compare_combos_delegates_to_ensemble_module(tmp_path):
    import quantpits.scripts.ensemble_fusion as ef

    expected = pd.DataFrame({"combo": ["combo1"]})
    with patch("quantpits.ensemble.comparison.compare_combos", return_value=expected) as mock_compare:
        result = ef.compare_combos([], "2026-01-07", tmp_path, "day")

    assert result is expected
    mock_compare.assert_called_once_with([], "2026-01-07", tmp_path, "day")
