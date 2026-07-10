from unittest.mock import MagicMock, patch

import pandas as pd

from quantpits.ensemble.risk_report import (
    RiskLeaderboardRequest,
    build_daily_amount_frame,
    build_risk_leaderboard,
    calculate_safe_risk,
    leaderboard_output_path,
)


def _report_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account": [100.0, 110.0],
            "bench": [0.0, 0.05],
            "return": [0.0, 0.10],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )


def _norm_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2020-01-01"), "A"),
            (pd.Timestamp("2020-01-02"), "A"),
        ],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"M1": [0.5, 0.6], "M2": [0.4, 0.7]}, index=index)


def test_leaderboard_output_path_preserves_legacy_names(tmp_path):
    assert leaderboard_output_path(tmp_path, "2026-01-07") == (
        tmp_path / "leaderboard_2026-01-07.csv"
    )
    assert leaderboard_output_path(tmp_path, "2026-01-07", combo_name="combo_a") == (
        tmp_path / "leaderboard_combo_a_2026-01-07.csv"
    )


def test_calculate_safe_risk_accepts_dataframe_and_flattens_dataframe_result():
    with patch("qlib.contrib.evaluate.risk_analysis") as mock_risk:
        mock_risk.return_value = pd.DataFrame({"risk": [0.1, 0.2]}, index=["mean", "std"])

        result = calculate_safe_risk(pd.DataFrame({"ret": [0.01, 0.02]}), "day")

    assert result == {"mean": 0.1, "std": 0.2}
    passed_returns = mock_risk.call_args.args[0]
    assert isinstance(passed_returns, pd.Series)
    assert passed_returns.tolist() == [0.01, 0.02]


def test_calculate_safe_risk_returns_empty_dict_on_failure():
    with patch("qlib.contrib.evaluate.risk_analysis", side_effect=RuntimeError("boom")):
        assert calculate_safe_risk(pd.Series([0.01]), "day") == {}


def test_build_daily_amount_frame_normalizes_legacy_columns():
    amount_frame = build_daily_amount_frame(_report_frame(), "SH000300")

    assert isinstance(amount_frame.index, pd.DatetimeIndex)
    assert amount_frame.index.name == "成交日期"
    assert amount_frame["收盘价值"].tolist() == [100.0, 110.0]
    assert amount_frame["SH000300"].round(4).tolist() == [1.0, 1.05]


@patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer")
@patch("quantpits.utils.strategy.load_strategy_config")
def test_build_risk_leaderboard_ensemble_only_writes_legacy_csv(
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
        "CAGR_252": 0.12,
        "Benchmark_CAGR_252": 0.08,
        "Benchmark_Sharpe": 0.5,
        "Benchmark_Max_Drawdown": -0.04,
        "Excess_Return_CAGR_252": 0.03,
        "Information_Ratio_(Arithmetic)": 1.2,
        "Max_Drawdown": -0.05,
    }
    mock_analyzer_cls.return_value = analyzer

    result = build_risk_leaderboard(
        RiskLeaderboardRequest(
            report_df=_report_frame(),
            norm_df=_norm_frame(),
            train_records={"experiment_name": "E", "models": {}},
            loaded_models=[],
            freq="day",
            output_dir=tmp_path,
            anchor_date="2020-01-02",
        ),
        verbose=False,
    )

    assert result.csv_path == tmp_path / "leaderboard_2020-01-02.csv"
    assert result.csv_path.exists()
    assert list(result.all_reports) == ["Ensemble"]
    assert "Ensemble" in result.leaderboard.index
    assert result.leaderboard.loc["Ensemble", "annualized_return"] == 0.12
    assert result.leaderboard.loc["Ensemble", "annualized_excess"] == 0.03


@patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer")
@patch("quantpits.utils.strategy.load_strategy_config")
def test_build_risk_leaderboard_skips_missing_record_id(
    mock_load_strategy,
    mock_analyzer_cls,
    monkeypatch,
    tmp_path,
):
    import qlib.data
    import qlib.workflow

    mock_load_strategy.return_value = {"benchmark": "SH000300"}
    mock_d = MagicMock()
    mock_d.calendar.return_value = pd.date_range("2020-01-01", periods=2, freq="D")
    monkeypatch.setattr(qlib.data, "D", mock_d)

    analyzer = MagicMock()
    analyzer.calculate_traditional_metrics.return_value = {
        "CAGR_252": 0.10,
        "Excess_Return_CAGR_252": 0.04,
        "Information_Ratio_(Arithmetic)": 1.0,
        "Max_Drawdown": -0.02,
    }
    mock_analyzer_cls.return_value = analyzer
    monkeypatch.setattr(qlib.workflow, "R", MagicMock())

    result = build_risk_leaderboard(
        RiskLeaderboardRequest(
            report_df=None,
            norm_df=_norm_frame(),
            train_records={"experiment_name": "E", "models": {"M1": None}},
            loaded_models=["M1"],
            freq="day",
            output_dir=tmp_path,
            anchor_date="2020-01-02",
        ),
        verbose=False,
    )

    qlib.workflow.R.get_recorder.assert_not_called()
    assert result.all_reports == {}
    assert result.leaderboard is None
    assert result.csv_path is None


@patch("quantpits.scripts.analysis.portfolio_analyzer.PortfolioAnalyzer")
@patch("quantpits.utils.train_utils.get_experiment_name_for_model")
@patch("quantpits.utils.strategy.load_strategy_config")
def test_build_risk_leaderboard_skips_recorder_failures(
    mock_load_strategy,
    mock_get_exp,
    mock_analyzer_cls,
    monkeypatch,
    tmp_path,
):
    import qlib.data
    import qlib.workflow

    mock_load_strategy.return_value = {"benchmark": "SH000300"}
    mock_get_exp.return_value = "E"
    mock_d = MagicMock()
    mock_d.calendar.return_value = pd.date_range("2020-01-01", periods=2, freq="D")
    monkeypatch.setattr(qlib.data, "D", mock_d)
    monkeypatch.setattr(qlib.workflow, "R", MagicMock())
    qlib.workflow.R.get_recorder.side_effect = RuntimeError("recorder fail")

    result = build_risk_leaderboard(
        RiskLeaderboardRequest(
            report_df=None,
            norm_df=_norm_frame(),
            train_records={"experiment_name": "E", "models": {"M1": "rid1"}},
            loaded_models=["M1"],
            freq="day",
            output_dir=tmp_path,
            anchor_date="2020-01-02",
        ),
        verbose=False,
    )

    assert result.all_reports == {}
    assert result.leaderboard is None
    assert result.csv_path is None


def test_script_risk_analysis_wrapper_delegates_to_ensemble_module(tmp_path):
    import quantpits.scripts.ensemble_fusion as ef

    expected = ({"Ensemble": _report_frame()}, pd.DataFrame({"metric": [1.0]}, index=["Ensemble"]))
    with patch(
        "quantpits.ensemble.risk_report.risk_analysis_and_leaderboard",
        return_value=expected,
    ) as mock_risk:
        result = ef.risk_analysis_and_leaderboard(
            _report_frame(),
            _norm_frame(),
            {"experiment_name": "E", "models": {}},
            [],
            "day",
            tmp_path,
            "2020-01-02",
            combo_name="combo_a",
        )

    assert result is expected
    mock_risk.assert_called_once()
    assert mock_risk.call_args.kwargs == {"combo_name": "combo_a"}
