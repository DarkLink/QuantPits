import pytest
import pandas as pd
from quantpits.utils.backtest_utils import extract_report_df, run_backtest_with_strategy

def test_extract_report_df_dataframe():
    df = pd.DataFrame({"account": [100, 110], "bench": [0, 0.1]})
    res = extract_report_df(df)
    pd.testing.assert_frame_equal(res, df)

def test_extract_report_df_dict():
    df = pd.DataFrame({"account": [100, 110]})
    metrics = {"key": df}
    res = extract_report_df(metrics)
    pd.testing.assert_frame_equal(res, df)
    
    # Nested tuple in dict
    metrics_tuple = {"key": (df, "info")}
    res = extract_report_df(metrics_tuple)
    pd.testing.assert_frame_equal(res, df)

def test_extract_report_df_tuple():
    df = pd.DataFrame({"account": [100, 110]})
    # Simple tuple
    res = extract_report_df((df, "info"))
    pd.testing.assert_frame_equal(res, df)
    
    # Nested tuple
    res = extract_report_df(((df, "nested"), "info"))
    pd.testing.assert_frame_equal(res, df)

def test_extract_report_df_other():
    # Should return input as is if not matched
    res = extract_report_df("not a metric")
    assert res == "not a metric"


@pytest.mark.parametrize("benchmark", [None, "SH000905"])
def test_backtest_accounts_are_fresh_and_explicit_benchmark_is_forwarded(monkeypatch, benchmark):
    from unittest.mock import MagicMock
    accounts = [MagicMock(), MagicMock()]
    factory = MagicMock(side_effect=accounts)
    infra = MagicMock()
    monkeypatch.setattr("qlib.backtest.account.Account", factory)
    monkeypatch.setattr("qlib.backtest.utils.CommonInfrastructure", infra)
    monkeypatch.setattr("qlib.backtest.executor.SimulatorExecutor", MagicMock())
    report = pd.DataFrame({"account": [1000], "bench": [0]})
    monkeypatch.setattr("qlib.backtest.backtest_loop", lambda **kwargs: ({"1day": (report.copy(), {})}, {}))
    for _ in range(2):
        run_backtest_with_strategy(MagicMock(), MagicMock(), "day", 1000, "2024-05-13", "2024-05-17", benchmark)
    expected = {"init_cash": 1000}
    if benchmark is not None:
        expected["benchmark_config"] = {"benchmark": benchmark, "start_time": "2024-05-13", "end_time": "2024-05-17"}
    assert [c.kwargs for c in factory.call_args_list] == [expected, expected]
    assert [c.kwargs["trade_account"] for c in infra.call_args_list] == accounts
