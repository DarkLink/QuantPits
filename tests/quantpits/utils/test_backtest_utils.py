import pytest
import pandas as pd
from quantpits.utils.backtest_utils import extract_report_df

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
