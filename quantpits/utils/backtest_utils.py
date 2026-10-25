import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, Any, Optional

def extract_report_df(metrics: Any) -> pd.DataFrame:
    """提取 Qlib 回测的 portfolio metrics 到独立的 DataFrame 中。"""
    if isinstance(metrics, dict):
        val = list(metrics.values())[0]
        return val[0] if isinstance(val, tuple) else val
    elif isinstance(metrics, tuple):
        first = metrics[0]
        if isinstance(first, pd.DataFrame):
            return first
        elif isinstance(first, tuple) and len(first) >= 1:
            return first[0]
        return metrics
    return metrics

def run_backtest_with_strategy(
    strategy_inst, 
    trade_exchange, 
    freq: str, 
    account_cash: float, 
    bt_start: str, 
    bt_end: str,
    benchmark: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    使用共同的 trade_exchange，实例化一个安全的 SimulatorExecutor 并执行 backtest_loop。
    该方法不仅提供了安全的内存隔离（每次创建新 Account 并在 CommonInfrastructure 同享 Exchange），
    同时也为大规模穷举优化了性能表现。

    Returns:
        report_df: pd.DataFrame 包含 account 和 bench 的表现轨迹
        executor_obj: 完成回测的 SimulatorExecutor，内置了完备的 trade_account 细节。
    """
    from qlib.backtest import backtest_loop
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.backtest.utils import CommonInfrastructure
    from qlib.backtest.account import Account

    account_options = {"init_cash": account_cash}
    if benchmark is not None:
        account_options["benchmark_config"] = {
            "benchmark": benchmark, "start_time": bt_start, "end_time": bt_end,
        }
    trade_account = Account(**account_options)
    
    common_infra = CommonInfrastructure(
        trade_account=trade_account,
        trade_exchange=trade_exchange
    )

    # 策略必须重新绑定对应的 infra
    strategy_inst.reset_common_infra(common_infra)

    executor_obj = SimulatorExecutor(
        time_per_step=freq,
        generate_portfolio_metrics=True,
        verbose=False,
        common_infra=common_infra,
    )

    with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_portfolio_metrics, _ = backtest_loop(
            start_time=bt_start,
            end_time=bt_end,
            trade_strategy=strategy_inst,
            trade_executor=executor_obj,
        )

    report_df = extract_report_df(raw_portfolio_metrics)
    if report_df is not None:
        report_df['nav'] = report_df['account']
        
    return report_df, executor_obj

def standard_evaluate_portfolio(
    report_df: pd.DataFrame, 
    benchmark_col: str, 
    freq: str
) -> Dict[str, float]:
    """
    从 report_df 换算成基于 PortfolioAnalyzer 的完备绩效指标。
    因为各业务段（穷举、融合）经常需要将含有 account 和 bench 列的原始输出提取为传统金融指标。
    """
    from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer
    from qlib.data import D
    
    da_df = pd.DataFrame(index=report_df.index)
    da_df['收盘价值'] = report_df['account']
    da_df[benchmark_col] = (1 + report_df['bench']).cumprod()
    
    if not isinstance(da_df.index, pd.DatetimeIndex):
        da_df.index = pd.to_datetime(da_df.index)
    da_df.index.name = '成交日期'
    
    bt_start_dt = da_df.index.min()
    bt_end_dt = da_df.index.max()
    daily_dates = D.calendar(start_time=bt_start_dt, end_time=bt_end_dt, freq='day')
    
    da_df = da_df.reindex(daily_dates, method='ffill').dropna(subset=['收盘价值'])
    da_df = da_df.reset_index().rename(columns={'index': '成交日期', 'datetime': '成交日期'})

    pa = PortfolioAnalyzer(
        daily_amount_df=da_df, 
        trade_log_df=pd.DataFrame(), 
        holding_log_df=pd.DataFrame(),
        benchmark_col=benchmark_col, 
        freq=freq
    )
    return pa.calculate_traditional_metrics()
