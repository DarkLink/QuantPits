"""
详尽回测分析报告

从 ensemble_fusion.py 的 run_detailed_backtest_analysis 抽取。
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


def run_detailed_backtest_analysis(executor_obj, combo_name, anchor_date, output_dir, freq, benchmark='SH000300'):
    """
    运行详尽的回测分析报告（复用 PortfolioAnalyzer 等组件）
    """
    from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer

    print(f"\n{'='*60}")
    print("详尽回测分析")
    print(f"{'='*60}")

    ta = executor_obj.trade_account
    pm_tuple = ta.get_portfolio_metrics()
    if not pm_tuple or len(pm_tuple) < 1:
        print("Error: No portfolio metrics found.")
        return None

    # 1. 构造 daily_amount_df
    source_df = pm_tuple[0].copy()
    daily_amount_df = pd.DataFrame(index=source_df.index)
    daily_amount_df['收盘价值'] = source_df['account']
    daily_amount_df[benchmark] = (1 + source_df['bench']).cumprod()

    # Ensure index is datetime
    daily_amount_df.index = pd.to_datetime(daily_amount_df.index)

    # Up-sample to daily frequency
    from qlib.data import D
    bt_start = daily_amount_df.index.min()
    bt_end = daily_amount_df.index.max()
    daily_dates = D.calendar(start_time=bt_start, end_time=bt_end, freq='day')
    daily_amount_df = daily_amount_df.reindex(daily_dates, method='ffill').dropna(subset=['收盘价值'])
    daily_amount_df.index.name = '成交日期'
    daily_amount_df = daily_amount_df.reset_index()

    # 2. 构造 holding_log_df 和 trade_log_df
    hist_pos = ta.get_hist_positions()
    holding_data = []
    trades = []
    prev_pos_counts = {}

    sorted_dates = sorted(hist_pos.keys())
    for date in sorted_dates:
        pos_obj = hist_pos[date]
        dt = pd.to_datetime(date)

        curr_pos_counts = {}
        for inst, unit in pos_obj.position.items():
            try:
                if hasattr(unit, 'count'):
                    count = unit.count
                    price = unit.price
                else:
                    count = unit
                    price = 1.0 if inst == 'CASH' else 0.0

                # Defensive handling for unexpected types
                if hasattr(count, 'iloc'): count = count.iloc[-1]
                if isinstance(count, dict): count = next(iter(count.values())) if count else 0.0
                if hasattr(price, 'iloc'): price = price.iloc[-1]
                if isinstance(price, dict): price = next(iter(price.values())) if price else 0.0

                count = float(count)
                price = float(price)
            except Exception as e:
                print(f"  [DEBUG] Error extracting {inst}: {e}, unit type: {type(unit)}")
                count = 0.0
                price = 0.0

            holding_data.append({
                '成交日期': dt,
                '证券代码': inst,
                '当前持仓': count,
                '收盘价值': (count * price) if inst != 'CASH' else count
            })
            if inst != 'CASH':
                curr_pos_counts[inst] = count

        # Simple trade reconstruction
        all_insts = set(prev_pos_counts.keys()) | set(curr_pos_counts.keys())
        for inst in all_insts:
            prev_n = prev_pos_counts.get(inst, 0)
            curr_n = curr_pos_counts.get(inst, 0)
            if curr_n > prev_n:
                u = pos_obj.position.get(inst)
                price = u.price if hasattr(u, 'price') else 0
                trades.append({
                    '成交日期': dt,
                    '证券代码': inst,
                    '交易类别': '买入',
                    '成交数量': curr_n - prev_n,
                    '成交金额': (curr_n - prev_n) * price
                })
            elif curr_n < prev_n:
                u = pos_obj.position.get(inst)
                if hasattr(u, 'price'):
                    price = u.price
                else:
                    price = 0.0

                trades.append({
                    '成交日期': dt,
                    '证券代码': inst,
                    '交易类别': '卖出',
                    '成交数量': prev_n - curr_n,
                    '成交金额': (prev_n - curr_n) * price
                })
        prev_pos_counts = curr_pos_counts

    holding_log_df = pd.DataFrame(holding_data)
    trade_log_df = pd.DataFrame(trades)

    # 3. 运行分析组件
    pa = PortfolioAnalyzer(
        daily_amount_df=daily_amount_df,
        trade_log_df=trade_log_df,
        holding_log_df=holding_log_df,
        benchmark_col=benchmark,
        freq=freq
    )

    bt_start = daily_amount_df['成交日期'].min().strftime('%Y-%m-%d')
    bt_end = daily_amount_df['成交日期'].max().strftime('%Y-%m-%d')

    # 4. 生成报告 (Markdown)
    report = []
    report.append(f"# Backtest Detailed Analysis Report - {combo_name if combo_name else 'Ensemble'}")
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n## Analysis Scope")
    report.append(f"- **Combo**: {combo_name}")
    report.append(f"- **Anchor Date**: {anchor_date}")
    report.append(f"- **Backtest Period**: {bt_start} to {bt_end}")
    report.append(f"- **Benchmark**: {benchmark}")
    report.append(f"- **Frequency**: {freq}")

    # Run metrics
    metrics = pa.calculate_traditional_metrics()
    exposure = pa.calculate_factor_exposure()
    style_exp = pa.calculate_style_exposures()
    holding_metrics = pa.calculate_holding_metrics()

    report.append(f"\n## 1. Portfolio Strategy & Returns")

    if holding_metrics:
        report.append("### Holding Analytics")
        report.append(f"- **Avg Daily Holdings Count**: {holding_metrics.get('Avg_Daily_Holdings_Count', 0):.1f}")
        report.append(f"- **Avg Top 1 Concentration**: {holding_metrics.get('Avg_Top1_Concentration', 0):.2%}")
        if not pd.isna(holding_metrics.get('Avg_Floating_Return')):
            report.append(f"- **Avg Floating Return**: {holding_metrics.get('Avg_Floating_Return', 0):.2%}")
        if not pd.isna(holding_metrics.get('Daily_Holding_Win_Rate')):
            report.append(f"- **Daily Holding Win Rate**: {holding_metrics.get('Daily_Holding_Win_Rate', 0):.2%}")

    report.append("\n### Traditional Return & Risk")
    if metrics:
        report.append(f"- **Absolute Return**: {metrics.get('Absolute_Return', 0):.2%}")
        report.append(f"- **Benchmark Absolute Return**: {metrics.get('Benchmark_Absolute_Return', 0):.2%}")
        report.append(f"- **CAGR**: {metrics.get('CAGR', 0):.2%}")
        report.append(f"- **Benchmark CAGR**: {metrics.get('Benchmark_CAGR', 0):.2%}")
        report.append(f"- **Excess Return CAGR**: {metrics.get('Excess_Return_CAGR', 0):.2%}")
        report.append(f"- **Volatility**: {metrics.get('Volatility', 0):.2%}")
        report.append(f"- **Benchmark Volatility**: {metrics.get('Benchmark_Volatility', 0):.2%}")
        report.append(f"- **Tracking Error**: {metrics.get('Tracking_Error', 0):.2%}")
        report.append(f"- **Sharpe**: {metrics.get('Sharpe', 0):.4f}")
        report.append(f"- **Benchmark Sharpe**: {metrics.get('Benchmark_Sharpe', 0):.4f}")
        report.append(f"- **Information Ratio**: {metrics.get('Information_Ratio', 0):.4f}")
        report.append(f"- **Max Drawdown**: {metrics.get('Max_Drawdown', 0):.2%}")
        report.append(f"- **Benchmark Max Drawdown**: {metrics.get('Benchmark_Max_Drawdown', 0):.2%}")
        report.append(f"- **Turnover Rate Annual**: {metrics.get('Turnover_Rate_Annual', 0):.2f}x")

    if exposure:
        report.append(f"\n### Factor Exposure")
        report.append(f"- **Beta (Market)**: {exposure.get('Beta_Market', 0):.4f}")
        report.append(f"- **Annualized Alpha**: {exposure.get('Annualized_Alpha', 0):.2%}")
        report.append(f"- **R-Squared**: {exposure.get('R_Squared', 0):.4f}")
        if style_exp:
            report.append(f"- **Multi-Factor Beta**: {style_exp.get('Multi_Factor_Beta', 0):.4f}")
            report.append(f"- **Barra Size Exposure**: {style_exp.get('Barra_Liquidity_Exp', 0):.4f}")
            report.append(f"- **Barra Momentum Exposure**: {style_exp.get('Barra_Momentum_Exp', 0):.4f}")
            report.append(f"- **Barra Volatility Exposure**: {style_exp.get('Barra_Volatility_Exp', 0):.4f}")
            report.append(f"- **Style R-Squared**: {style_exp.get('Barra_Style_R_Squared', 0):.4f}")

    if metrics and style_exp:
        cagr = metrics.get('CAGR', 0)
        bench_cagr = metrics.get('Benchmark_CAGR', 0)
        beta = exposure.get('Beta_Market', 0)
        factor_ann = style_exp.get('Factor_Annualized', {})

        beta_ret = beta * bench_cagr
        style_ret = 0.0
        if 'size' in factor_ann and 'momentum' in factor_ann and 'volatility' in factor_ann:
            style_ret += style_exp.get('Barra_Liquidity_Exp', 0) * factor_ann['size']
            style_ret += style_exp.get('Barra_Momentum_Exp', 0) * factor_ann['momentum']
            style_ret += style_exp.get('Barra_Volatility_Exp', 0) * factor_ann['volatility']

        idio_alpha = style_exp.get('Multi_Factor_Intercept', 0)
        arithmetic_total = beta_ret + style_ret + idio_alpha
        cagr_gap = cagr - arithmetic_total

        report.append("\n### Performance Attribution")
        report.append(f"- **Total Strategy CAGR**: {cagr:.2%}")
        report.append(f"  - **Beta Return (Market Exposure)**: {beta_ret:.2%}")
        report.append(f"  - **Style Alpha (Risk Factors)**: {style_ret:.2%}")
        report.append(f"  - **Idiosyncratic Alpha (Selection/Timing)**: {idio_alpha:.2%}")
        report.append(f"  - **Compounding Gap (Residual)**: {cagr_gap:.2%}")

    content = "\n".join(report)
    suffix = f"_{combo_name}" if combo_name else ""
    report_file = os.path.join(output_dir, f"backtest_analysis_report{suffix}_{anchor_date}.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"详细回测分析报告已生成: {report_file}")
    return report_file
