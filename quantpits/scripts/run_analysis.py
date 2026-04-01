#!/usr/bin/env python3
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Adjust path so we can import analysis module
from quantpits.utils import env
os.chdir(env.ROOT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(PROJECT_ROOT)

from quantpits.scripts.analysis.utils import init_qlib, load_model_predictions, get_forward_returns, load_market_config
from quantpits.scripts.analysis.single_model_analyzer import SingleModelAnalyzer
from quantpits.scripts.analysis.ensemble_analyzer import EnsembleAnalyzer
from quantpits.scripts.analysis.execution_analyzer import ExecutionAnalyzer
from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer, BARRA_LIQD_KEY, BARRA_MOMT_KEY, BARRA_VOLA_KEY

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Analysis Module")
    parser.add_argument('--models', type=str, nargs='+', help="Model names to analyze (e.g., gru mlp tabnet, supports model@mode)")
    parser.add_argument('--start-date', type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end-date', type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument('--output', type=str, default="output/analysis_report.md", help="Output markdown file path")
    parser.add_argument('--training-mode', type=str, default=None,
                        choices=['static', 'rolling'],
                        help='训练模式过滤 (默认 None=自动解析)')
    parser.add_argument('--shareable', action='store_true', help="Redact monetary amounts and individual stock details for sharing")
    args = parser.parse_args()

    print("Initializing Qlib...")
    init_qlib()
    
    # 从配置文件读取市场和基准
    market, benchmark = load_market_config()
    print(f"Market: {market}, Benchmark: {benchmark}")
    
    title_suffix = " (Shareable)" if args.shareable else ""
    if args.shareable:
        report = [f"# Comprehensive Analysis Report{title_suffix}"]
    else:
        report = [f"# Comprehensive Analysis Report{title_suffix} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"]
    report.append("\n## Analysis Scope")
    report.append(f"- Models: {args.models if args.models else 'None (Portfolio/Execution Only)'}")
    def format_date_range(start_idx, end_idx):
        if not args.shareable or pd.isna(start_idx) or pd.isna(end_idx):
            return f"{start_idx} to {end_idx}"
        try:
            start_dt = pd.to_datetime(start_idx)
            end_dt = pd.to_datetime(end_idx)
            months = (end_dt.year - start_dt.year) * 12 + end_dt.month - start_dt.month
            
            def season(month):
                if month in [12, 1, 2]: return "Winter"
                if month in [3, 4, 5]: return "Spring"
                if month in [6, 7, 8]: return "Summer"
                return "Fall"
                
            start_str = f"{season(start_dt.month)} {start_dt.year}"
            end_str = f"{season(end_dt.month)} {end_dt.year}"
            return f"~{max(1, months)} Months ({start_str} - {end_str})"
        except:
            return "Redacted Range"

    report.append(f"- Date Range: {format_date_range(args.start_date, args.end_date)}")
    
    def format_count(val):
        if not args.shareable: return str(val)
        if pd.isna(val) or val == 0: return "0"
        if val < 10: return str(int(val))
        elif val < 100: return f"~{int(round(val, -1))}"
        else: return f"~{int(round(val, -2))}"

    def format_adv(val):
        if not args.shareable: return f"{val:.4%}"
        if pd.isna(val): return "N/A"
        if val < 0.001: return "< 0.1%"
        elif val < 0.01: return "< 1.0%"
        elif val < 0.05: return "< 5.0%"
        else: return "> 5.0%"

    # 1. Single Model Analysis & 2. Ensemble Analysis
    if args.models:
        print("Loading predictions...")
        models_preds = {}
        for m in args.models:
            df = load_model_predictions(m, args.start_date, args.end_date)
            if not df.empty:
                models_preds[m] = df
            else:
                print(f"Warning: No predictions found for model {m}")
                
        if models_preds:
            # We need standard forward returns for IC
            min_date = min(df.index.get_level_values('datetime').min() for df in models_preds.values()).strftime('%Y-%m-%d')
            max_date = max(df.index.get_level_values('datetime').max() for df in models_preds.values()).strftime('%Y-%m-%d')
            print(f"Fetching forward returns from {min_date} to {max_date}...")
            # We want T+1 returns for IC
            fwd_ret = get_forward_returns(min_date, max_date, n_days=1)
            
            report.append("\n## 1. Single Model Quality")
            for m, df in models_preds.items():
                print(f"Analyzing Single Model: {m}")
                sma = SingleModelAnalyzer(df)
                daily_ic, ic_win, icir = sma.calculate_rank_ic(fwd_ret.dropna())
                ic_decay = sma.calculate_ic_decay()
                spread_df = sma.calculate_quantile_spread(fwd_ret.dropna(), top_q=0.1, bottom_q=0.1)
                long_ic_series, long_ic_mean = sma.calculate_long_only_ic(fwd_ret.dropna(), top_k=22)
                
                report.append(f"\n### Model: {m}")
                report.append(f"- **Rank IC Mean**: {daily_ic.mean() if not daily_ic.empty else 'N/A':.4f}")
                report.append(f"- **ICIR (Information Ratio)**: {icir:.4f}")
                report.append(f"- **IC Win Rate**: {ic_win:.2%}")
                if not spread_df.empty:
                    report.append(f"- **Decile Spread (Top 10% - Bottom 10%)**: {spread_df['Spread'].mean():.4%}")
                report.append(f"- **Long-Only IC (Top 22)**: {long_ic_mean:.4f}")
                report.append("- **IC Decay Curve**:")
                if ic_decay:
                    for k, v in ic_decay.items():
                        if args.shareable:
                            report.append(f"  - {k}: {v:.1f}")
                        else:
                            report.append(f"  - {k}: {v:.4f}")
                else:
                    report.append("  - N/A")
                    
            if len(models_preds) > 1:
                print("Analyzing Ensemble Metrics...")
                report.append("\n## 2. Ensemble & Correlation")
                ea = EnsembleAnalyzer(models_preds)
                corr_matrix = ea.calculate_signal_correlation()
                report.append("\n### Signal Spearman Correlation Matrix")
                if not corr_matrix.empty:
                    report.append("\n| Model | " + " | ".join(corr_matrix.columns) + " |")
                    report.append("| " + " | ".join(["---"] * (len(corr_matrix.columns) + 1)) + " |")
                    for idx, row in corr_matrix.iterrows():
                        if args.shareable:
                            row_vals = [f"{val:.1f}" for val in row]
                        else:
                            row_vals = [f"{val:.3f}" for val in row]
                        report.append(f"| **{idx}** | " + " | ".join(row_vals) + " |")
                else:
                    report.append("*N/A*")
                    
                marginal = ea.calculate_marginal_contribution(fwd_ret.dropna())
                if marginal:
                    report.append("\n### Marginal Contribution to Sharpe (Top 20%)")
                    report.append(f"- **Full Ensemble Equal-Weight Sharpe**: {marginal['Full_Ensemble_Sharpe']:.4f}")
                    for m, drop_sharpe in marginal['Marginal_Contributions'].items():
                        if args.shareable:
                            report.append(f"  - Drop `{m}` -> impact on Sharpe: {drop_sharpe:+.1f}")
                        else:
                            report.append(f"  - Drop `{m}` -> impact on Sharpe: {drop_sharpe:+.4f}")
                        
                ensemble_metrics = ea.calculate_ensemble_ic_metrics(fwd_ret.dropna(), top_k=22, top_q=0.1, bottom_q=0.1)
                if ensemble_metrics:
                    report.append("\n### Ensemble Combined Model Quality")
                    if args.shareable:
                        report.append(f"- **Rank IC Mean**: {ensemble_metrics.get('Rank_IC_Mean', np.nan):.1f}")
                        report.append(f"- **ICIR**: {ensemble_metrics.get('ICIR', np.nan):.1f}")
                        report.append(f"- **IC Win Rate**: {ensemble_metrics.get('IC_Win_Rate', np.nan):.1%}")
                        report.append(f"- **Decile Spread (Top 10% - Bottom 10%)**: {ensemble_metrics.get('Spread_Mean', np.nan):.1%}")
                        report.append(f"- **Long-Only IC (Top 22)**: {ensemble_metrics.get('Long_Only_IC_Mean', np.nan):.1f}")
                    else:
                        report.append(f"- **Rank IC Mean**: {ensemble_metrics.get('Rank_IC_Mean', np.nan):.4f}")
                        report.append(f"- **ICIR**: {ensemble_metrics.get('ICIR', np.nan):.4f}")
                        report.append(f"- **IC Win Rate**: {ensemble_metrics.get('IC_Win_Rate', np.nan):.2%}")
                        report.append(f"- **Decile Spread (Top 10% - Bottom 10%)**: {ensemble_metrics.get('Spread_Mean', np.nan):.4%}")
                        report.append(f"- **Long-Only IC (Top 22)**: {ensemble_metrics.get('Long_Only_IC_Mean', np.nan):.4f}")
                        
    # 3. Execution Friction
    print("Analyzing Execution Friction...")
    exec_a = ExecutionAnalyzer(start_date=args.start_date, end_date=args.end_date)
    slip_df = exec_a.calculate_slippage_and_delay()
    path_df = exec_a.calculate_path_dependency()
    explicit_costs = exec_a.analyze_explicit_costs()
    order_dir = os.path.join(ROOT_DIR, "data", "order_history")
    discrepancy = exec_a.analyze_order_discrepancies(order_dir, market="all")
    
    report.append("\n## 3. Execution Friction & Path Dependency (Quantitative Only)")
    quant_slip_df = pd.DataFrame() # Initialize to avoid UnboundLocalError
    if not slip_df.empty:
        # Drop NaNs across all components simultaneously so denominators exactly match
        slip_df = slip_df.dropna(subset=['Delay_Cost', 'Exec_Slippage', 'Total_Friction', '成交金额'])
        
        # Count total trades for reconciliation context
        if '交易类别' in exec_a.trade_log.columns:
            all_trades_count = len(exec_a.trade_log[exec_a.trade_log['交易类别'].str.contains('买入|卖出', na=False)])
        else:
            all_trades_count = 0
            
        # Exclude manual trades from quant execution friction
        if 'trade_class' in slip_df.columns:
            quant_slip_df = slip_df[slip_df['trade_class'] != 'M'].copy()
            manual_count = len(slip_df[slip_df['trade_class'] == 'M'])
        else:
            quant_slip_df = slip_df.copy()
            manual_count = 0
            
        buy_slip = quant_slip_df[quant_slip_df['交易类别'].str.contains('买入', na=False)]
        sell_slip = quant_slip_df[quant_slip_df['交易类别'].str.contains('卖出', na=False)]
        analyzed_count = len(buy_slip) + len(sell_slip)
        
        if not args.shareable:
            report.append(f"- **Note**: Analyzing {analyzed_count} quantitative trades. {manual_count} manual trade(s) excluded for friction accuracy.")

        def weighted_avg(df, col, weight_col='成交金额'):
            if df.empty or df[weight_col].sum() == 0: return 0.0
            return (df[col] * df[weight_col]).sum() / df[weight_col].sum()
            
        if args.shareable:
            report.append(f"- **Buy Transactions (Quant)**:")
            report.append(f"  - Vol-Weighted Total Implementation Shortfall (IS) (Signal -> Execution): {weighted_avg(buy_slip, 'Total_Friction'):.2%}")
        else:
            report.append(f"- **Buy Transactions**: {format_count(len(buy_slip))}")
            report.append(f"  - Vol-Weighted Delay Cost (Signal Close -> Exec Open): {weighted_avg(buy_slip, 'Delay_Cost'):.4%}")
            report.append(f"  - Vol-Weighted Exec Slippage (Exec Open -> Exec): {weighted_avg(buy_slip, 'Exec_Slippage'):.4%}")
            report.append(f"  - Vol-Weighted Total Friction (Buy): {weighted_avg(buy_slip, 'Total_Friction'):.4%}")
        
        if 'Absolute_Slippage_Amount' in buy_slip.columns and not args.shareable:
            abs_slip_buy = buy_slip['Absolute_Slippage_Amount'].sum()
            report.append(f"  - Absolute Slippage Amount (Total): {abs_slip_buy:.2f}")
            report.append(f"    - Component (Delay Cost): {buy_slip['Abs_Delay_Cost'].sum():+.2f}")
            report.append(f"    - Component (Exec Slippage): {buy_slip['Abs_Exec_Slippage'].sum():+.2f}")

        if 'ADV_Participation_Rate' in buy_slip.columns:
            buy_adv = buy_slip['ADV_Participation_Rate'].dropna()
            if not buy_adv.empty:
                report.append(f"  - ADV Participation Rate (Mean / Max): {format_adv(buy_adv.mean())} / {format_adv(buy_adv.max())}")
        
        if args.shareable:
            report.append(f"- **Sell Transactions (Quant)**:")
            report.append(f"  - Vol-Weighted Total Implementation Shortfall (IS) (Signal -> Execution): {weighted_avg(sell_slip, 'Total_Friction'):.2%}")
        else:
            report.append(f"- **Sell Transactions**: {format_count(len(sell_slip))}")
            report.append(f"  - Vol-Weighted Delay Cost (Signal Close -> Exec Open): {weighted_avg(sell_slip, 'Delay_Cost'):.4%}")
            report.append(f"  - Vol-Weighted Exec Slippage (Exec Open -> Exec): {weighted_avg(sell_slip, 'Exec_Slippage'):.4%}")
            report.append(f"  - Vol-Weighted Total Friction (Sell): {weighted_avg(sell_slip, 'Total_Friction'):.4%}")

        if 'Absolute_Slippage_Amount' in sell_slip.columns and not args.shareable:
            abs_slip_sell = sell_slip['Absolute_Slippage_Amount'].sum()
            report.append(f"  - Absolute Slippage Amount (Total): {abs_slip_sell:.2f}")
            report.append(f"    - Component (Delay Cost): {sell_slip['Abs_Delay_Cost'].sum():+.2f}")
            report.append(f"    - Component (Exec Slippage): {sell_slip['Abs_Exec_Slippage'].sum():+.2f}")

        if 'ADV_Participation_Rate' in sell_slip.columns:
            sell_adv = sell_slip['ADV_Participation_Rate'].dropna()
            if not sell_adv.empty:
                report.append(f"  - ADV Participation Rate (Mean / Max): {format_adv(sell_adv.mean())} / {format_adv(sell_adv.max())}")

    if explicit_costs:
        report.append("\n### Explicit Trading Costs & Dividends")
        fee_str = f" (Total explicit fees amount: {explicit_costs.get('total_fees', 0):.2f})" if not args.shareable else ""
        if args.shareable:
            report.append(f"- **Avg Transaction Fee Rate**: ~{explicit_costs.get('fee_ratio', 0):.2%}")
        else:
            report.append(f"- **Avg Transaction Fee Rate**: {explicit_costs.get('fee_ratio', 0):.4%}{fee_str}")
        
        div_val = explicit_costs.get('total_dividend', 0)
        if args.shareable:
            # Show dividend as offset percentage of total slippage if slippage is available
            abs_slip_buy = quant_slip_df[quant_slip_df['交易类别'].str.contains('买入', na=False)]['Absolute_Slippage_Amount'].sum() if 'Absolute_Slippage_Amount' in quant_slip_df.columns else 0
            abs_slip_sell = quant_slip_df[quant_slip_df['交易类别'].str.contains('卖出', na=False)]['Absolute_Slippage_Amount'].sum() if 'Absolute_Slippage_Amount' in quant_slip_df.columns else 0
            total_abs_slip = abs_slip_buy + abs_slip_sell
            if total_abs_slip > 0:
                offset_pct = div_val / total_abs_slip
                report.append(f"- **Dividend Offset as % of Total Slippage**: {offset_pct:.2%}")
            else:
                report.append(f"- **Dividend Impact**: *Positive (Redacted)*")
        else:
            report.append(f"- **Total Dividend Accumulation (net)**: {div_val:.2f}")
        
    if discrepancy:
        report.append("\n### Order Suggestion vs Actual Discrepancy (Buys)")
        if discrepancy.get('total_missed_count', 0) > 0:
            theo_bias_val = discrepancy.get('theoretical_substitute_bias_impact', 0)
            real_bias_val = discrepancy.get('realized_substitute_bias_impact', 0)
            theo_bias_str = "Lucky/Gain" if theo_bias_val > 0 else "Unlucky/Loss"
            real_bias_str = "Lucky/Gain" if real_bias_val > 0 else "Unlucky/Loss"
            days_str = format_count(discrepancy.get('total_days_with_misses', 0))
            if args.shareable:
                report.append(f"- **Substitution Bias ({theo_bias_str}) (Theoretical)**: {theo_bias_val:.1%}")
                report.append(f"- **Substitution Bias ({real_bias_str}) (Realized with Cost)**: {real_bias_val:.1%}")
                report.append(f"  - Scope: Missed Top Buy Occurrences: {format_count(discrepancy.get('total_missed_count', 0))}.")
                report.append(f"  - Avg Missed Top Buys Expected Return: {discrepancy.get('avg_missed_buys_return', 0):.1%}")
                report.append(f"  - Avg Actual Substitute Buys Return (Theoretical): {discrepancy.get('theoretical_avg_substitute_return', 0):.1%}")
                report.append(f"  - Avg Actual Substitute Buys Return (Realized): {discrepancy.get('realized_avg_substitute_return', 0):.1%}")
            else:
                report.append(f"- **Substitution Bias ({theo_bias_str}) (Theoretical)**: {theo_bias_val:.4%}")
                report.append(f"- **Substitution Bias ({real_bias_str}) (Realized with Cost)**: {real_bias_val:.4%}")
                report.append(f"  - Scope: Missed Top Buy Occurrences: {format_count(discrepancy.get('total_missed_count', 0))}, spread across {days_str} trading days.")
                report.append(f"  - Avg Missed Top Buys Expected Return: {discrepancy.get('avg_missed_buys_return', 0):.4%}")
                report.append(f"  - Avg Actual Substitute Buys Return (Theoretical): {discrepancy.get('theoretical_avg_substitute_return', 0):.4%}")
                report.append(f"  - Avg Actual Substitute Buys Return (Realized): {discrepancy.get('realized_avg_substitute_return', 0):.4%}")
        else:
            report.append("- No measurable substitution bias or date mismatch for buys.")
            
    if not path_df.empty:
        def weighted_avg(df, col, weight_col='成交金额'):
            clean_df = df.dropna(subset=[col, weight_col])
            if clean_df.empty or clean_df[weight_col].sum() == 0: return 0.0
            return (clean_df[col] * clean_df[weight_col]).sum() / clean_df[weight_col].sum()
            
        report.append("\n### Intra-trade Path Excursions")
        if args.shareable:
            report.append(f"- **Vol-Weighted MFE (Max Favorable Relative to Exec)**: {weighted_avg(path_df, 'MFE'):.2%}")
            report.append(f"- **Vol-Weighted MAE (Max Adverse Relative to Exec)**: {weighted_avg(path_df, 'MAE'):.2%}")
        else:
            report.append(f"- **Vol-Weighted MFE (Max Favorable Relative to Exec)**: {weighted_avg(path_df, 'MFE'):.4%}")
            report.append(f"- **Vol-Weighted MAE (Max Adverse Relative to Exec)**: {weighted_avg(path_df, 'MAE'):.4%}")

    # 4. Portfolio Return & Risk
    print("Analyzing Portfolio & Traditional Risk...")
    port_a = PortfolioAnalyzer(start_date=args.start_date, end_date=args.end_date)
    metrics = port_a.calculate_traditional_metrics()
    exposure = port_a.calculate_factor_exposure()
    style_exposure = port_a.calculate_style_exposures()
    if style_exposure:
        exposure.update(style_exposure)
    
    holding_metrics = port_a.calculate_holding_metrics()
    
    report.append("\n## 4. Portfolio Strategy & Returns")
    if holding_metrics:
        report.append("### Holding Analytics")
        for k, v in holding_metrics.items():
            if k == 'Avg_Daily_Holdings_Count':
                if args.shareable:
                    report.append(f"- **{k}**: {format_count(v)}")
                else:
                    report.append(f"- **{k}**: {v:.1f}")
            else:
                report.append(f"- **{k}**: {v:.2%}")

    report.append("\n### Traditional Return & Risk")
    if metrics:
        def format_fuzzy_days(val):
            if pd.isna(val): return "N/A"
            if val < 5: return "< 5 days"
            rounded = int(round(val / 5.0) * 5)
            if rounded == 0: return "~0 days"
            lower = max(0, rounded - 5)
            upper = rounded + 5
            return f"{lower}-{upper} days"

        for k, v in metrics.items():
            if 'CAGR_252' in k:
                name = k.replace('_252', ' (252-day basis)')
                if args.shareable:
                    report.append(f"- **{name}**: {v:.1%}")
                else:
                    report.append(f"- **{name}**: {v:.2%}")
            elif 'CAGR_Calendar' in k:
                name = k.replace('_Calendar', ' (Calendar basis)')
                if args.shareable:
                    report.append(f"- **{name}**: {v:.1%}")
                else:
                    report.append(f"- **{name}**: {v:.2%}")
            elif k in ['Absolute_Return', 'Benchmark_Absolute_Return', 'Volatility', 'Benchmark_Volatility', 'Tracking_Error', 'Max_Drawdown', 'Benchmark_Max_Drawdown', 'Realized_Trade_Win_Rate', 'Daily_Return_Win_Rate', 'Annualized_Active_Return_(Arithmetic)']:
                if args.shareable:
                    report.append(f"- **{k}**: {v:.1%}")
                else:
                    report.append(f"- **{k}**: {v:.2%}")
            elif k == 'Turnover_Rate_Annual':
                if args.shareable:
                    report.append(f"- **{k}**: ~{v:.1f}x")
                else:
                    report.append(f"- **{k}**: {v:.2f}x")
            elif k in ['Max_Time_Under_Water_Days', 'Benchmark_Max_Time_Under_Water_Days', 'Avg_Time_Under_Water_Days', 'Benchmark_Avg_Time_Under_Water_Days', 'Days_Below_Initial_Capital']:
                if args.shareable:
                    report.append(f"- **{k}**: {format_fuzzy_days(v)}")
                else:
                    report.append(f"- **{k}**: {v:.0f}")
            elif 'Daily_Profit_Factor' in k:
                # Handle both Returns and PnL versions
                name = k.replace('_', ' ')
                if args.shareable:
                    report.append(f"- **{name}**: {v:.1f}")
                else:
                    report.append(f"- **{name}**: {v:.4f}")
            else:
                if args.shareable:
                    report.append(f"- **{k}**: {v:.1f}")
                else:
                    report.append(f"- **{k}**: {v:.4f}")
                
    if exposure:
        factor_ann = exposure.pop('Factor_Annualized', {})
        beta = exposure.get('Beta_Market', 0)
        
        report.append(f"\n### Factor Exposure ({'Redacted' if args.shareable else market} Basis)")
        # Removed format_factor qualitative rounding

        for k, v in exposure.items():
            if 'R_Squared' in k:
                if args.shareable:
                    report.append(f"- **{k}**: {v:.2f}")
                else:
                    report.append(f"- **{k}**: {v:.4f}")
            elif 'Alpha' in k or 'Intercept' in k:
                if args.shareable:
                    report.append(f"- **{k}**: {v:.1%}")
                else:
                    report.append(f"- **{k}**: {v:.4%}")
            else:
                if args.shareable:
                    report.append(f"- **{k}**: {v:.2f}")
                else:
                    report.append(f"- **{k}**: {v:.4f}")
                
        if metrics.get('CAGR_252') is not None and metrics.get('Benchmark_CAGR_252') is not None and not pd.isna(metrics.get('CAGR_252')):
            cagr = metrics['CAGR_252']
            
            # Market_Total_Return_Annualized is the arithmetic mean of daily total market returns * 252.
            # This should be > Benchmark_CAG_252 due to volatility drag.
            # Market_Excess_Return_Annualized is used for CAPM-based attribution.
            market_ann_total = exposure.get('Market_Total_Return_Annualized', metrics.get('Benchmark_CAGR_252', 0))
            market_ann_excess = exposure.get('Market_Excess_Return_Annualized', market_ann_total - 0.0135)
            
            # Single-Factor Alpha & Beta Return
            # We use Total Return for the Beta component to stay closer to CAGR components.
            beta_ret_single = beta * market_ann_total
            idio_alpha_single = exposure.get('Annualized_Alpha', 0)
            
            # Multi-Factor Component Calculations
            style_ret = 0.0
            if 'liquidity' in factor_ann and 'momentum' in factor_ann and 'volatility' in factor_ann:
                style_ret += exposure.get(BARRA_LIQD_KEY, 0) * factor_ann['liquidity']
                style_ret += exposure.get(BARRA_MOMT_KEY, 0) * factor_ann['momentum']
                style_ret += exposure.get(BARRA_VOLA_KEY, 0) * factor_ann['volatility']
                
            idio_alpha_multi = exposure.get('Multi_Factor_Intercept', 0)
            multi_beta = exposure.get('Multi_Factor_Beta', 0)
            beta_ret_multi = multi_beta * market_ann_total
            
            # Compute Residual Gaps (Compound/Math mismatch)
            # For Single-Factor, Style Alpha is explicitly zero.
            arithmetic_total_single = beta_ret_single + 0.0 + idio_alpha_single
            cagr_gap_single = cagr - arithmetic_total_single
            
            arithmetic_total_multi = beta_ret_multi + style_ret + idio_alpha_multi
            cagr_gap_multi = cagr - arithmetic_total_multi
            
            report.append("\n### Performance Attribution (Single-Factor Market Beta)")
            if args.shareable:
                report.append(f"- **Total Strategy CAGR**: {cagr:.1%}")
                report.append(f"  - Beta Return (Exposure to Market): {beta_ret_single:.1%}")
                report.append(f"  - Style Alpha (Exposure to Risk Factors): 0.0%")
                report.append(f"  - Idiosyncratic Alpha (Stock Selection / Timing): {idio_alpha_single:.1%}")
                report.append(f"  - Math/Compounding Gap (Residual): {cagr_gap_single:.1%}")
            else:
                report.append(f"- **Total Strategy CAGR**: {cagr:.2%}")
                report.append(f"  - Beta Return (Exposure to Market): {beta_ret_single:.2%}")
                report.append(f"  - Style Alpha (Exposure to Risk Factors): 0.00%")
                report.append(f"  - Idiosyncratic Alpha (Stock Selection / Timing): {idio_alpha_single:.2%}")
                report.append(f"  - Math/Compounding Gap (Residual): {cagr_gap_single:.2%}")
                
            report.append("\n### Performance Attribution (Multi-Factor Strict Alignment)")
            if args.shareable:
                report.append(f"- **Total Strategy CAGR**: {cagr:.1%}")
                report.append(f"  - Beta Return (Multi-Factor Exposure to Market): {beta_ret_multi:.1%}")
                report.append(f"  - Style Alpha (Exposure to Risk Factors): {style_ret:.1%}")
                report.append(f"  - Idiosyncratic Alpha (Stock Selection / Timing): {idio_alpha_multi:.1%}")
                report.append(f"  - Math/Compounding Gap (Residual): {cagr_gap_multi:.1%}")
            else:
                report.append(f"- **Total Strategy CAGR**: {cagr:.2%}")
                report.append(f"  - Beta Return (Multi-Factor Exposure to Market): {beta_ret_multi:.2%}")
                report.append(f"  - Style Alpha (Exposure to Risk Factors): {style_ret:.2%}")
                report.append(f"  - Idiosyncratic Alpha (Stock Selection / Timing): {idio_alpha_multi:.2%}")
                report.append(f"  - Math/Compounding Gap (Residual): {cagr_gap_multi:.2%}")
            
    # 5. Trade Classification & Manual Impact
    print("Analyzing Trade Classification & Manual Impact...")
    class_returns = port_a.calculate_classified_returns()
    if class_returns and 'class_df' in class_returns:
        class_df = class_returns['class_df']
        report.append("\n## 5. Trade Classification & Manual Impact")
        
        # Classification Distribution
        report.append("\n### Classification Distribution")
        if args.shareable:
            report.append("| Class | Count | Pct |")
            report.append("|-------|-------|-----|")
        else:
            col_name = "Total Amount"
            report.append(f"| Class | Count | Pct | {col_name} |")
            report.append("|-------|-------|-----|--------------|")
        
        # We need the full trade log to get amounts
        trade_log = exec_a.trade_log
        if not trade_log.empty and 'trade_class' in trade_log.columns:
            # Only count actual trades (excluding dividends, fees, etc.)
            real_trades = trade_log[trade_log['交易类别'].str.contains('买入|卖出', na=False)]
            total_trades = len(real_trades)
            if total_trades > 0:
                for cls, label in [('S', 'SIGNAL'), ('A', 'SUBSTITUTE'), ('M', 'MANUAL'), ('U', 'UNCLASSIFIED')]:
                    subset = real_trades[real_trades['trade_class'] == cls]
                    count = len(subset)
                    if count == 0 and cls == 'U':
                        continue  # Hide UNCLASSIFIED if 0
                    pct = count / total_trades
                    amt = subset['成交金额'].sum() if '成交金额' in subset.columns else 0.0
                    if args.shareable:
                        report.append(f"| {label} | {format_count(count)} | {pct:.1%} |")
                    else:
                        amt_str = f"¥{amt:,.0f}"
                        report.append(f"| {label} | {format_count(count)} | {pct:.1%} | {amt_str} |")
                    
        # Quantitative Performance & Details
        if not args.shareable:
            quant_cagr_str = "N/A (Rigorous separation coming in v2)"
            report.append("\n### Quantitative-Only Performance")
            report.append(f"- Quant CAGR: {quant_cagr_str}")
            
            # Manual Trade Details
            report.append("\n### Manual Trade Details")
            manual_buys = class_returns['manual_buys']
            manual_sells = class_returns['manual_sells']
            
            if manual_buys.empty and manual_sells.empty:
                report.append("*No manual trades recorded in this period.*")
            else:
                report.append("| Date | Instrument | Direction | Amount |")
                report.append("|------|-----------|-----------|--------|")
                
                # Combine buys and sells for display
                manual_all = pd.concat([manual_buys, manual_sells])
                if not manual_all.empty:
                    manual_all = manual_all.sort_values('成交日期')
                    for _, row in manual_all.iterrows():
                        date_val = row['成交日期'].strftime('%Y-%m-%d') if pd.notna(row['成交日期']) else ""
                        inst = row.get('证券代码', '')
                        direction = "BUY" if '买入' in str(row.get('交易类别', '')) else "SELL"
                        amt = row.get('成交金额', 0)
                        report.append(f"| {date_val} | {inst} | {direction} | ¥{amt:,.0f} |")
                        
                report.append("\n*(Detailed manual trade PnL tracking and T+5 returns require dual-ledger system integration)*")
            
    if args.shareable:
        report.append("\n---\n**Disclaimer**: This report is for informational purposes only and does not constitute financial advice. Sensitive data, including market names, monetary amounts, and specific transaction details, have been redacted or rounded. Performance metrics shown are based on historical data and do not guarantee future results.")

    # Write report
    report_text = "\n".join(report)
    out_path = os.path.join(ROOT_DIR, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report_text)
        
    print(f"\nAnalysis completed successfully. Report written to {out_path}")


if __name__ == "__main__":
    main()
