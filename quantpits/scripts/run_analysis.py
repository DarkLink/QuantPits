#!/usr/bin/env python3
import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Adjust path so we can import analysis module
import env
os.chdir(env.ROOT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(PROJECT_ROOT)

from quantpits.scripts.analysis.utils import init_qlib, load_model_predictions, get_forward_returns, load_market_config
from quantpits.scripts.analysis.single_model_analyzer import SingleModelAnalyzer
from quantpits.scripts.analysis.ensemble_analyzer import EnsembleAnalyzer
from quantpits.scripts.analysis.execution_analyzer import ExecutionAnalyzer
from quantpits.scripts.analysis.portfolio_analyzer import PortfolioAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Analysis Module")
    parser.add_argument('--models', type=str, nargs='+', help="Model names to analyze (e.g., gru mlp tabnet)")
    parser.add_argument('--start-date', type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end-date', type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument('--output', type=str, default="output/analysis_report.md", help="Output markdown file path")
    args = parser.parse_args()

    print("Initializing Qlib...")
    init_qlib()
    
    # 从配置文件读取市场和基准
    market, benchmark = load_market_config()
    print(f"Market: {market}, Benchmark: {benchmark}")
    
    report = [f"# Comprehensive Analysis Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"]
    report.append("\n## Analysis Scope")
    report.append(f"- Models: {args.models if args.models else 'None (Portfolio/Execution Only)'}")
    report.append(f"- Date Range: {args.start_date or 'Auto'} to {args.end_date or 'Auto'}")
    
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
                        row_vals = [f"{val:.3f}" for val in row]
                        report.append(f"| **{idx}** | " + " | ".join(row_vals) + " |")
                else:
                    report.append("*N/A*")
                    
                marginal = ea.calculate_marginal_contribution(fwd_ret.dropna())
                if marginal:
                    report.append("\n### Marginal Contribution to Sharpe (Top 20%)")
                    report.append(f"- **Full Ensemble Equal-Weight Sharpe**: {marginal['Full_Ensemble_Sharpe']:.4f}")
                    for m, drop_sharpe in marginal['Marginal_Contributions'].items():
                        report.append(f"  - Drop `{m}` -> impact on Sharpe: {drop_sharpe:+.4f}")
                        
                ensemble_metrics = ea.calculate_ensemble_ic_metrics(fwd_ret.dropna(), top_k=22, top_q=0.1, bottom_q=0.1)
                if ensemble_metrics:
                    report.append("\n### Ensemble Combined Model Quality")
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
    
    report.append("\n## 3. Execution Friction & Path Dependency")
    if not slip_df.empty:
        # Drop NaNs across all components simultaneously so denominators exactly match
        slip_df = slip_df.dropna(subset=['Delay_Cost', 'Exec_Slippage', 'Total_Friction', '成交金额'])
        
        # Exclude manual trades from quant execution friction
        if 'trade_class' in slip_df.columns:
            quant_slip_df = slip_df[slip_df['trade_class'] != 'M'].copy()
        else:
            quant_slip_df = slip_df.copy()
            
        buy_slip = quant_slip_df[quant_slip_df['交易类别'].str.contains('买入', na=False)]
        sell_slip = quant_slip_df[quant_slip_df['交易类别'].str.contains('卖出', na=False)]
        
        def weighted_avg(df, col, weight_col='成交金额'):
            if df.empty or df[weight_col].sum() == 0: return 0.0
            return (df[col] * df[weight_col]).sum() / df[weight_col].sum()
            
        report.append(f"- **Buy Transactions**: {len(buy_slip)}")
        report.append(f"  - Vol-Weighted Delay Cost (Signal Close -> Exec Open): {weighted_avg(buy_slip, 'Delay_Cost'):.4%}")
        report.append(f"  - Vol-Weighted Exec Slippage (Exec Open -> Exec): {weighted_avg(buy_slip, 'Exec_Slippage'):.4%}")
        report.append(f"  - Vol-Weighted Total Friction (Buy): {weighted_avg(buy_slip, 'Total_Friction'):.4%}")
        if 'Absolute_Slippage_Amount' in buy_slip.columns:
            abs_slip_buy = buy_slip['Absolute_Slippage_Amount'].sum()
            report.append(f"  - Absolute Slippage Amount: {abs_slip_buy:.2f}")
        if 'ADV_Participation_Rate' in buy_slip.columns:
            buy_adv = buy_slip['ADV_Participation_Rate'].dropna()
            if not buy_adv.empty:
                report.append(f"  - ADV Participation Rate (Mean / Max): {buy_adv.mean():.4%} / {buy_adv.max():.4%}")
        
        report.append(f"- **Sell Transactions**: {len(sell_slip)}")
        report.append(f"  - Vol-Weighted Delay Cost (Signal Close -> Exec Open): {weighted_avg(sell_slip, 'Delay_Cost'):.4%}")
        report.append(f"  - Vol-Weighted Exec Slippage (Exec Open -> Exec): {weighted_avg(sell_slip, 'Exec_Slippage'):.4%}")
        report.append(f"  - Vol-Weighted Total Friction (Sell): {weighted_avg(sell_slip, 'Total_Friction'):.4%}")
        if 'Absolute_Slippage_Amount' in sell_slip.columns:
            abs_slip_sell = sell_slip['Absolute_Slippage_Amount'].sum()
            report.append(f"  - Absolute Slippage Amount: {abs_slip_sell:.2f}")
        if 'ADV_Participation_Rate' in sell_slip.columns:
            sell_adv = sell_slip['ADV_Participation_Rate'].dropna()
            if not sell_adv.empty:
                report.append(f"  - ADV Participation Rate (Mean / Max): {sell_adv.mean():.4%} / {sell_adv.max():.4%}")

    if explicit_costs:
        report.append("\n### Explicit Trading Costs & Dividends")
        report.append(f"- **Avg Transaction Fee Rate**: {explicit_costs.get('fee_ratio', 0):.4%} (Total explicit fees amount: {explicit_costs.get('total_fees', 0):.2f})")
        report.append(f"- **Total Dividend Accumulation (net)**: {explicit_costs.get('total_dividend', 0):.2f}")
        
    if discrepancy:
        report.append("\n### Order Suggestion vs Actual Discrepancy (Buys)")
        if discrepancy.get('total_missed_count', 0) > 0:
            bias_val = discrepancy.get('substitute_bias_impact', 0)
            bias_str = "Lucky/Gain" if bias_val > 0 else "Unlucky/Loss"
            report.append(f"- **Substitution Bias ({bias_str})**: {bias_val:.4%}")
            report.append(f"  - Scope: Missed Top Buy Occurrences: {discrepancy.get('total_missed_count', 0)}, spread across {discrepancy.get('total_days_with_misses', 0)} trading days.")
            report.append(f"  - Avg Missed Top Buys Expected Return: {discrepancy.get('avg_missed_buys_return', 0):.4%}")
            report.append(f"  - Avg Actual Substitute Buys Return: {discrepancy.get('avg_substitute_buys_return', 0):.4%}")
        else:
            report.append("- No measurable substitution bias or date mismatch for buys.")
            
    if not path_df.empty:
        def weighted_avg(df, col, weight_col='成交金额'):
            clean_df = df.dropna(subset=[col, weight_col])
            if clean_df.empty or clean_df[weight_col].sum() == 0: return 0.0
            return (clean_df[col] * clean_df[weight_col]).sum() / clean_df[weight_col].sum()
            
        report.append("\n### Intra-trade Path Excursions")
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
                report.append(f"- **{k}**: {v:.1f}")
            else:
                report.append(f"- **{k}**: {v:.2%}")

    report.append("\n### Traditional Return & Risk")
    if metrics:
        for k, v in metrics.items():
            if k in ['CAGR', 'Excess_Return_CAGR']:
                report.append(f"- **{k} (252-day basis)**: {v:.2%}")
            elif k in ['Absolute_Return', 'Benchmark_Absolute_Return', 'Benchmark_CAGR', 'Volatility', 'Benchmark_Volatility', 'Tracking_Error', 'Max_Drawdown', 'Benchmark_Max_Drawdown', 'Realized_Trade_Win_Rate', 'Turnover_Rate_Annual']:
                report.append(f"- **{k}**: {v:.2%}")
            elif k in ['Max_Time_Under_Water_Days', 'Benchmark_Max_Time_Under_Water_Days', 'Avg_Time_Under_Water_Days', 'Benchmark_Avg_Time_Under_Water_Days', 'Days_Below_Initial_Capital']:
                report.append(f"- **{k}**: {v:.0f}")
            else:
                report.append(f"- **{k}**: {v:.4f}")
                
    if exposure:
        factor_ann = exposure.pop('Factor_Annualized', {})
        beta = exposure.get('Beta_Market', 0)
        
        report.append(f"\n### Factor Exposure ({market} Basis)")
        for k, v in exposure.items():
            if 'R_Squared' in k:
                report.append(f"- **{k}**: {v:.4f}")
            elif 'Alpha' in k or 'Intercept' in k:
                report.append(f"- **{k}**: {v:.4%}")
            else:
                report.append(f"- **{k}**: {v:.4f}")
                
        if metrics.get('CAGR') is not None and metrics.get('Benchmark_CAGR') is not None and not pd.isna(metrics.get('CAGR')):
            cagr = metrics['CAGR']
            
            # Use Benchmark_CAGR and standalone Beta_Market to match expectations intuitively
            beta_ret = beta * metrics.get('Benchmark_CAGR', 0)
            
            style_ret = 0.0
            if 'size' in factor_ann and 'momentum' in factor_ann and 'volatility' in factor_ann:
                style_ret += exposure.get('Barra_Size_Exp', 0) * factor_ann['size']
                style_ret += exposure.get('Barra_Momentum_Exp', 0) * factor_ann['momentum']
                style_ret += exposure.get('Barra_Volatility_Exp', 0) * factor_ann['volatility']
                
            # Intercept is already Annualized since we multiply by 252 in portfolio_analyzer
            idio_alpha = exposure.get('Multi_Factor_Intercept', 0)
            
            # The arithmetic sum of components vs actual geometric CAGR gap
            # This completely patches the "Yield Leakage" by fully resolving the math
            arithmetic_total = beta_ret + style_ret + idio_alpha
            cagr_gap = cagr - arithmetic_total
            
            report.append("\n### Performance Attribution")
            report.append(f"- **Total Strategy CAGR**: {cagr:.2%}")
            report.append(f"  - Beta Return (Exposure to Market): {beta_ret:.2%}")
            report.append(f"  - Style Alpha (Exposure to Risk Factors): {style_ret:.2%}")
            report.append(f"  - Idiosyncratic Alpha (Stock Selection / Timing): {idio_alpha:.2%}")
            report.append(f"  - Math/Compounding Gap (Residual): {cagr_gap:.2%}")
            
    # 5. Trade Classification & Manual Impact
    print("Analyzing Trade Classification & Manual Impact...")
    class_returns = port_a.calculate_classified_returns()
    if class_returns and 'class_df' in class_returns:
        class_df = class_returns['class_df']
        report.append("\n## 5. Trade Classification & Manual Impact")
        
        # Classification Distribution
        report.append("\n### Classification Distribution")
        report.append("| Class | Count | Pct | Total Amount |")
        report.append("|-------|-------|-----|--------------|")
        
        # We need the full trade log to get amounts
        trade_log = exec_a.trade_log
        if not trade_log.empty and 'trade_class' in trade_log.columns:
            total_trades = len(trade_log)
            if total_trades > 0:
                for cls, label in [('S', 'SIGNAL'), ('A', 'SUBSTITUTE'), ('M', 'MANUAL')]:
                    subset = trade_log[trade_log['trade_class'] == cls]
                    count = len(subset)
                    pct = count / total_trades
                    amt = subset['成交金额'].sum() if '成交金额' in subset.columns else 0.0
                    report.append(f"| {label} | {count} | {pct:.1%} | ¥{amt:,.0f} |")
                    
        # Quantitative Performance
        quant_cagr_str = "N/A (Rigorous separation coming in v2)"
        report.append("\n### Quantitative-Only Performance")
        report.append(f"- Quant CAGR: {quant_cagr_str}")
        
        # Manual Trade Details
        manual_buys = class_returns['manual_buys']
        manual_sells = class_returns['manual_sells']
        
        report.append("\n### Manual Trade Details")
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
            
    # Write report
    report_text = "\n".join(report)
    out_path = os.path.join(ROOT_DIR, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report_text)
        
    print(f"\nAnalysis completed successfully. Report written to {out_path}")


if __name__ == "__main__":
    main()
