#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from datetime import datetime

# Adjust path so we can import analysis module
import env
os.chdir(env.ROOT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR
sys.path.append(ROOT_DIR)

from scripts.analysis.utils import init_qlib, get_daily_features
from scripts.analysis.portfolio_analyzer import PortfolioAnalyzer
from scripts.analysis.execution_analyzer import ExecutionAnalyzer

def compute_rolling_metrics(windows=[60, 20], sub_window=20, market='csi300'):
    print(f"Initializing Qlib for Market: {market}...")
    init_qlib()
    
    # 1. Load Portfolio returns
    print("Loading Portfolio Returns...")
    port_a = PortfolioAnalyzer()
    returns = port_a.calculate_daily_returns()
    if returns.empty:
        print("No returns found! Ensure data/daily_amount_log_full.csv is present.")
        return
        
    min_date = returns.index.min().strftime('%Y-%m-%d')
    max_date = returns.index.max().strftime('%Y-%m-%d')
    
    # 2. Market Features & Rolling Risk Exposure (Barra)
    print("Loading Market and Factor Features...")
    features_dict = {'close': '$close', 'volume': '$volume'}
    features = get_daily_features(min_date, max_date, market=market, features=features_dict)
    
    aligned = pd.DataFrame()
    if not features.empty:
        features = features.reset_index()
        features['datetime'] = pd.to_datetime(features['datetime'])
        features = features.sort_values(['instrument', 'datetime'])
        features['size'] = np.log(features['close'] * features['volume'] + 1e-9)
        features['momentum'] = features.groupby('instrument')['close'].pct_change(sub_window)
        features['prev_close'] = features.groupby('instrument')['close'].shift(1)
        features['ret'] = (features['close'] - features['prev_close']) / features['prev_close']
        features['volatility'] = features.groupby('instrument')['ret'].rolling(sub_window, min_periods=5).std().reset_index(0, drop=True)
        features = features.dropna(subset=['ret', 'size', 'momentum', 'volatility'])
        
        factor_returns = {}
        for factor in ['size', 'momentum', 'volatility']:
            def _factor_ret(df):
                if len(df) < 5: return 0.0
                q_top = df[factor].quantile(0.8)
                q_bot = df[factor].quantile(0.2)
                return df[df[factor] >= q_top]['ret'].mean() - df[df[factor] <= q_bot]['ret'].mean()
            factor_returns[factor] = features.groupby('datetime').apply(_factor_ret)
            
        factor_df = pd.DataFrame(factor_returns).fillna(0)
        
        market_close = port_a.daily_amount['CSI300'] if 'CSI300' in port_a.daily_amount.columns else pd.Series(dtype=float)
        if market_close.empty:
            mc = get_daily_features(min_date, max_date, market=market, features={'close': '$close'})
            if not mc.empty:
                mc = mc.reset_index()
                mc['datetime'] = pd.to_datetime(mc['datetime'])
                market_close = mc.groupby('datetime')['close'].mean()
                
        market_ret = market_close.pct_change().reindex(returns.index).fillna(0)
        aligned = pd.concat([returns, factor_df, market_ret.rename('Market')], axis=1).dropna()
        aligned.rename(columns={aligned.columns[0]: 'Portfolio'}, inplace=True)

    # 3. Pre-process Execution Metrics & Win-Rates
    print("Pre-processing Execution & Trade Logs...")
    exec_a = ExecutionAnalyzer()
    slip_df = exec_a.calculate_slippage_and_delay()
    daily_slip = pd.DataFrame()
    if not slip_df.empty:
        slip_df = slip_df.dropna(subset=['Exec_Slippage', 'Delay_Cost', 'Total_Friction'])
        daily_slip = slip_df.groupby('成交日期')[['Exec_Slippage', 'Delay_Cost', 'Total_Friction']].mean()
        daily_slip = daily_slip.reindex(returns.index).fillna(0)
        
    trade_log = exec_a.trade_log
    daily_trades = pd.DataFrame()
    if not trade_log.empty:
        trades = trade_log[trade_log['交易类别'].str.contains('买入|卖出', na=False)].copy()
        trades = trades.sort_values(by=['证券代码', '成交日期'])
        completed_trades = []
        for inst, group in trades.groupby('证券代码'):
            buys = []
            for _, row in group.iterrows():
                if "买入" in row['交易类别']:
                    buys.append({"date": row['成交日期'], "price": row['成交价格'], "shares": row['成交数量']})
                elif "卖出" in row['交易类别']:
                    sell_shares = row['成交数量']
                    sell_price = row['成交价格']
                    sell_date = row['成交日期']
                    while sell_shares > 0 and buys:
                        b = buys[0]
                        match_shares = min(sell_shares, b['shares'])
                        pnl_pct = (sell_price - b['price']) / b['price']
                        completed_trades.append({
                            'Sell_Date': pd.to_datetime(sell_date),
                            'PnL_Pct': pnl_pct,
                            'Profit_Status': 1 if pnl_pct > 0 else 0
                        })
                        b['shares'] -= match_shares
                        sell_shares -= match_shares
                        if b['shares'] == 0:
                            buys.pop(0)
                            
        ct_df = pd.DataFrame(completed_trades)
        if not ct_df.empty:
            ct_df = ct_df.set_index('Sell_Date').sort_index()
            daily_trades = ct_df.groupby(ct_df.index).apply(
                lambda x: pd.Series({
                    'total_trades': len(x),
                    'wins': x['Profit_Status'].sum(),
                    'gross_profit': x[x['PnL_Pct'] > 0]['PnL_Pct'].sum(),
                    'gross_loss': abs(x[x['PnL_Pct'] < 0]['PnL_Pct'].sum()),
                    'win_count': (x['PnL_Pct'] > 0).sum(),
                    'loss_count': (x['PnL_Pct'] < 0).sum(),
                })
            )
            daily_trades = daily_trades.reindex(returns.index).fillna(0)

    # 4. Generate Rolling Windows
    out_dir = os.path.join(ROOT_DIR, 'output')
    os.makedirs(out_dir, exist_ok=True)

    for window in windows:
        print(f"--- Computing metrics for {window}-Day Window ---")
        results_df = pd.DataFrame(index=returns.index)
        results_df['Portfolio_Return'] = returns
        
        # OLS
        if not aligned.empty and len(aligned) > window:
            X = sm.add_constant(aligned[['Market', 'size', 'momentum', 'volatility']])
            y = aligned['Portfolio']
            rolling_model = RollingOLS(y, X, window=window).fit()
            params = rolling_model.params
            results_df['Exposure_Market_Beta'] = params['Market']
            results_df['Exposure_Size'] = params['size']
            results_df['Exposure_Momentum'] = params['momentum']
            results_df['Exposure_Volatility'] = params['volatility']
            
            results_df['Beta_Return'] = params['Market'] * aligned['Market']
            results_df['Style_Alpha'] = (params['size'] * aligned['size']) + \
                                        (params['momentum'] * aligned['momentum']) + \
                                        (params['volatility'] * aligned['volatility'])
            results_df['Idiosyncratic_Alpha'] = aligned['Portfolio'] - results_df['Beta_Return'] - results_df['Style_Alpha']

        # Friction
        if not daily_slip.empty:
            results_df['Exec_Slippage_Mean'] = daily_slip['Exec_Slippage'].rolling(window=window, min_periods=min(10, window)).mean()
            results_df['Delay_Cost_Mean'] = daily_slip['Delay_Cost'].rolling(window=window, min_periods=min(10, window)).mean()
            results_df['Total_Friction_Mean'] = daily_slip['Total_Friction'].rolling(window=window, min_periods=min(10, window)).mean()

        # Win Rates
        if not daily_trades.empty:
            roll_wins = daily_trades['wins'].rolling(window=window, min_periods=min(10, window)).sum()
            roll_tot = daily_trades['total_trades'].rolling(window=window, min_periods=min(10, window)).sum()
            results_df['Win_Rate'] = (roll_wins / roll_tot).replace([np.inf, -np.inf], np.nan)
            
            roll_gross_profit = daily_trades['gross_profit'].rolling(window=window, min_periods=min(10, window)).sum()
            roll_gross_loss = daily_trades['gross_loss'].rolling(window=window, min_periods=min(10, window)).sum()
            roll_win_count = daily_trades['win_count'].rolling(window=window, min_periods=min(10, window)).sum()
            roll_loss_count = daily_trades['loss_count'].rolling(window=window, min_periods=min(10, window)).sum()
            
            avg_win = roll_gross_profit / roll_win_count.replace(0, np.nan)
            avg_loss = roll_gross_loss / roll_loss_count.replace(0, np.nan)
            results_df['Payoff_Ratio'] = (avg_win / avg_loss).replace([np.inf, -np.inf], np.nan)
            
        out_file = os.path.join(out_dir, f'rolling_metrics_{window}.csv')
        results_df.index.name = 'Date'
        results_df = results_df.dropna(how='all') 
        results_df.to_csv(out_file)
        print(f"Successfully wrote {window}-day rolling analysis to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rolling metrics for dashboard visualization.")
    parser.add_argument('--windows', type=int, nargs='+', default=[20, 60], help="List of rolling window sizes (days), e.g., --windows 20 60 120")
    parser.add_argument('--sub-window', type=int, default=20, help="Sub-window size for feature generation, default 20.")
    parser.add_argument('--market', type=str, default='csi300', help="Market benchmark.")
    args = parser.parse_args()
    
    compute_rolling_metrics(windows=args.windows, sub_window=args.sub_window, market=args.market)
