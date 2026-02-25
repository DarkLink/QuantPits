import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime

# Setup paths to import from backend
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from scripts import env
os.chdir(env.ROOT_DIR)

# Must initialize Qlib before importing analyzers if they fetch data
from scripts.analysis.utils import init_qlib, load_market_config
try:
    init_qlib()
except Exception as e:
    st.warning(f"Qlib initialization might have already run or failed: {e}")

from scripts.analysis.portfolio_analyzer import PortfolioAnalyzer
from scripts.analysis.execution_analyzer import ExecutionAnalyzer

st.set_page_config(page_title="Quantitative Strategy Dashboard", layout="wide", initial_sidebar_state="expanded")

def main():
    st.title("📈 Quantitative Strategy Analysis Dashboard")
    st.markdown("Interactive multi-dimensional visualization for strategy review.")

    # Sidebar parameters
    st.sidebar.header("Analysis Parameters")
    
    default_start = datetime(2025, 1, 1)
    default_end = datetime.today()
    
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", default_end)
    
    # 从配置文件读取默认市场
    config_market, config_benchmark = load_market_config()
    market_options = list(dict.fromkeys([config_market, "csi300", "csi500"]))
    market = st.sidebar.selectbox("Market Benchmark", market_options)
    
    run_btn = st.sidebar.button("Generate Dashboard", type="primary")
    
    if run_btn:
        with st.spinner("Fetching data and computing metrics..."):
            port_a = PortfolioAnalyzer(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            exec_a = ExecutionAnalyzer(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
            
            returns = port_a.calculate_daily_returns()
            if returns.empty:
                st.error("No return data found for the selected period!")
                return
                
            render_macro_performance(port_a, returns, market)
            render_micro_execution(exec_a)
            render_factor_exposure(port_a, market)
            render_holdings_and_trades(port_a, exec_a)

def render_macro_performance(port_a, returns, market):
    st.header("Module 1: Macro Performance & Risk")
    
    daily_amount = port_a.daily_amount
    # 根据市场名称确定基准列名
    market_col = market.upper() if market else 'CSI300'
    
    bench_returns = pd.Series(0.0, index=returns.index)
    if market_col in daily_amount.columns:
        market_close = daily_amount[market_col].astype(float)
        market_close = market_close.loc[returns.index].dropna()
        if len(market_close) >= 2:
            bench_returns = market_close.pct_change().fillna(0)
    
    # 1. Cumulative Returns on Log Scale
    cum_ret = (1.0 + returns).cumprod()
    bench_cum = (1.0 + bench_returns).cumprod()
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret.values, mode='lines', name='Strategy', line=dict(color='blue', width=2)))
    if not bench_returns.eq(0).all():
        fig1.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, mode='lines', name=f'Benchmark ({market_col})', line=dict(color='orange', width=2)))
        
    fig1.update_layout(title="Cumulative Returns on Log Scale", yaxis_type="log", yaxis_title="Cumulative Return (Log)", xaxis_title="Date", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Underwater Plot
    rolling_max = cum_ret.cummax()
    drawdowns = (cum_ret / rolling_max) - 1.0
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns.values, fill='tozeroy', mode='lines', name='Drawdown', line=dict(color='red', width=1), fillcolor='rgba(255, 0, 0, 0.3)'))
    fig2.update_layout(title="Underwater Plot (Drawdown)", yaxis_title="Drawdown (%)", xaxis_title="Date", template="plotly_white")
    fig2.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Rolling Metrics (20-Day Sharpe)
    rolling_window = 20
    rf_daily = 0.0135 / 252
    
    rolling_sharpe = ((returns.rolling(window=rolling_window).mean() - rf_daily) / returns.rolling(window=rolling_window).std()) * np.sqrt(252)
    rolling_alpha = (returns.rolling(window=rolling_window).mean() - bench_returns.rolling(window=rolling_window).mean()) * 252
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, mode='lines', name=f'{rolling_window}d Sharpe', line=dict(color='purple')))
    fig3.add_trace(go.Scatter(x=rolling_alpha.index, y=rolling_alpha.values, mode='lines', name=f'{rolling_window}d Alpha', line=dict(color='green'), yaxis="y2"))
    fig3.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    
    fig3.update_layout(
        title=f"Rolling {rolling_window}-Day Sharpe Ratio & Alpha", 
        xaxis_title="Date", 
        yaxis_title="Sharpe Ratio",
        yaxis2=dict(title="Annualized Alpha", overlaying='y', side='right', tickformat=".2%"),
        template="plotly_white"
    )
    st.plotly_chart(fig3, use_container_width=True)

def render_micro_execution(exec_a):
    st.header("Module 2: Micro-Execution & Friction")
    
    # 1. MFE vs MAE Scatter Plot
    path_df = exec_a.calculate_path_dependency()
    if not path_df.empty:
        path_df = path_df.dropna(subset=['MFE', 'MAE'])
        
        # Determine Trade result. Since we don't have perfect trade matching here,
        # A good proxy is if MFE > abs(MAE), it's "profitable" for the day. (Green vs Red)
        path_df['Trade_Type'] = np.where(path_df['交易类别'].str.contains('买入'), 'Buy', 'Sell')
        path_df['Is_Profitable_Intraday'] = np.where(path_df['MFE'] > path_df['MAE'].abs(), 'Profitable', 'Loss')
        
        fig1 = px.scatter(
            path_df, 
            x='MAE', 
            y='MFE', 
            color='Is_Profitable_Intraday', 
            hover_data=['证券代码', '成交日期', '成交价格', 'Trade_Type'],
            title="MFE vs. MAE Scatter (Intraday Path Dependency)",
            color_discrete_map={'Profitable': 'green', 'Loss': 'red'}
        )
        fig1.update_xaxes(tickformat=".2%")
        fig1.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig1, use_container_width=True)
        
    # 2. Execution Friction Distribution (Histogram + KDE)
    slip_df = exec_a.calculate_slippage_and_delay()
    if not slip_df.empty:
        slip_df = slip_df.dropna(subset=['Exec_Slippage', 'Delay_Cost'])
        slip_df['Exec_Slippage_%'] = slip_df['Exec_Slippage'] * 100
        slip_df['Delay_Cost_%'] = slip_df['Delay_Cost'] * 100
        
        # Replace inf with nan
        slip_df = slip_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Exec_Slippage_%', 'Delay_Cost_%'])
        
        fig2 = ff.create_distplot([slip_df['Delay_Cost_%'].values], ['Delay Cost'], show_hist=True, show_rug=False, colors=['orange'])
        fig2.update_layout(title="Delay Cost Distribution (%)", template="plotly_white", xaxis_title="Cost (%)", yaxis_title="Density")
        st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = ff.create_distplot([slip_df['Exec_Slippage_%'].values], ['Execution Slippage'], show_hist=True, show_rug=False, colors=['blue'])
        fig3.update_layout(title="Execution Slippage Distribution (%)", template="plotly_white", xaxis_title="Slippage (%)", yaxis_title="Density")
        st.plotly_chart(fig3, use_container_width=True)

    # 3. Substitution Bias Tracking
    order_dir = os.path.join(env.ROOT_DIR, "data", "order_history")
    if os.path.exists(order_dir):
        discrepancy = exec_a.analyze_order_discrepancies(order_dir)
        if discrepancy and discrepancy.get('total_missed_count', 0) > 0:
            st.write(f"**Total Missed Top Buys**: {discrepancy.get('total_missed_count')}")
            
            bar_data = pd.DataFrame({
                'Category': ['Missed Top Buys Expected Return', 'Actual Substitute Buys Return'],
                'Return': [discrepancy.get('avg_missed_buys_return', 0), discrepancy.get('avg_substitute_buys_return', 0)]
            })
            
            fig4 = px.bar(bar_data, x='Category', y='Return', color='Category', title="Substitution Bias Tracking (Avg 5-Day Forward Return)")
            fig4.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig4, use_container_width=True)

def render_factor_exposure(port_a, market):
    st.header("Module 3: Factor Exposure & Attribution")
    
    returns = port_a.calculate_daily_returns()
    if returns.empty: return
    
    min_date = returns.index.min().strftime('%Y-%m-%d')
    max_date = returns.index.max().strftime('%Y-%m-%d')
    
    from scripts.analysis.utils import get_daily_features
    features_dict = {'close': '$close', 'volume': '$volume'}
    features = get_daily_features(min_date, max_date, market=market, features=features_dict)
    
    if not features.empty:
        features = features.reset_index()
        features['datetime'] = pd.to_datetime(features['datetime'])
        features = features.sort_values(['instrument', 'datetime'])
        features['size'] = np.log(features['close'] * features['volume'] + 1e-9)
        features['momentum'] = features.groupby('instrument')['close'].pct_change(20)
        features['prev_close'] = features.groupby('instrument')['close'].shift(1)
        features['ret'] = (features['close'] - features['prev_close']) / features['prev_close']
        features['volatility'] = features.groupby('instrument')['ret'].rolling(20, min_periods=5).std().reset_index(0, drop=True)
        features = features.dropna(subset=['ret', 'size', 'momentum', 'volatility'])
        
        factor_returns = {}
        for factor in ['size', 'momentum', 'volatility']:
            def _factor_ret(df):
                if len(df) < 5: return 0.0
                q_top, q_bot = df[factor].quantile(0.8), df[factor].quantile(0.2)
                return df[df[factor] >= q_top]['ret'].mean() - df[df[factor] <= q_bot]['ret'].mean()
            factor_returns[factor] = features.groupby('datetime').apply(_factor_ret)
            
        factor_df = pd.DataFrame(factor_returns).fillna(0)
        aligned = pd.concat([returns, factor_df], axis=1).dropna()
        aligned.rename(columns={aligned.columns[0]: 'Portfolio'}, inplace=True)
        
        # Market for beta
        import statsmodels.api as sm
        market_close = get_daily_features(min_date, max_date, market=market, features={'close': '$close'})
        if not market_close.empty:
            market_close = market_close.reset_index()
            market_close['datetime'] = pd.to_datetime(market_close['datetime'])
            market_ret = market_close.groupby('datetime')['close'].mean().pct_change().fillna(0)
            
            aligned = pd.concat([aligned, market_ret.rename('Market')], axis=1).dropna()
            
            # 1. Dynamic Rolling Regression (20 Days)
            rolling_window = 20
            from statsmodels.regression.rolling import RollingOLS
            
            X = sm.add_constant(aligned[['Market', 'size', 'momentum', 'volatility']])
            y = aligned['Portfolio']
            
            if len(y) > rolling_window:
                rolling_model = RollingOLS(y, X, window=rolling_window).fit()
                params = rolling_model.params.dropna()
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=params.index, y=params['size'], mode='lines', name='Size'))
                fig1.add_trace(go.Scatter(x=params.index, y=params['momentum'], mode='lines', name='Momentum'))
                fig1.add_trace(go.Scatter(x=params.index, y=params['volatility'], mode='lines', name='Volatility'))
                fig1.update_layout(title=f"Dynamic Barra Factor Exposures (Rolling {rolling_window}-Day OLS Beta)", template="plotly_white")
                st.plotly_chart(fig1, use_container_width=True)
                
            # 2. Cumulative Return Attribution Stacked Area
            static_model = sm.OLS(y, X).fit()
            global_beta = static_model.params['Market']
            global_size = static_model.params['size']
            global_mom = static_model.params['momentum']
            global_vol = static_model.params['volatility']
            
            aligned['Beta_Ret'] = global_beta * aligned['Market']
            aligned['Style_Alpha'] = global_size * aligned['size'] + global_mom * aligned['momentum'] + global_vol * aligned['volatility']
            aligned['Idio_Alpha'] = aligned['Portfolio'] - aligned['Beta_Ret'] - aligned['Style_Alpha']
            
            cum_beta = (1 + aligned['Beta_Ret']).cumprod() - 1
            cum_style = (1 + aligned['Style_Alpha']).cumprod() - 1
            cum_idio = (1 + aligned['Idio_Alpha']).cumprod() - 1
            
            # Use stackgroup to stack them
            # Convert cumulative isolated returns to daily contributions, then stack
            fig2 = go.Figure()
            # Stacking raw returns is better for additive stacked area, not cumprod. 
            # We will cumsum the returns for additive stacking
            cum_beta_add = aligned['Beta_Ret'].cumsum()
            cum_style_add = aligned['Style_Alpha'].cumsum()
            cum_idio_add = aligned['Idio_Alpha'].cumsum()
            
            fig2.add_trace(go.Scatter(x=cum_beta_add.index, y=cum_beta_add.values, mode='lines', stackgroup='one', name='Beta Return', line=dict(color='blue')))
            fig2.add_trace(go.Scatter(x=cum_style_add.index, y=cum_style_add.values, mode='lines', stackgroup='one', name='Style Alpha', line=dict(color='orange')))
            fig2.add_trace(go.Scatter(x=cum_idio_add.index, y=cum_idio_add.values, mode='lines', stackgroup='one', name='Idio Alpha', line=dict(color='green')))
            
            fig2.update_layout(title="Cumulative Return Attribution (Additive Stacked Area)", yaxis_title="Cumulative Return (Additive)", template="plotly_white")
            fig2.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig2, use_container_width=True)

def render_holdings_and_trades(port_a, exec_a):
    st.header("Module 4: Holdings & Trade Analytics")
    
    # 1. Holding Count & Top1 Concentration 
    holding_log = port_a.holding_log
    if not holding_log.empty:
        df = holding_log[holding_log['证券代码'] != 'CASH'].copy()
        if not df.empty:
            daily_groups = df.groupby('成交日期')
            daily_count = daily_groups.size()
            
            def top1_conc(g):
                total_val = g['收盘价值'].sum()
                if total_val == 0: return 0
                max_val = g['收盘价值'].max()
                return max_val / total_val
                
            concentration = daily_groups.apply(top1_conc)
            
            # Dual Y-axis
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(go.Scatter(x=daily_count.index, y=daily_count.values, name="Holdings Count", mode="lines", line=dict(color="blue")), secondary_y=False)
            fig1.add_trace(go.Scatter(x=concentration.index, y=concentration.values, name="Top1 Concentration", mode="lines", line=dict(color="red")), secondary_y=True)
            
            fig1.update_layout(title="Holdings Count & Concentration Over Time", template="plotly_white")
            fig1.update_yaxes(title_text="Count", secondary_y=False)
            fig1.update_yaxes(title_text="Concentration", tickformat=".1%", secondary_y=True)
            st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Holding Duration vs PnL & Win Rate Heatmap (FIFO Matching)
    trade_log = exec_a.trade_log
    if not trade_log.empty:
        trades = trade_log[trade_log['交易类别'].str.contains('买入|卖出', na=False)].copy()
        trades = trades.sort_values(by=['证券代码', '成交日期'])
        
        completed_trades = []
        
        # Simple FIFO matching per instrument to calculate trade roundtrip PnL and Duration
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
                        duration = (pd.to_datetime(sell_date) - pd.to_datetime(b['date'])).days
                        
                        completed_trades.append({
                            'Instrument': inst,
                            'Buy_Date': pd.to_datetime(b['date']),
                            'Sell_Date': pd.to_datetime(sell_date),
                            'Duration': duration,
                            'PnL_Pct': pnl_pct,
                            'Profit_Status': 'Win' if pnl_pct > 0 else 'Loss'
                        })
                        
                        b['shares'] -= match_shares
                        sell_shares -= match_shares
                        if b['shares'] == 0:
                            buys.pop(0)

        ct_df = pd.DataFrame(completed_trades)
        if not ct_df.empty:
            # Holding Duration vs PnL Scatter
            fig2 = px.scatter(
                ct_df, 
                x='Duration', 
                y='PnL_Pct', 
                color='Profit_Status', 
                hover_data=['Instrument', 'Buy_Date', 'Sell_Date'],
                title="Holding Duration vs PnL Scatter Plot",
                color_discrete_map={'Win': 'green', 'Loss': 'red'}
            )
            fig2.update_layout(xaxis_title="Holding Duration (Days)", yaxis_title="Trade Return", template="plotly_white")
            fig2.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Win Rate Heatmap (Month vs Day of Week)
            ct_df['Buy_Month'] = ct_df['Buy_Date'].dt.month
            ct_df['Buy_DOW'] = ct_df['Buy_Date'].dt.day_name()
            
            # Calculate Win Rate percentage
            heatmap_data = ct_df.groupby(['Buy_Month', 'Buy_DOW'])['Profit_Status'].apply(lambda x: (x == 'Win').mean() * 100).reset_index(name='Win_Rate')
            
            # Pivot table for heatmap 
            # Force DOW order
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            pivot_df = heatmap_data.pivot(index='Buy_DOW', columns='Buy_Month', values='Win_Rate').reindex(dow_order)
            
            fig3 = px.imshow(pivot_df, 
                             labels=dict(x="Month", y="Day of Week", color="Win Rate (%)"),
                             x=pivot_df.columns, 
                             y=pivot_df.index,
                             text_auto=".1f",
                             aspect="auto",
                             color_continuous_scale="RdYlGn",
                             title="Static Calendar Effect: Win Rate by Buy-Month vs Day of Week")
            
            st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
