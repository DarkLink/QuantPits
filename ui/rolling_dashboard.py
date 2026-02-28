import os
import glob
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page layout
st.set_page_config(page_title="Rolling Analysis Dashboard", layout="wide")

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantpits.scripts import env

def get_available_windows():
    out_dir = os.path.join(env.ROOT_DIR, "output")
    files = glob.glob(os.path.join(out_dir, "rolling_metrics_*.csv"))
    windows = []
    for f in files:
        basename = os.path.basename(f)
        try:
            # parsing 'rolling_metrics_{window}.csv'
            wstr = basename.replace('rolling_metrics_', '').replace('.csv', '')
            windows.append(int(wstr))
        except ValueError:
            pass
    return sorted(windows)

def load_data(window):
    csv_path = os.path.join(env.ROOT_DIR, "output", f"rolling_metrics_{window}.csv")
    
    if not os.path.exists(csv_path):
        st.error(f"Rolling metrics file not found for window {window}.")
        return pd.DataFrame()
        
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df

def main():
    st.title("🔄 Rolling Strategy Analysis Dashboard")
    st.markdown("Monitor the temporal evolution of strategy behavior, stylistic drift, and systematic friction.")
    
    available_windows = get_available_windows()
    
    if not available_windows:
        st.error("No pre-computed rolling metrics found in `output/` directory. Please run `scripts/run_rolling_analysis.py --windows 20 60` first.")
        return
        
    st.sidebar.header("Rolling Parameters")
    
    # 1. Provide a selectbox to choose window size
    selected_window = st.sidebar.selectbox("Select Rolling Window (Days)", available_windows, index=len(available_windows)-1)
    st.sidebar.info(f"Currently displaying data pre-computed using a {selected_window}-day sliding window (1-day step).")
    
    df = load_data(selected_window)
    if df.empty: return
    
    start_date = st.sidebar.date_input("Filter View Start Date", df.index.min().date())
    end_date = st.sidebar.date_input("Filter View End Date", df.index.max().date())
    
    view_df = df.loc[start_date:end_date].copy()
    if view_df.empty:
        st.warning("No data found for the selected date range.")
        return

    # Section 1: Rolling Factor Exposure
    st.header("1. Rolling Factor Exposure (风格漂移监控)")
    st.markdown("Monitors the OLS beta estimates of the strategy against standard Barra Proxy risk factors.")
    
    fig1 = go.Figure()
    if 'Exposure_Size' in view_df.columns:
        fig1.add_trace(go.Scatter(x=view_df.index, y=view_df['Exposure_Size'], mode='lines', name='Size Exp', fill='tozeroy'))
        fig1.add_trace(go.Scatter(x=view_df.index, y=view_df['Exposure_Momentum'], mode='lines', name='Momentum Exp', fill='tozeroy'))
        fig1.add_trace(go.Scatter(x=view_df.index, y=view_df['Exposure_Volatility'], mode='lines', name='Volatility Exp', fill='tozeroy'))
        
    fig1.update_layout(title=f"Dynamic {selected_window}-Day Barra Factor Exposure", yaxis_title="Factor Beta", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    # Section 2: Rolling Return Attribution
    st.header("2. Rolling Return Attribution (阿尔法生命周期)")
    st.markdown("Decomposes daily returns into Beta, Style Alpha, and Pure Idiosyncratic independent Alpha.")
    
    fig2 = go.Figure()
    if 'Beta_Return' in view_df.columns:
        fig2.add_trace(go.Bar(x=view_df.index, y=view_df['Beta_Return']*100, name='Beta Return', marker_color='blue'))
        fig2.add_trace(go.Bar(x=view_df.index, y=view_df['Style_Alpha']*100, name='Style Alpha', marker_color='orange'))
        fig2.add_trace(go.Bar(x=view_df.index, y=view_df['Idiosyncratic_Alpha']*100, name='Idiosyncratic Alpha', marker_color='green'))
        
    fig2.update_layout(barmode='stack', title=f"{selected_window}-Day Component Return Attribution (%)", yaxis_title="Daily Return Contribution (%)", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    # Section 3: Rolling Friction & Environment
    st.header("3. Trading Friction & Environment (交易摩擦监控)")
    st.markdown(f"Monitors execution slippage and delay cost averages over a rolling {selected_window}-day window.")
    
    fig3 = go.Figure()
    if 'Delay_Cost_Mean' in view_df.columns:
        fig3.add_trace(go.Scatter(x=view_df.index, y=view_df['Delay_Cost_Mean']*100, mode='lines', name='Delay Cost Mean', line=dict(color='orange')))
        fig3.add_trace(go.Scatter(x=view_df.index, y=view_df['Exec_Slippage_Mean']*100, mode='lines', name='Execution Slippage Mean', line=dict(color='red')))
        fig3.add_trace(go.Scatter(x=view_df.index, y=view_df['Total_Friction_Mean']*100, mode='lines', name='Total Friction Mean', line=dict(color='purple', dash='dot')))
        
    fig3.update_layout(title=f"Rolling {selected_window}-Day Execution Friction Averages (%)", yaxis_title="Friction Cost Rate (%)", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    # Section 4: Rolling Win-Rate vs Payoff Ratio
    st.header("4. Win-Rate vs. Payoff Ratio (胜率与盈亏比剪刀差)")
    st.markdown("Examines if hit rate and payoff ratio structurally drift in opposite directions.")
    
    if 'Win_Rate' in view_df.columns and 'Payoff_Ratio' in view_df.columns:
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Left axis = Win Rate
        fig4.add_trace(go.Scatter(x=view_df.index, y=view_df['Win_Rate'], name="Win Rate (%)", mode="lines", line=dict(color="blue")), secondary_y=False)
        # Right axis = Payoff Ratio
        fig4.add_trace(go.Scatter(x=view_df.index, y=view_df['Payoff_Ratio'], name="Payoff Ratio (W/L)", mode="lines", line=dict(color="green")), secondary_y=True)
        
        fig4.update_layout(title=f"Rolling {selected_window}-Day Win Rate vs Payoff Ratio", template="plotly_white")
        fig4.update_yaxes(title_text="Win Rate (%)", tickformat=".1%", secondary_y=False)
        fig4.update_yaxes(title_text="Payoff Ratio", secondary_y=True)
        st.plotly_chart(fig4, use_container_width=True)

if __name__ == "__main__":
    main()
