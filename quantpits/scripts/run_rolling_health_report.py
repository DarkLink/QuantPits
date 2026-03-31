#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

try:
    from quantpits.utils import env
except ImportError:
    from quantpits.utils import env
os.chdir(env.ROOT_DIR)

from quantpits.scripts.analysis.portfolio_analyzer import BARRA_LIQD_KEY

def evaluate_health():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(env.ROOT_DIR, "output")
    
    file_20 = os.path.join(output_dir, "rolling_metrics_20.csv")
    file_60 = os.path.join(output_dir, "rolling_metrics_60.csv")
    
    if not os.path.exists(file_20) or not os.path.exists(file_60):
        print(f"FAILED: CSVs not found. 20: {os.path.exists(file_20)}, 60: {os.path.exists(file_60)}")
        print(f"Path 20: {file_20}")
        return
        
    df_20 = pd.read_csv(file_20, parse_dates=['Date']).set_index('Date').sort_index()
    df_60 = pd.read_csv(file_60, parse_dates=['Date']).set_index('Date').sort_index()
    
    print(f"DEBUG: len(df_20)={len(df_20)}, len(df_60)={len(df_60)}")
    
    # Need at least 60 days of data
    if len(df_60) < 60:
        print(f"DEBUG: Too short: {len(df_60)}")
        return
        
    last_date = df_20.index[-1]
    
    # Initialize metrics with defaults to avoid NameError
    curr_wr, prev_wr, wr_trend = 0.0, 0.0, "➡️"
    pr_curr, pr_prev = 0.0, 0.0
    
    alerts = []
    recommendations = []
    
    # 1. Z-Score Anomaly Detection (Friction limits)
    # Compare today's delay cost & slippage against the past 60 days distribution
    recent_60_slip = df_60['Exec_Slippage_Mean'].tail(60)
    current_slip = recent_60_slip.iloc[-1]
    slip_mean = recent_60_slip.mean()
    slip_std = recent_60_slip.std()
    
    if slip_std > 0:
        z_slip = (current_slip - slip_mean) / slip_std
        if z_slip < -2.0: # Highly negative slippage
            alerts.append(f"🔴 **执行摩擦崩盘**：过去20个交易日，Execution Slippage 骤降至 {current_slip*100:.2f}% (Z-Score: {z_slip:.2f})。诊断：开盘价到成交价的滑点损失极大超出了60日正常警戒范围！")
            recommendations.append("[参数调整]：建议立即将买卖点打单算法从市价单 (Market/Moo) 切换为限价单 (Limit) 或 TWAP/VWAP 算法，以规避价格打滑极值。")
            
    recent_60_delay = df_60['Delay_Cost_Mean'].tail(60)
    current_delay = recent_60_delay.iloc[-1]
    delay_mean = recent_60_delay.mean()
    delay_std = recent_60_delay.std()
    
    if delay_std > 0:
        z_delay = (current_delay - delay_mean) / delay_std
        if z_delay < -2.0:
            alerts.append(f"🔴 **隔夜跳空恶化**：Delay Cost 跌至 {current_delay*100:.2f}% (Z-Score: {z_delay:.2f})。诊断：T日收盘至T+1日开盘的天然跳空缺口造成巨额隐藏损失，做市环境正在急剧恶化。")
            
    # 2. Moving Average Crossover (Alpha decay)
    # Check if short-term (20d) Idiosyncratic Alpha drops below long-term (60d)
    idio_20 = df_20['Idiosyncratic_Alpha'].tail(5).mean()
    idio_60 = df_60['Idiosyncratic_Alpha'].tail(5).mean()
    
    if idio_20 < idio_60 and idio_20 < 0:
        alerts.append(f"🟡 **Alpha Decay (阿尔法衰减)**：近5日监控点显示，20日滚动的短期特质Alpha ({idio_20*100:.2f}%) 已显著下穿 60日均线 ({idio_60*100:.2f}%) 并跌入负域。诊断：策略的底层选股网络正在迅速失效！")
        recommendations.append("[风控熔断]：短期选股超额收益出现死叉，若未来一周无法修复回归零轴，建议启动防洪机制降低系统整体运作杠杆或强制空仓。")
    elif idio_20 > idio_60 and idio_20 > 0:
        alerts.append(f"🟢 **选股Alpha爆发**：20日特质超额 ({idio_20*100:.2f}%) 强劲向上穿越 60日水准 ({idio_60*100:.2f}%)。诊断：当前分化行情深度契合因子挖掘池，纯粹的选股能力呈现井喷态势！")

    # 3. Historical Percentile Extremes (Factor Drift)
    # Evaluate 1-year historical percentiles for Barra exposures
    past_year_60 = df_60.last("252D").dropna(subset=['Exposure_Liquidity', 'Exposure_Momentum', 'Exposure_Volatility'])
    
    print(f"DEBUG: len(past_year_60)={len(past_year_60)}")
    
    if not past_year_60.empty:
        curr_size = past_year_60['Exposure_Liquidity'].iloc[-1]
        size_5th = past_year_60['Exposure_Liquidity'].quantile(0.05)
        size_95th = past_year_60['Exposure_Liquidity'].quantile(0.95)
        
        if curr_size <= size_5th:
            alerts.append(f"🔴 **极端微盘暴露**：{BARRA_LIQD_KEY} 滚动均值跌至 {{curr_size:.2f}} (突破一年内5%极低分位)。诊断：策略已严重偏离中盘基准，当前正暴露于极高的微盘股流动性风险区！")
            recommendations.append("[因子剥离]：建议在模型融合训练层面加入 Size 因子中性化约束，强行斩断针对流通市值的过度拥挤偏好。")
        elif curr_size >= size_95th:
            alerts.append(f"🔴 **大盘股超载**：{BARRA_LIQD_KEY} 攀升至 {{curr_size:.2f}} (突破一年内95%极高分位)。诊断：风格向权重蓝筹严重漂移。")

        # Win Rate Trend
        past_20_wr = df_20['Win_Rate'].dropna()
        if len(past_20_wr) >= 20: # Ensure we have at least 20 days to look back on 20d data
             curr_wr = past_20_wr.iloc[-1]
             prev_wr = past_20_wr.iloc[-20]
             
             wr_trend = "➡️"
             if curr_wr > prev_wr + 0.02: wr_trend = "📈 上升"
             elif curr_wr < prev_wr - 0.02: wr_trend = "📉 下降"
             
             past_20_pr = df_20['Payoff_Ratio'].dropna()
             if len(past_20_pr) >= 20:
                  pr_curr = past_20_pr.iloc[-1]
                  pr_prev = past_20_pr.iloc[-20]
             
             top_holdings = ""
             if "Holdings_Top1_Concentration" in df_20.columns: # Placeholder logic if holding feature expands
                 top_holdings = df_20['Holdings_Top1_Concentration'].iloc[-1]
    
    # Assemble Status
    health_status = "🟢 正常运转 (无系统性危机)"
    if len([a for a in alerts if '🔴' in a]) >= 2:
         health_status = "🔴 危险 (核心因子漂移或摩擦剧烈重仓积聚，建议停机干预)"
    elif len(alerts) > 0:
         health_status = "🟡 警告 (出现异动或轻微摩擦侵蚀，需关注监控)"

    # Format Markdown string
    report_md = f"""# 📊 策略滚动体检报告 (Rolling Health Summary)

**日期标签：** {last_date.strftime('%Y-%m-%d')} | **对比基期：** 过去20个交易日 (T-20) 
**总体健康度：** {health_status}

### 1. 🚨 核心异动警告 (Anomalies & Alerts)
"""
    if alerts:
        for alert in alerts:
            report_md += f"- {alert}\n"
    else:
        report_md += "- 暂未检测到显著的风格漂移、阿尔法衰减或统计级摩擦异常。\n"

    report_md += f"""
### 2. 📈 性能趋势 (Performance Trends: 20-Day Rolling)
- **Win Rate (日历胜率):** {prev_wr*100:.2f}% ➡️ {curr_wr*100:.2f}% ({wr_trend})
- **Payoff Ratio (盈亏比):** {pr_prev:.2f} ➡️ {pr_curr:.2f}
"""
    
    report_md += """
### 3. 🤖 系统自动化操作建议 (Actionable Recommendations)
"""
    if recommendations:
        # Deduplicate
        unique_recs = list(set(recommendations))
        for rec in unique_recs:
             report_md += f"- {rec}\n"
    else:
        report_md += "- 当前环境良好，建议保持原有流水线惯性运转。\n"

    # Write output
    out_file = os.path.join(output_dir, "rolling_health_report.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"\n✅ Rolling Health Summary correctly generated at: {out_file}")
    print("="*60)
    print(report_md)
    print("="*60)
    
if __name__ == "__main__":
    evaluate_health()
