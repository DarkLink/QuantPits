"""
Training Health Agent (Phase 2).

Evaluates the health, scheduling, progress, and performance trends of training pipelines:
1. Mode Coverage: Audits which models have static, cpcv, rolling, and cpcv_rolling runs.
2. Rolling Progress & Staleness: Inspects rolling states for completion and anchor_date gaps.
3. Alphas and Frictions: Absorbs rules from run_rolling_health_report.py to monitor alpha decay,
   execution slippage/delay cost anomalies, and factor drift (Barra exposures) using rolling metrics CSVs.
"""

import os
import logging
import pandas as pd

from ..base_agent import BaseAgent, AgentFindings, AnalysisContext

logger = logging.getLogger(__name__)


class TrainingHealthAgent(BaseAgent):
    name = "Training Health"
    description = "Evaluates model training modes, rolling window progress, and performance decay metrics."

    def analyze(self, ctx: AnalysisContext) -> AgentFindings:
        findings = []
        recommendations = []
        raw_metrics = {}

        tc = ctx.training_context
        if not tc:
            findings.append(self._make_finding(
                'info', 'No training context available',
                'TrainingModeContext is missing. Skipping training health analysis.'
            ))
            return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)

        # --- 1. Mode Coverage Audit ---
        coverage_stats = {}
        for name, modes_dict in tc.models_by_name.items():
            coverage_stats[name] = list(modes_dict.keys())
        
        raw_metrics['mode_coverage'] = coverage_stats

        # Flag models that are missing expected modes
        expected_modes = []
        if tc.rolling_states:
            expected_modes.append('rolling')
        if tc.cpcv_config:
            expected_modes.append('cpcv')
        
        # Check active models mode coverage
        for model_name, modes in coverage_stats.items():
            missing = [m for m in expected_modes if m not in modes]
            if missing:
                findings.append(self._make_finding(
                    'info', f'{model_name}: Incomplete training modes',
                    f"Model is trained under {modes} but missing expected modes: {missing}.",
                    {'model': model_name, 'missing': missing, 'present': modes}
                ))

        # --- 2. Rolling Progress & Staleness Audit ---
        rolling_progress = {}
        for mode, state in tc.rolling_states.items():
            total = state.get("total_windows", 0)
            curr = state.get("current_window_idx", 0)
            anchor = state.get("anchor_date", "Unknown")
            gap = tc.get_rolling_gap_days(mode)

            rolling_progress[mode] = {
                'total_windows': total,
                'current_window_idx': curr,
                'anchor_date': anchor,
                'gap_days': gap
            }

            # Rule: Staleness warning if last rolling anchor is too far behind current date
            if gap is not None and gap > 90:
                findings.append(self._make_finding(
                    'warning', f'Rolling staleness: {mode}',
                    f"Rolling pipeline for '{mode}' has not been updated in {gap} days (last anchor: {anchor}). "
                    f"Recommend triggering a rolling update run.",
                    {'mode': mode, 'gap_days': gap, 'last_anchor': anchor}
                ))
                recommendations.append(f"[训练推进] 检测到 '{mode}' 滚动窗口存在严重延迟（T-{gap} 天），建议立即手动运行滚动增量训练。")

            # Check if rolling finished
            if total > 0 and curr == total:
                findings.append(self._make_finding(
                    'positive', f'Rolling pipeline complete: {mode}',
                    f"All {total} windows for rolling mode '{mode}' have completed training.",
                    {'mode': mode, 'total_windows': total}
                ))

        raw_metrics['rolling_progress'] = rolling_progress

        # --- 3. Alpha, Friction, and Factor Drift Monitoring ---
        # Look for rolling_metrics_20.csv and rolling_metrics_60.csv in output dir
        output_dir = os.path.join(ctx.workspace_root, "output")
        file_20 = os.path.join(output_dir, "rolling_metrics_20.csv")
        file_60 = os.path.join(output_dir, "rolling_metrics_60.csv")

        if os.path.exists(file_20) and os.path.exists(file_60):
            try:
                df_20 = pd.read_csv(file_20, parse_dates=['Date']).set_index('Date').sort_index()
                df_60 = pd.read_csv(file_60, parse_dates=['Date']).set_index('Date').sort_index()

                # A. Slippage z-score
                if 'Exec_Slippage_Mean' in df_60.columns and len(df_60) >= 60:
                    try:
                        recent_60_slip = df_60['Exec_Slippage_Mean'].tail(60)
                        current_slip = recent_60_slip.iloc[-1]
                        slip_mean = recent_60_slip.mean()
                        slip_std = recent_60_slip.std()

                        if slip_std > 0:
                            z_slip = (current_slip - slip_mean) / slip_std
                            raw_metrics['slippage_zscore'] = z_slip
                            if z_slip < -2.0:
                                findings.append(self._make_finding(
                                    'critical', '执行摩擦崩盘 (High Slippage Extreme)',
                                    f"执行滑点 Exec_Slippage_Mean ({current_slip*100:.2f}%) 显著突破60日警示范围 (Z-Score: {z_slip:.2f})。",
                                    {'current_slippage': current_slip, 'z_score': z_slip}
                                ))
                                recommendations.append("[参数调整] 建议切换下单算法为 Limit (限价) 或 TWAP/VWAP 算法，以减缓价格冲击。")
                    except Exception as slip_err:
                        logger.warning(f"Slippage check warning: {slip_err}")

                # B. Delay cost z-score
                if 'Delay_Cost_Mean' in df_60.columns and len(df_60) >= 60:
                    try:
                        recent_60_delay = df_60['Delay_Cost_Mean'].tail(60)
                        current_delay = recent_60_delay.iloc[-1]
                        delay_mean = recent_60_delay.mean()
                        delay_std = recent_60_delay.std()

                        if delay_std > 0:
                            z_delay = (current_delay - delay_mean) / delay_std
                            raw_metrics['delay_cost_zscore'] = z_delay
                            if z_delay < -2.0:
                                findings.append(self._make_finding(
                                    'warning', '隔夜跳空恶化 (Delay Cost Degradation)',
                                    f"隔夜成本 Delay_Cost_Mean 恶化至 {current_delay*100:.2f}% (Z-Score: {z_delay:.2f})。",
                                    {'current_delay': current_delay, 'z_score': z_delay}
                                ))
                    except Exception as delay_err:
                        logger.warning(f"Delay cost check warning: {delay_err}")

                # C. Alpha decay
                if 'Idiosyncratic_Alpha' in df_20.columns and 'Idiosyncratic_Alpha' in df_60.columns:
                    try:
                        idio_20 = df_20['Idiosyncratic_Alpha'].tail(5).mean()
                        idio_60 = df_60['Idiosyncratic_Alpha'].tail(5).mean()
                        raw_metrics['idiosyncratic_alpha_20d'] = idio_20
                        raw_metrics['idiosyncratic_alpha_60d'] = idio_60

                        if idio_20 < idio_60 and idio_20 < 0:
                            findings.append(self._make_finding(
                                'warning', 'Alpha Decay (选股能力衰减)',
                                f"短期(20d) 特质选股Alpha ({idio_20*100:.2f}%) 下穿长期(60d) 水平并落入负值区间。",
                                {'alpha_20d': idio_20, 'alpha_60d': idio_60}
                            ))
                            recommendations.append("[风控降杠杆] 选股阿尔法下穿，建议启动风控机制，适当缩减组合整体仓位或降低运作杠杆。")
                        elif idio_20 > idio_60 and idio_20 > 0:
                            findings.append(self._make_finding(
                                'positive', '选股 Alpha 强劲爆发',
                                f"短期(20d) 特质选股Alpha ({idio_20*100:.2f}%) 强劲向上跨越长期均线，体现优异选股能力。",
                                {'alpha_20d': idio_20}
                            ))
                    except Exception as alpha_err:
                        logger.warning(f"Alpha decay check warning: {alpha_err}")

                # D. Factor drift (Barra exposures percentile)
                if 'Exposure_Liquidity' in df_60.columns:
                    try:
                        past_year_60 = df_60.last("252D").dropna(subset=['Exposure_Liquidity'])
                        if not past_year_60.empty:
                            curr_size = past_year_60['Exposure_Liquidity'].iloc[-1]
                            size_5th = past_year_60['Exposure_Liquidity'].quantile(0.05)
                            size_95th = past_year_60['Exposure_Liquidity'].quantile(0.95)

                            raw_metrics['barra_liquidity_exposure'] = curr_size
                            if curr_size <= size_5th:
                                findings.append(self._make_finding(
                                    'critical', '极端微盘漂移 (Size Drift Extreme)',
                                    f"风格暴露倾向微盘股，Barra Liquidity Exposure 均值跌至 {curr_size:.2f} (处于近一年5%极低水平)。",
                                    {'current_exposure': curr_size, 'quantile_5%': size_5th}
                                ))
                                recommendations.append("[因子剥离] 建议在组合模型融合中增加 Size (市值) 风格中性化因子约束，隔离小市值流动性危机。")
                            elif curr_size >= size_95th:
                                findings.append(self._make_finding(
                                    'warning', '大盘蓝筹超载 (Large Cap Shift)',
                                    f"风格风格朝大盘蓝筹漂移，Exposure_Liquidity 上升至 {curr_size:.2f} (超一年内95%水平)。",
                                    {'current_exposure': curr_size, 'quantile_95%': size_95th}
                                ))
                    except Exception as drift_err:
                        logger.warning(f"Factor drift check warning: {drift_err}")

            except Exception as e:
                logger.error(f"Failed to analyze rolling metrics CSVs in TrainingHealthAgent: {e}")
                findings.append(self._make_finding(
                    'info', 'Failed to load rolling metrics CSVs',
                    f"Error encountered while parsing rolling metrics: {e}"
                ))

        return AgentFindings(self.name, ctx.window_label, findings, recommendations, raw_metrics)
