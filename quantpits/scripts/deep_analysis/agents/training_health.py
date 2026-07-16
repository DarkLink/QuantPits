"""
Training Health Agent (Phase 2).

Evaluates the health, scheduling, progress, and performance trends of training pipelines:
1. Mode Coverage: Audits which models have static, cpcv, rolling, and cpcv_rolling runs.
2. Rolling Progress & Staleness: Inspects rolling states for completion and anchor_date gaps.
3. Alphas and Frictions: Absorbs rules from run_rolling_health_report.py to monitor alpha decay,
   execution slippage/delay cost anomalies, and factor drift (Barra exposures) using rolling metrics CSVs.
"""

import os
import json
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

        state_diagnostics = {}
        for mode, inspection in tc.rolling_state_inspections.items():
            classification = inspection.classification
            state_diagnostics[mode] = {
                "classification": classification,
                "reason_code": inspection.reason_code,
                "fingerprint": inspection.fingerprint,
            }
            if classification in ("missing", "valid_legacy"):
                continue
            severity = (
                "info" if classification == "valid_versioned" else "warning"
            )
            findings.append(self._make_finding(
                severity,
                f"Rolling state classification: {mode}",
                f"Rolling state for '{mode}' is {classification} "
                f"({inspection.reason_code}); legacy progress analysis is blocked.",
                {
                    "mode": mode,
                    "classification": classification,
                    "reason_code": inspection.reason_code,
                    "consumption": inspection.consumption,
                },
            ))
        raw_metrics["rolling_state_inspections"] = state_diagnostics

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
        
        # Check active models mode coverage (suppress in predict-only cycles
        # where modes aren't missing — they just weren't run this cycle)
        for model_name, modes in coverage_stats.items():
            missing = [m for m in expected_modes if m not in modes]
            if missing:
                # In predict-only cycles, only flag models that have NEVER been
                # trained in the expected mode (per training_history.jsonl).
                if tc.is_predict_only_cycle and tc.last_train:
                    last_train_mode = tc.last_train.get(model_name, {}).get('mode', '')
                    if last_train_mode in expected_modes:
                        continue  # Model has training history, just not this cycle
                findings.append(self._make_finding(
                    'info', f'{model_name}: Incomplete training modes',
                    f"Model is trained under {modes} but missing expected modes: {missing}.",
                    {'model': model_name, 'missing': missing, 'present': modes}
                ))

        # --- 2. Orphan Model Detection ---
        orphan_findings, orphan_recs = self._detect_orphan_models(ctx)
        findings.extend(orphan_findings)
        recommendations.extend(orphan_recs)

        # --- 3. Rolling Progress & Staleness Audit ---
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

        # --- 4. Alpha, Friction, and Factor Drift Monitoring ---
        # Look for rolling_metrics_20.csv and rolling_metrics_60.csv in output dir
        output_dir = os.path.join(ctx.workspace_root, "output")
        file_20 = os.path.join(output_dir, "rolling_metrics_20.csv")
        file_60 = os.path.join(output_dir, "rolling_metrics_60.csv")

        # Report missing rolling metrics files instead of silently skipping
        missing_metrics = []
        if not os.path.exists(file_20):
            missing_metrics.append("rolling_metrics_20.csv")
        if not os.path.exists(file_60):
            missing_metrics.append("rolling_metrics_60.csv")

        if missing_metrics:
            findings.append(self._make_finding(
                'info' if len(missing_metrics) == 1 else 'warning',
                'Rolling metrics data unavailable',
                f"Missing CSV: {', '.join(missing_metrics)}. "
                f"Alpha decay, slippage, and factor drift checks are skipped. "
                f"These files are generated by the rolling analysis pipeline. "
                f"If rolling training is not configured, this is expected.",
                {"missing_files": missing_metrics, "expected_dir": output_dir}
            ))

        if os.path.exists(file_20) or os.path.exists(file_60):
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

    def _detect_orphan_models(self, ctx: AnalysisContext) -> tuple:
        """Detect enabled models that are not in any active combo.

        An orphan model is one that:
        - Is enabled in model_registry.yaml
        - Is NOT listed in any combo in ensemble_config.json
        - May be stale (no recent training record)

        Returns (findings, recommendations).
        """
        findings = []
        recommendations = []

        # 1. Read model_registry.yaml for enabled models
        registry_path = os.path.join(
            ctx.workspace_root, "config", "model_registry.yaml"
        )
        if not os.path.exists(registry_path):
            return findings, recommendations

        try:
            import yaml
            with open(registry_path, "r") as f:
                registry = yaml.safe_load(f) or {}
        except Exception:
            return findings, recommendations

        enabled_models = set()
        for name, spec in registry.get("models", {}).items():
            if spec.get("enabled", False):
                enabled_models.add(name)

        if not enabled_models:
            return findings, recommendations

        # 2. Read ensemble_config.json for models in any combo
        ec_path = os.path.join(
            ctx.workspace_root, "config", "ensemble_config.json"
        )
        combo_models = set()
        if os.path.exists(ec_path):
            try:
                with open(ec_path, "r") as f:
                    ec = json.load(f)
                for combo_def in ec.get("combos", ec.get("combo_groups", {})).values():
                    for m in combo_def.get("models", []):
                        # Strip @suffix if present (e.g., lstm@static -> lstm)
                        base_name = m.split("@")[0]
                        combo_models.add(base_name)
            except Exception:
                pass

        # 3. Find orphans: enabled but not in any combo
        for model_name in sorted(enabled_models):
            if model_name not in combo_models:
                # Check staleness via training_context
                is_stale = False
                stale_detail = ""
                tc = ctx.training_context
                if tc and hasattr(tc, "models_by_name"):
                    model_modes = tc.models_by_name.get(model_name, {})
                    if not model_modes:
                        is_stale = True
                        stale_detail = (
                            f" Model '{model_name}' has no recent training "
                            f"records in latest_train_records.json."
                        )

                severity = "warning" if is_stale else "info"
                findings.append(self._make_finding(
                    severity,
                    f"{model_name}: Orphan model detected (孤立模型)",
                    f"Model '{model_name}' is enabled in model_registry.yaml "
                    f"but is not in any active combo. This model produces "
                    f"predictions that are never used in ensemble fusion."
                    + stale_detail,
                    {"model": model_name, "in_any_combo": False,
                     "is_stale": is_stale}
                ))

                if is_stale:
                    recommendations.append(
                        f"[模型清理] '{model_name}' 是孤立且陈旧的模型，"
                        f"建议禁用 (enabled: false) 或将其加入活跃组合。"
                    )
                else:
                    recommendations.append(
                        f"[模型审查] '{model_name}' 不在任何活跃组合中，"
                        f"建议确认是否需要保留此模型。"
                    )

        return findings, recommendations
