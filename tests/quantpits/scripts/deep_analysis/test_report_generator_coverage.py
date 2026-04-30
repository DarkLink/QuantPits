"""
Supplemental tests targeting uncovered branches in report_generator.py.

Coverage targets:
- _section_market: period_return/window_vol None (82,84), regime_switches (90-94)
- _section_model_health: empty best_af early return (110), convergence summary (166-176),
  >5 models retrain (146)
- _section_ensemble: empty series skip (206), multi-date period_str (222),
  trend arrow delta (228-234), no-change events (251), OOS trend (256-270)
- _section_execution: fill rate + latency (310-317)
- _section_prediction: per-model IC proxy + underperformers (399-407)
- _section_change_impact: non-retrain events (453-459)
"""

import pytest
from quantpits.scripts.deep_analysis.report_generator import ReportGenerator
from quantpits.scripts.deep_analysis.base_agent import AgentFindings, Finding


def test_market_regime_no_period_return_or_vol():
    """Lines 82, 84: period_return and window_vol are None → not rendered."""
    af = AgentFindings(
        agent_name="Market Regime",
        window_label="W1",
        findings=[],
        raw_metrics={
            "regime": "Bull",
            "period_return": None,
            "window_volatility": None,
            "max_drawdown": 0.05,
            "current_drawdown": 0.02,
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "Period Return" not in report
    assert "Annualized Volatility" not in report
    assert "Bull" in report


def test_market_regime_with_switches():
    """Lines 90-94: regime_switches with switch_count > 0."""
    af = AgentFindings(
        agent_name="Market Regime",
        window_label="W1",
        findings=[],
        raw_metrics={
            "regime": "Neutral",
            "period_return": 0.05,
            "window_volatility": 0.15,
            "max_drawdown": 0.03,
            "current_drawdown": 0.01,
            "regime_switches": {
                "switch_count": 2,
                "current_regime": "Neutral",
                "current_streak_days": 5,
                "switches": [
                    {"approx_date": "2026-01-15", "from": "Bear", "to": "Bull"},
                    {"approx_date": "2026-02-20", "from": "Bull", "to": "Neutral"},
                ],
            },
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "Regime Switches:" in report
    assert "2 detected" in report
    assert "2026-01-15" in report
    assert "Bear → Bull" in report


def test_model_health_no_best_af():
    """Line 110: findings list exists but best_af is None (all scorecards empty)."""
    af = AgentFindings(
        agent_name="Model Health",
        window_label="W1",
        findings=[],
        raw_metrics={},  # no scorecard
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "2. Model Health Dashboard" in report
    assert "2.1 IC/ICIR Scorecard" not in report


def test_model_health_convergence_summary():
    """Lines 166-176: convergence summary with underfitting and full-epoch models."""
    af = AgentFindings(
        agent_name="Model Health",
        window_label="W1",
        findings=[],
        raw_metrics={
            "scorecard": {
                "m1": {"ic_mean": 0.05, "ic_latest": 0.04, "icir_mean": 0.5, "ic_trend": "degrading", "n_snapshots": 10},
            },
            "convergence_summary": {
                "total_models": 5,
                "pct_early_stopped": 0.6,
                "avg_duration_s": 180,
                "underfitting_candidates": ["m3", "m5"],
                "full_epoch_models": ["m1", "m2"],
            },
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "2.4 Convergence Summary" in report
    assert "Models tracked:** 5" in report
    assert "Early-stopped:** 60%" in report
    assert "Underfitting candidates:** m3, m5" in report
    assert "Full-epoch models:** m1, m2" in report


def test_model_health_retrain_many_models():
    """Line 146: >5 models retrained on same date → truncated display."""
    af = AgentFindings(
        agent_name="Model Health",
        window_label="W1",
        findings=[],
        raw_metrics={
            "scorecard": {},
            "retrain_events": [
                {"date": "2026-03-01", "model": f"m{i}"} for i in range(1, 8)
            ],
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "7 models retrained" in report
    assert "m1, m2, m3" in report


def test_model_health_convergence_no_underfitting():
    """Lines 170-176: convergence present but no underfitting/full-epoch lists."""
    af = AgentFindings(
        agent_name="Model Health",
        window_label="W1",
        findings=[],
        raw_metrics={
            "scorecard": {},
            "convergence_summary": {
                "total_models": 3,
                "pct_early_stopped": 0.33,
                "avg_duration_s": 120,
            },
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "2.4 Convergence Summary" in report
    # No underfitting/full-epoch text since keys are missing
    assert "Underfitting candidates" not in report
    assert "Full-epoch models" not in report


def test_ensemble_combo_trends_empty_series():
    """Line 206: empty series in combo_trends → skip."""
    af = AgentFindings(
        agent_name="Ensemble Evolution",
        window_label="W1",
        findings=[],
        raw_metrics={
            "combo_trends": {
                "combo1": [],  # empty series
                "combo2": [
                    {"_date": "2026-03-01", "total_return": 5.0, "calmar_ratio": 2.0, "excess_return": 1.0},
                ],
            },
            "change_events": [],
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "combo2" in report
    # combo1 should be skipped (empty series)
    lines = [l for l in report.split("\n") if "combo1" in l]
    assert len(lines) == 0


def test_ensemble_combo_trends_multi_date():
    """Line 222: first_date != latest_date → date range display."""
    af = AgentFindings(
        agent_name="Ensemble Evolution",
        window_label="W1",
        findings=[],
        raw_metrics={
            "combo_trends": {
                "combo1": [
                    {"_date": "2026-01-01", "total_return": 2.0, "calmar_ratio": 1.5, "excess_return": 0.5},
                    {"_date": "2026-03-01", "total_return": 8.0, "calmar_ratio": 2.5, "excess_return": 1.5},
                ],
            },
            "change_events": [],
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "2026-01-01→2026-03-01" in report


def test_ensemble_trend_arrow_improving():
    """Lines 228-230: delta > 1 → improving arrow."""
    af = AgentFindings(
        agent_name="Ensemble Evolution",
        window_label="W1",
        findings=[],
        raw_metrics={
            "combo_trends": {
                "combo1": [
                    {"_date": "2026-01-01", "total_return": 2.0, "calmar_ratio": 1.5, "excess_return": 0.5},
                    {"_date": "2026-03-01", "total_return": 8.0, "calmar_ratio": 2.5, "excess_return": 1.5},
                ],
            },
            "change_events": [],
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "📈" in report


def test_ensemble_trend_arrow_degrading():
    """Lines 231-232: delta < -1 → degrading arrow."""
    af = AgentFindings(
        agent_name="Ensemble Evolution",
        window_label="W1",
        findings=[],
        raw_metrics={
            "combo_trends": {
                "combo1": [
                    {"_date": "2026-01-01", "total_return": 8.0, "calmar_ratio": 2.5, "excess_return": 1.5},
                    {"_date": "2026-03-01", "total_return": 2.0, "calmar_ratio": 1.5, "excess_return": 0.5},
                ],
            },
            "change_events": [],
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "📉" in report


def test_ensemble_trend_flat():
    """Line 233-234: delta between -1 and 1 → flat."""
    af = AgentFindings(
        agent_name="Ensemble Evolution",
        window_label="W1",
        findings=[],
        raw_metrics={
            "combo_trends": {
                "combo1": [
                    {"_date": "2026-01-01", "total_return": 5.0, "calmar_ratio": 2.0, "excess_return": 1.0},
                    {"_date": "2026-03-01", "total_return": 5.5, "calmar_ratio": 2.1, "excess_return": 1.1},
                ],
            },
            "change_events": [],
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "flat" in report


def test_ensemble_no_change_events():
    """Line 251: empty change_events → no-changes message."""
    af = AgentFindings(
        agent_name="Ensemble Evolution",
        window_label="W1",
        findings=[],
        raw_metrics={
            "change_events": [],
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "No ensemble composition changes detected" in report


def test_ensemble_oos_trend():
    """Lines 256-270: OOS trend with calmar slope, latest/best, and decay."""
    af = AgentFindings(
        agent_name="Ensemble Evolution",
        window_label="W1",
        findings=[],
        raw_metrics={
            "change_events": [],
            "oos_trend": {
                "oos_runs": 5,
                "oos_calmar_slope": 0.15,
                "latest_oos_calmar": 2.1,
                "best_oos_calmar": 2.5,
                "is_oos_decay": [
                    {"combo": "combo1", "decay_ratio": 0.3, "full_calmar": 3.0, "oos_calmar": 2.1},
                ],
            },
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "3.3 OOS Performance Trend" in report
    assert "OOS runs analyzed:** 5" in report
    assert "improving" in report  # slope > 0
    assert "Latest OOS Calmar:** 2.10" in report
    assert "Best:** 2.50" in report
    assert "IS→OOS Decay" in report
    assert "30.0%" in report


def test_ensemble_oos_trend_degrading():
    """Line 260: slope < 0 → 'degrading'."""
    af = AgentFindings(
        agent_name="Ensemble Evolution",
        window_label="W1",
        findings=[],
        raw_metrics={
            "change_events": [],
            "oos_trend": {
                "oos_runs": 3,
                "oos_calmar_slope": -0.08,
            },
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "degrading" in report


def test_execution_fill_rate_and_latency():
    """Lines 310-317: fill_rate not None → render timing + latency metrics."""
    af = AgentFindings(
        agent_name="Execution Quality",
        window_label="W1",
        findings=[],
        raw_metrics={
            "fill_rate": 0.85,
            "cancel_rate": 0.15,
            "latency_mean_sec": 2.5,
            "latency_median_sec": 2.0,
            "latency_p90_sec": 5.0,
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "Fill Rate 85.0%" in report
    assert "Cancel Rate 15.0%" in report
    assert "Mean 2.50s" in report
    assert "Median 2.00s" in report
    assert "P90 5.00s" in report


def test_execution_fill_rate_no_latency():
    """Line 314: lat_mean is None → skip latency details."""
    af = AgentFindings(
        agent_name="Execution Quality",
        window_label="W1",
        findings=[],
        raw_metrics={
            "fill_rate": 0.90,
            "cancel_rate": 0.10,
            "latency_mean_sec": None,
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "Fill Rate 90.0%" in report
    assert "Latency" not in report


def test_prediction_per_model_ic_proxy():
    """Lines 399-407: per_model_hit_rate with IC proxy and underperformers."""
    af = AgentFindings(
        agent_name="Prediction Audit",
        window_label="W1",
        findings=[],
        raw_metrics={
            "per_model_hit_rate": {
                "per_model_ic": {"m1": 0.15, "m2": 0.08, "m3": -0.02},
                "ensemble_overall_proxy_ic": 0.12,
                "underperformers": ["m3"],
            },
        },
    )
    gen = ReportGenerator([af], {}, "")
    report = gen.generate()
    assert "6.1 Per-Model IC Proxy" in report
    assert "0.1200" in report  # ensemble_overall_proxy_ic
    assert "m1: 0.1500" in report
    assert "m3: -0.0200" in report
    assert "⚠️" in report
    assert "Underperformers:** m3" in report


def test_change_impact_non_retrain():
    """Lines 453-459: non-retrain events displayed."""
    gen = ReportGenerator(
        all_findings=[],
        synthesis_result={
            "change_impact": [
                {"event": {"type": "feature_change", "date": "2026-02-01", "detail": "Added Alpha158_v2"}},
                {"event": {"type": "retrain", "date": "2026-02-15", "model": "m1"}},
                {"event": {"type": "dataset_change", "date": "2026-02-20", "detail": "Extended history"}},
            ],
        },
        executive_summary="",
    )
    report = gen.generate()
    assert "8. Holistic Change Impact Assessment" in report
    assert "Non-Retrain Changes" in report
    assert "feature_change" in report
    assert "Alpha158_v2" in report
    assert "dataset_change" in report
    assert "Extended history" in report
    assert "Section 2.2" in report  # retrain reference


def test_change_impact_all_retrains():
    """Lines 450-451: change_impact has only retrains → no Non-Retrain section."""
    gen = ReportGenerator(
        all_findings=[],
        synthesis_result={
            "change_impact": [
                {"event": {"type": "retrain", "date": "2026-02-15", "model": "m1"}},
            ],
        },
        executive_summary="",
    )
    report = gen.generate()
    assert "8. Holistic Change Impact Assessment" in report
    assert "Non-Retrain Changes" not in report
    assert "Section 2.2" in report


def test_section_appendix_with_notes():
    """Line 489: external_notes present → appendix section."""
    gen = ReportGenerator(
        all_findings=[],
        synthesis_result={"external_notes": "These are some external notes."},
        executive_summary="",
    )
    report = gen.generate()
    assert "Appendix: External Notes" in report
    assert "These are some external notes." in report


def test_section_appendix_empty_notes():
    """Line 489: external_notes empty string → no appendix."""
    gen = ReportGenerator(
        all_findings=[],
        synthesis_result={"external_notes": ""},
        executive_summary="",
    )
    report = gen.generate()
    assert "Appendix" not in report
