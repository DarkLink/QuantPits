"""Tests for Signal Extractor (Phase 3)."""

import pytest
from quantpits.scripts.deep_analysis.base_agent import AgentFindings
from quantpits.scripts.deep_analysis.signal_extractor import Signal, SignalExtractor


def _make_findings(agent_name, raw_metrics, window="full"):
    return AgentFindings(
        agent_name=agent_name,
        window_label=window,
        findings=[],
        recommendations=[],
        raw_metrics=raw_metrics,
    )


# ------------------------------------------------------------------
# Rule 1: underfitting
# ------------------------------------------------------------------

class TestUnderfitting:
    def test_underfitting_candidates_produces_signal(self):
        af = _make_findings("Model Health", {
            "convergence_summary": {
                "underfitting_candidates": ["gru_Alpha158"],
                "model_details": {
                    "gru_Alpha158": {
                        "actual_epochs": 30,
                        "configured_epochs": 200,
                        "early_stopped": True,
                    }
                },
                "full_epoch_models": [],
            },
            "scorecard": {},
            "stale_models": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})

        matching = [s for s in signals if s.target == "gru_Alpha158"]
        assert len(matching) >= 1
        s = matching[0]
        # Should be upgraded to severe_underfitting since 30 < 200*0.25=50
        assert s.signal_type == "severe_underfitting"
        assert s.scope == "hyperparams"
        assert s.severity == "warning"

    def test_empty_underfitting_candidates(self):
        af = _make_findings("Model Health", {
            "convergence_summary": {
                "underfitting_candidates": [],
                "model_details": {},
                "full_epoch_models": [],
            },
            "scorecard": {},
            "stale_models": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        assert not any(s.signal_type == "underfitting" for s in signals)


# ------------------------------------------------------------------
# Rule 2: severe_underfitting
# ------------------------------------------------------------------

class TestSevereUnderfitting:
    def test_severe_underfitting_standalone(self):
        """Model not in underfitting_candidates but epochs ratio < 0.25."""
        af = _make_findings("Model Health", {
            "convergence_summary": {
                "underfitting_candidates": [],
                "model_details": {
                    "alstm_Alpha158": {
                        "actual_epochs": 10,
                        "configured_epochs": 200,
                    }
                },
                "full_epoch_models": [],
            },
            "scorecard": {},
            "stale_models": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        severe = [s for s in signals if s.signal_type == "severe_underfitting"]
        assert len(severe) == 1
        assert severe[0].target == "alstm_Alpha158"
        assert severe[0].metrics["ratio"] == 0.05

    def test_not_severe_when_ratio_above_threshold(self):
        af = _make_findings("Model Health", {
            "convergence_summary": {
                "underfitting_candidates": [],
                "model_details": {
                    "lstm_Alpha158": {
                        "actual_epochs": 60,
                        "configured_epochs": 200,
                    }
                },
                "full_epoch_models": [],
            },
            "scorecard": {},
            "stale_models": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        assert not any(s.signal_type == "severe_underfitting" for s in signals)


# ------------------------------------------------------------------
# Rule 3: overfitting
# ------------------------------------------------------------------

class TestOverfitting:
    def test_full_epoch_low_ic(self):
        af = _make_findings("Model Health", {
            "convergence_summary": {
                "underfitting_candidates": [],
                "model_details": {},
                "full_epoch_models": ["catboost_Alpha158"],
            },
            "scorecard": {
                "catboost_Alpha158": {"ic_mean": 0.01, "ic_trend": "stable"},
            },
            "stale_models": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        ovf = [s for s in signals if s.signal_type == "overfitting"]
        assert len(ovf) == 1
        assert ovf[0].target == "catboost_Alpha158"

    def test_full_epoch_good_ic_no_signal(self):
        af = _make_findings("Model Health", {
            "convergence_summary": {
                "underfitting_candidates": [],
                "model_details": {},
                "full_epoch_models": ["catboost_Alpha158"],
            },
            "scorecard": {
                "catboost_Alpha158": {"ic_mean": 0.05, "ic_trend": "stable"},
            },
            "stale_models": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        assert not any(s.signal_type == "overfitting" for s in signals)


# ------------------------------------------------------------------
# Rule 4: ic_decay
# ------------------------------------------------------------------

class TestICDecay:
    def test_degrading_ic_trend(self):
        af = _make_findings("Model Health", {
            "convergence_summary": {
                "underfitting_candidates": [],
                "model_details": {},
                "full_epoch_models": [],
            },
            "scorecard": {
                "gru_Alpha158": {"ic_trend": "degrading", "ic_mean": 0.04},
                "lstm_Alpha158": {"ic_trend": "stable", "ic_mean": 0.05},
            },
            "stale_models": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        decay = [s for s in signals if s.signal_type == "ic_decay"]
        assert len(decay) == 1
        assert decay[0].target == "gru_Alpha158"


# ------------------------------------------------------------------
# Rule 5: model_stale
# ------------------------------------------------------------------

class TestModelStale:
    def test_stale_model_recommend_retrain(self):
        af = _make_findings("Model Health", {
            "convergence_summary": {
                "underfitting_candidates": [],
                "model_details": {},
                "full_epoch_models": [],
            },
            "scorecard": {},
            "stale_models": {
                "gru_Alpha158": {
                    "recommend_retrain": True,
                    "last_retrain": "2025-12-01",
                    "ic_trend": "degrading",
                },
                "lstm_Alpha158": {
                    "recommend_retrain": False,
                    "last_retrain": "2026-03-01",
                    "ic_trend": "stable",
                },
            },
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        stale = [s for s in signals if s.signal_type == "model_stale"]
        assert len(stale) == 1
        assert stale[0].target == "gru_Alpha158"
        assert stale[0].severity == "info"


# ------------------------------------------------------------------
# Rule 6/7: oos_degradation / oos_degradation_limited_sample
# ------------------------------------------------------------------

class TestOOSDegradation:
    def test_oos_degradation_sufficient_samples(self):
        af = _make_findings("Ensemble Evolution", {
            "oos_trend": {
                "oos_calmar_slope": -0.5,
                "oos_runs": 8,
                "latest_oos_calmar": 1.2,
                "best_oos_calmar": 2.5,
            },
            "model_contributions": {},
            "combo_trends": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        oos = [s for s in signals if s.signal_type == "oos_degradation"]
        assert len(oos) == 1
        assert oos[0].severity == "warning"

    def test_oos_degradation_limited_sample(self):
        af = _make_findings("Ensemble Evolution", {
            "oos_trend": {
                "oos_calmar_slope": -0.5,
                "oos_runs": 3,
            },
            "model_contributions": {},
            "combo_trends": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        limited = [s for s in signals if s.signal_type == "oos_degradation_limited_sample"]
        assert len(limited) == 1
        assert limited[0].severity == "info"

    def test_oos_no_signal_when_slope_positive(self):
        af = _make_findings("Ensemble Evolution", {
            "oos_trend": {
                "oos_calmar_slope": 0.1,
                "oos_runs": 10,
            },
            "model_contributions": {},
            "combo_trends": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        assert not any("oos" in s.signal_type for s in signals)


# ------------------------------------------------------------------
# Rule 8: negative_contribution
# ------------------------------------------------------------------

class TestNegativeContribution:
    def test_consistently_negative(self):
        af = _make_findings("Ensemble Evolution", {
            "oos_trend": {},
            "model_contributions": {
                "consistently_negative": ["gats_Alpha158_origin_N"],
                "loo_deltas": {
                    "gats_Alpha158_origin_N": {"mean": -0.05, "count": 5},
                },
                "model_excess": {
                    "gats_Alpha158_origin_N": {"mean": -0.03},
                },
            },
            "combo_trends": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        neg = [s for s in signals if s.signal_type == "negative_contribution"]
        assert len(neg) == 1
        assert neg[0].scope == "model_selection"
        assert neg[0].target == "gats_Alpha158_origin_N"


# ------------------------------------------------------------------
# Rule 9: poor_predictor
# ------------------------------------------------------------------

class TestPoorPredictor:
    def test_underperformers(self):
        af = _make_findings("Prediction Audit", {
            "per_model_hit_rate": {
                "underperformers": ["gats_Alpha158_origin_N"],
                "per_model_ic": {
                    "gats_Alpha158_origin_N": 0.12,
                    "gru_Alpha158": 0.45,
                },
                "ensemble_overall_proxy_ic": 0.35,
            },
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        poor = [s for s in signals if s.signal_type == "poor_predictor"]
        assert len(poor) == 1
        assert poor[0].target == "gats_Alpha158_origin_N"

    def test_empty_underperformers(self):
        af = _make_findings("Prediction Audit", {
            "per_model_hit_rate": {
                "underperformers": [],
                "per_model_ic": {},
            },
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        assert not any(s.signal_type == "poor_predictor" for s in signals)


# ------------------------------------------------------------------
# Rule 10: regime_instability
# ------------------------------------------------------------------

class TestRegimeInstability:
    def test_frequent_switches(self):
        af = _make_findings("Market Regime", {
            "regime_switches": {
                "switch_count": 5,
                "current_regime": "High-Vol",
                "current_streak_days": 10,
            },
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        reg = [s for s in signals if s.signal_type == "regime_instability"]
        assert len(reg) == 1
        assert reg[0].severity == "info"

    def test_few_switches_no_signal(self):
        af = _make_findings("Market Regime", {
            "regime_switches": {
                "switch_count": 2,
            },
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        assert not any(s.signal_type == "regime_instability" for s in signals)


# ------------------------------------------------------------------
# Rule 11: combo_stale
# ------------------------------------------------------------------

class TestComboStale:
    def test_stale_combo(self):
        af = _make_findings("Ensemble Evolution", {
            "oos_trend": {},
            "model_contributions": {},
            "combo_trends": {
                "default": [{"_date": "2020-01-01"}],  # default is skipped
                "combo_A": [
                    {"_date": "2025-01-01", "total_return": 0.1},
                    {"_date": "2025-02-01", "total_return": 0.12},
                ],
            },
        })
        ext = SignalExtractor(reference_date="2026-04-30")
        signals = ext.extract([af], {})
        stale = [s for s in signals if s.signal_type == "combo_stale"]
        assert len(stale) == 1
        assert stale[0].target == "combo_A"
        assert stale[0].metrics["days_since_eval"] > 30

    def test_recent_combo_no_signal(self):
        af = _make_findings("Ensemble Evolution", {
            "oos_trend": {},
            "model_contributions": {},
            "combo_trends": {
                "combo_B": [
                    {"_date": "2026-04-25", "total_return": 0.15},
                ],
            },
        })
        ext = SignalExtractor(reference_date="2026-04-30")
        signals = ext.extract([af], {})
        assert not any(s.signal_type == "combo_stale" for s in signals)


# ------------------------------------------------------------------
# Rule 12: cross_agent_convergence
# ------------------------------------------------------------------

class TestCrossAgentConvergence:
    def test_same_target_two_agents(self):
        """Same model flagged by Model Health + Prediction Audit → convergence."""
        af1 = _make_findings("Model Health", {
            "convergence_summary": {
                "underfitting_candidates": [],
                "model_details": {},
                "full_epoch_models": [],
            },
            "scorecard": {
                "gats_Alpha158_origin_N": {"ic_trend": "degrading", "ic_mean": 0.02},
            },
            "stale_models": {},
        })
        af2 = _make_findings("Ensemble Evolution", {
            "oos_trend": {},
            "model_contributions": {
                "consistently_negative": ["gats_Alpha158_origin_N"],
                "loo_deltas": {"gats_Alpha158_origin_N": {"mean": -0.05, "count": 5}},
                "model_excess": {"gats_Alpha158_origin_N": {"mean": -0.03}},
            },
            "combo_trends": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af1, af2], {})
        conv = [s for s in signals if s.signal_type == "cross_agent_convergence"]
        assert len(conv) == 1
        assert conv[0].target == "gats_Alpha158_origin_N"
        assert "Model Health" in conv[0].metrics["contributing_agents"]
        assert "Ensemble Evolution" in conv[0].metrics["contributing_agents"]

    def test_no_convergence_single_agent(self):
        af = _make_findings("Model Health", {
            "convergence_summary": {
                "underfitting_candidates": [],
                "model_details": {},
                "full_epoch_models": [],
            },
            "scorecard": {
                "gru_Alpha158": {"ic_trend": "degrading", "ic_mean": 0.04},
            },
            "stale_models": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        conv = [s for s in signals if s.signal_type == "cross_agent_convergence"]
        assert len(conv) == 0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_findings(self):
        ext = SignalExtractor()
        signals = ext.extract([], {})
        assert signals == []

    def test_empty_raw_metrics(self):
        af = _make_findings("Model Health", {})
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        assert signals == []

    def test_per_model_hit_rate_empty_dict(self):
        af = _make_findings("Prediction Audit", {
            "per_model_hit_rate": {},
        })
        ext = SignalExtractor()
        signals = ext.extract([af], {})
        assert signals == []

    def test_signal_to_dict(self):
        s = Signal(
            signal_type="underfitting",
            severity="warning",
            scope="hyperparams",
            source_agent="Model Health",
            target="gru_Alpha158",
            metrics={"epochs": 30},
            context="test",
        )
        d = s.to_dict()
        assert d["signal_type"] == "underfitting"
        assert d["metrics"]["epochs"] == 30
