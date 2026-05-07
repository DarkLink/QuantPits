"""Coverage tests for SignalExtractor uncovered paths."""

import json
import pytest
from quantpits.scripts.deep_analysis.signal_extractor import SignalExtractor, Signal
from quantpits.scripts.deep_analysis.base_agent import AgentFindings, Finding


class TestSignal:
    def test_to_dict(self):
        s = Signal(
            signal_type="underfitting",
            severity="warning",
            scope="hyperparams",
            source_agent="Model Health",
            target="gru_Alpha158",
            context="test context",
            metrics={"ic": 0.01},
        )
        d = s.to_dict()
        assert d["signal_type"] == "underfitting"
        assert d["severity"] == "warning"
        assert d["scope"] == "hyperparams"
        assert d["source_agent"] == "Model Health"
        assert d["target"] == "gru_Alpha158"
        assert d["context"] == "test context"
        assert d["metrics"] == {"ic": 0.01}

    def test_to_dict_defaults(self):
        s = Signal(
            signal_type="ic_decay",
            severity="critical",
            scope="hyperparams",
            source_agent="Test Agent",
            target="m1",
        )
        d = s.to_dict()
        assert d["context"] == ""
        assert d["metrics"] == {}

    def test_repr(self):
        s = Signal(
            signal_type="ic_decay",
            severity="critical",
            scope="hyperparams",
            source_agent="Test Agent",
            target="m1",
        )
        r = repr(s)
        assert "ic_decay" in r
        assert "m1" in r


class TestSignalExtractorBasic:
    def test_init(self):
        ext = SignalExtractor()
        assert hasattr(ext, 'extract')

    def test_extract_empty(self):
        ext = SignalExtractor()
        signals = ext.extract([], {})
        assert signals == []

    def test_extract_with_empty_findings(self):
        ext = SignalExtractor()
        af = AgentFindings(
            agent_name="Model Health",
            window_label="1m",
            findings=[],
            recommendations=[],
            raw_metrics={},
        )
        signals = ext.extract([af], {})
        assert isinstance(signals, list)


class TestExtractModelHealth:
    """Test _extract_model_health with actual data structures."""

    def test_empty_findings(self):
        ext = SignalExtractor()
        signals = ext._extract_model_health({})
        assert signals == []

    def test_underfitting_candidates(self):
        """convergence_summary with underfitting_candidates generates signals.

        Note: actual_epochs=10, configured=200 => ratio=0.05 < 0.15,
        so the signal is upgraded from 'underfitting' to 'severe_underfitting'.
        """
        ext = SignalExtractor()
        af = AgentFindings(
            agent_name="Model Health",
            window_label="1m",
            findings=[],
            recommendations=[],
            raw_metrics={
                "convergence_summary": {
                    "underfitting_candidates": ["gru_Alpha158"],
                    "model_details": {
                        "gru_Alpha158": {
                            "actual_epochs": 10,
                            "configured_epochs": 200,
                            "early_stopped": True,
                        }
                    },
                },
            },
        )
        signals = ext._extract_model_health({"Model Health": [af]})
        assert len(signals) >= 1
        # actual=10/200=5% < 15% threshold → gets upgraded to severe_underfitting
        # actual=10/200=5% < 15% threshold → gets upgraded to severe_underfitting
        assert any("underfitting" in s.signal_type for s in signals)
        assert any(s.target == "gru_Alpha158" for s in signals)

    def test_severe_underfitting(self):
        """actual_epochs < configured * 0.15 should generate severe signal."""
        ext = SignalExtractor()
        af = AgentFindings(
            agent_name="Model Health",
            window_label="1m",
            findings=[],
            recommendations=[],
            raw_metrics={
                "convergence_summary": {
                    "underfitting_candidates": [],  # not already flagged
                    "model_details": {
                        "gru_Alpha158": {
                            "actual_epochs": 15,
                            "configured_epochs": 200,  # 15/200 = 0.075 < 0.15
                            "early_stopped": True,
                        }
                    },
                    "full_epoch_models": [],
                },
                "scorecard": {},
            },
        )
        signals = ext._extract_model_health({"Model Health": [af]})
        assert any(s.signal_type == "severe_underfitting" and s.target == "gru_Alpha158"
                   for s in signals)

    def test_ic_decay_signal(self):
        """ic_trend == 'degrading' should generate ic_decay signal."""
        ext = SignalExtractor()
        af = AgentFindings(
            agent_name="Model Health",
            window_label="1m",
            findings=[],
            recommendations=[],
            raw_metrics={
                "convergence_summary": {
                    "underfitting_candidates": [],
                    "model_details": {},
                    "full_epoch_models": [],
                },
                "scorecard": {
                    "model_a": {"ic_trend": "degrading", "ic_mean": 0.02},
                },
            },
        )
        signals = ext._extract_model_health({"Model Health": [af]})
        assert any(s.signal_type == "ic_decay" and s.target == "model_a"
                   for s in signals)

    def test_overfitting_signal(self):
        """full_epoch_models with low IC mean generates overfitting signal."""
        ext = SignalExtractor()
        af = AgentFindings(
            agent_name="Model Health",
            window_label="1m",
            findings=[],
            recommendations=[],
            raw_metrics={
                "convergence_summary": {
                    "underfitting_candidates": [],
                    "model_details": {},
                    "full_epoch_models": ["overfit_model"],
                },
                "scorecard": {
                    "overfit_model": {"ic_trend": "stable", "ic_mean": 0.01},
                },
            },
        )
        signals = ext._extract_model_health({"Model Health": [af]})
        assert any(s.signal_type == "overfitting" and s.target == "overfit_model"
                   for s in signals)

    def test_healthy_model_no_signals(self):
        """Healthy model with good metrics should not generate signals."""
        ext = SignalExtractor()
        af = AgentFindings(
            agent_name="Model Health",
            window_label="1m",
            findings=[],
            recommendations=[],
            raw_metrics={
                "convergence_summary": {
                    "underfitting_candidates": [],
                    "model_details": {"m1": {"actual_epochs": 150, "configured_epochs": 200}},
                    "full_epoch_models": [],
                },
                "scorecard": {
                    "m1": {"ic_trend": "stable", "ic_mean": 0.05},
                },
            },
        )
        signals = ext._extract_model_health({"Model Health": [af]})
        assert len(signals) == 0

    def test_no_model_health_agent(self):
        """Agent list without Model Health should return empty."""
        ext = SignalExtractor()
        af = AgentFindings(
            agent_name="Other Agent",
            window_label="1m",
            findings=[],
            recommendations=[],
            raw_metrics={},
        )
        signals = ext._extract_model_health({"Other Agent": [af]})
        assert signals == []


class TestExtractPredictionAudit:
    """Test _extract_prediction_audit coverage."""

    def test_empty_findings(self):
        ext = SignalExtractor()
        signals = ext._extract_prediction_audit({})
        assert signals == []

    def test_no_prediction_audit_agent(self):
        ext = SignalExtractor()
        af = AgentFindings(
            agent_name="Other Agent",
            window_label="1m",
            findings=[],
            recommendations=[],
            raw_metrics={},
        )
        signals = ext._extract_prediction_audit({"Other Agent": [af]})
        assert signals == []


class TestExtractEnsembleEval:
    """Test _extract_ensemble_eval coverage."""

    def test_empty_findings(self):
        ext = SignalExtractor()
        signals = ext._extract_ensemble_eval({})
        assert signals == []


class TestExtractMarketRegime:
    """Test _extract_market_regime coverage."""

    def test_empty_findings(self):
        ext = SignalExtractor()
        signals = ext._extract_market_regime({})
        assert signals == []
