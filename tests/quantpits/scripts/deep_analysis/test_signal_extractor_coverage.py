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


# ------------------------------------------------------------------
# New: extract_triage_input and helpers
# ------------------------------------------------------------------

def _make_af(agent_name, raw_metrics, window="full"):
    return AgentFindings(
        agent_name=agent_name, window_label=window,
        findings=[], recommendations=[], raw_metrics=raw_metrics,
    )


class TestClassifyArchitecture:
    def test_alpha360_rnn(self):
        assert SignalExtractor._classify_architecture("gru_Alpha360") == "Alpha360_RNN"
        assert SignalExtractor._classify_architecture("lstm_Alpha360") == "Alpha360_RNN"

    def test_alpha360_attention(self):
        assert SignalExtractor._classify_architecture("alstm_Alpha360") == "Alpha360_Attention"
        assert SignalExtractor._classify_architecture("igmtf_Alpha360") == "Alpha360_Attention"

    def test_alpha360_transformer(self):
        assert SignalExtractor._classify_architecture("transformer_Alpha360") == "Alpha360_Transformer"
        assert SignalExtractor._classify_architecture("localformer_Alpha360") == "Alpha360_Transformer"

    def test_alpha360_sfm(self):
        assert SignalExtractor._classify_architecture("sfm_Alpha360") == "Alpha360_SFM"

    def test_alpha360_adarnn(self):
        assert SignalExtractor._classify_architecture("adarnn_Alpha360") == "Alpha360_ADARNN"

    def test_alpha360_tra(self):
        assert SignalExtractor._classify_architecture("tra_Alpha360") == "Alpha360_TRA"

    def test_alpha360_structural(self):
        assert SignalExtractor._classify_architecture("tcn_Alpha360") == "Alpha360_Structural"
        assert SignalExtractor._classify_architecture("sandwich_Alpha360") == "Alpha360_Structural"
        assert SignalExtractor._classify_architecture("krnn_Alpha360") == "Alpha360_Structural"

    def test_alpha360_tabnet(self):
        assert SignalExtractor._classify_architecture("TabNet_Alpha360") == "Alpha360_TabNet"

    def test_alpha360_add(self):
        assert SignalExtractor._classify_architecture("add_Alpha360") == "Alpha360_ADD"

    def test_alpha158_rnn(self):
        assert SignalExtractor._classify_architecture("gru_Alpha158") == "Alpha158_RNN"
        assert SignalExtractor._classify_architecture("gats_Alpha158_plus") == "Alpha158_RNN"

    def test_alpha158_transformer(self):
        assert SignalExtractor._classify_architecture("transformer_Alpha158") == "Alpha158_Transformer"
        assert SignalExtractor._classify_architecture("tft_Alpha158") == "Alpha158_Transformer"

    def test_alpha158_tabular(self):
        assert SignalExtractor._classify_architecture("TabNet_Alpha158") == "Alpha158_Tabular"
        assert SignalExtractor._classify_architecture("catboost_Alpha158") == "Alpha158_Tabular"
        assert SignalExtractor._classify_architecture("lightgbm_Alpha158") == "Alpha158_Tabular"
        assert SignalExtractor._classify_architecture("mlp_Alpha158") == "Alpha158_Tabular"

    def test_alpha158_tra(self):
        assert SignalExtractor._classify_architecture("tra_Alpha158_full") == "Alpha158_TRA"

    def test_unknown_fallback(self):
        assert SignalExtractor._classify_architecture("mystery_Alpha360") == "Alpha360_Other"
        assert SignalExtractor._classify_architecture("mystery_Alpha158") == "Alpha158_Other"
        assert SignalExtractor._classify_architecture("unknown_model") == "Unknown"


class TestExtractTriageInput:
    def test_builds_all_sections(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "ensemble_config.json").write_text(json.dumps({
            "combos": {"test_combo": {"models": ["model_a"]}}
        }))

        af_health = _make_af("Model Health", {
            "scorecard": {
                "model_a": {"ic_mean": 0.04, "icir_mean": 0.3, "ic_trend": "stable",
                            "n_snapshots": 10},
                "model_b": {"ic_mean": 0.01, "icir_mean": 0.05, "ic_trend": "degrading",
                            "n_snapshots": 5},
            },
            "convergence_summary": {
                "model_details": {
                    "model_a": {"best_epoch": 5, "actual_epochs": 40, "early_stopped": True},
                    "model_b": {"best_epoch": 0, "actual_epochs": 5, "early_stopped": True},
                },
                "underfitting_candidates": [],
                "full_epoch_models": [],
            },
        })
        af_regime = _make_af("Market Regime", {
            "regime_switches": {"switch_count": 3, "current_regime": "Bullish"},
        })
        af_ensemble = _make_af("Ensemble Evolution", {
            "combo_trends": {
                "test_combo": [{"_date": "2026-05-01", "excess_return": 0.02,
                                "calmar": 3.0, "sharpe": 1.5}],
            },
            "oos_trend": {"oos_calmar_slope": -0.5, "oos_runs": 6,
                          "latest_oos_calmar": 2.0},
        })

        ext = SignalExtractor(workspace_root=str(ws))
        signals = ext.extract([af_health, af_regime, af_ensemble], {})
        triage = ext.extract_triage_input(
            [af_health, af_regime, af_ensemble], {}, signals,
        )

        assert "market_context" in triage
        assert "model_ranking" in triage
        assert "family_stats" in triage
        assert "combo_summary" in triage
        assert "signal_distribution" in triage

    def test_model_ranking_deduplicates(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "ensemble_config.json").write_text("{}")

        # Two windows with the same model — should only appear once
        af_full = _make_af("Model Health", {
            "scorecard": {"model_a": {"ic_mean": 0.04, "icir_mean": 0.3,
                                      "ic_trend": "stable", "n_snapshots": 10}},
            "convergence_summary": {"model_details": {}, "underfitting_candidates": [],
                                    "full_epoch_models": []},
        }, window="full")
        af_1m = _make_af("Model Health", {
            "scorecard": {"model_a": {"ic_mean": 0.03, "icir_mean": 0.25,
                                      "ic_trend": "stable", "n_snapshots": 3}},
            "convergence_summary": {"model_details": {}, "underfitting_candidates": [],
                                    "full_epoch_models": []},
        }, window="1m")

        ext = SignalExtractor(workspace_root=str(ws))
        triage = ext.extract_triage_input([af_full, af_1m], {}, [])
        ranking = triage["model_ranking"]
        # Only one entry, with the higher n_snapshots
        assert len(ranking) == 1
        assert ranking[0]["n_snapshots"] == 10

    def test_family_stats_aggregation(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "ensemble_config.json").write_text("{}")

        af = _make_af("Model Health", {
            "scorecard": {
                "gru_Alpha360": {"ic_mean": 0.04, "icir_mean": 0.25, "ic_trend": "stable",
                                 "n_snapshots": 10},
                "lstm_Alpha360": {"ic_mean": 0.05, "icir_mean": 0.30, "ic_trend": "stable",
                                  "n_snapshots": 10},
                "gru_Alpha158": {"ic_mean": 0.005, "icir_mean": 0.03, "ic_trend": "degrading",
                                 "n_snapshots": 10},
            },
            "convergence_summary": {"model_details": {}, "underfitting_candidates": [],
                                    "full_epoch_models": []},
        })

        ext = SignalExtractor(workspace_root=str(ws))
        triage = ext.extract_triage_input([af], {}, [])
        stats = triage["family_stats"]
        assert "Alpha360_RNN" in stats
        assert stats["Alpha360_RNN"]["count"] == 2
        assert stats["Alpha360_RNN"]["avg_ic"] > 0.04

    def test_signal_distribution_counts(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "ensemble_config.json").write_text("{}")

        ext = SignalExtractor(workspace_root=str(ws))
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Model Health", "model_a",
                   {"ic": 0.01}),
            Signal("underfitting", "warning", "hyperparams", "Model Health", "model_b",
                   {"ic": 0.02}),
            Signal("ic_decay", "warning", "hyperparams", "Model Health", "model_a",
                   {"ic_trend": "degrading"}),
            Signal("negative_contribution", "critical", "model_selection",
                   "Ensemble Evolution", "model_c", {}),
        ]
        dist = ext._build_signal_distribution(signals)
        assert dist["total_signals"] == 4
        assert dist["unique_targets"] == 3
        assert dist["by_severity"]["warning"] == 3
        assert dist["by_severity"]["critical"] == 1

    def test_market_context_from_regime(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "ensemble_config.json").write_text("{}")

        af_regime = _make_af("Market Regime", {
            "regime_switches": {"switch_count": 5, "current_regime": "Sideways"},
        })
        ext = SignalExtractor(workspace_root=str(ws))
        triage = ext.extract_triage_input([af_regime], {"executive_summary_data": {
            "market_regime": "Sideways", "cagr_1y": 0.05, "sharpe_1y": 0.5,
        }}, [])
        ctx = triage["market_context"]
        assert ctx["current_regime"] == "Sideways"
        assert ctx["regime_switch_count"] == 5

    def test_combo_summary_with_oos_trend(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "ensemble_config.json").write_text("{}")

        af = _make_af("Ensemble Evolution", {
            "combo_trends": {},
            "oos_trend": {"oos_calmar_slope": -0.69, "oos_runs": 6,
                          "latest_oos_calmar": 1.87},
        })
        ext = SignalExtractor(workspace_root=str(ws))
        triage = ext.extract_triage_input([af], {}, [])
        assert "_oos_trend" in triage["combo_summary"]
        assert triage["combo_summary"]["_oos_trend"]["calmar_slope"] == -0.69

    def test_no_ensemble_config(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        ext = SignalExtractor(workspace_root=str(ws))
        membership = ext._load_combo_membership()
        assert membership == {}

    def test_empty_signals_produces_valid_triage(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "ensemble_config.json").write_text("{}")

        af = _make_af("Model Health", {
            "scorecard": {},
            "convergence_summary": {"model_details": {}, "underfitting_candidates": [],
                                    "full_epoch_models": []},
        })
        ext = SignalExtractor(workspace_root=str(ws))
        triage = ext.extract_triage_input([af], {}, [])
        assert triage["model_ranking"] == []
        assert triage["signal_distribution"]["total_signals"] == 0


# ------------------------------------------------------------------
# Optimizer thrashing detection
# ------------------------------------------------------------------

class TestExtractOptimizerThrashing:
    def test_no_history_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        ext = SignalExtractor(workspace_root=str(ws))
        result = ext._extract_optimizer_thrashing()
        assert result == []

    def test_no_epoch_loss_data(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        records = [
            {"model_name": "m1", "trained_at": "2026-05-01", "n_epochs": 200},
        ]
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ext = SignalExtractor(workspace_root=str(ws))
        result = ext._extract_optimizer_thrashing()
        assert result == []

    def test_too_few_epochs(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        records = [
            {"model_name": "m1", "trained_at": "2026-05-01",
             "epoch_train_loss": [0.5, 0.4]},  # only 2 epochs, need >=3
        ]
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ext = SignalExtractor(workspace_root=str(ws))
        result = ext._extract_optimizer_thrashing()
        assert result == []

    def test_critical_thrashing(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        # Alternating losses: each adjacent change > 5% → high thrash ratio
        losses = []
        for i in range(20):
            losses.append(0.5 if i % 2 == 0 else 0.4)  # ~20% change each step
        records = [
            {"model_name": "m1", "trained_at": "2026-05-10",
             "epoch_train_loss": losses},
        ]
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ext = SignalExtractor(workspace_root=str(ws))
        result = ext._extract_optimizer_thrashing()
        assert len(result) == 1
        assert result[0].signal_type == "optimizer_thrashing"
        assert result[0].severity == "critical"

    def test_warning_thrashing(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        # One spike every 10 epochs → ~10% thrash events → under 30% but over 15%
        # Wait, we need >15% and <=30%. With 20 epochs, that's 4-6 events.
        # Let's use 3 events in 19 pairs = 15.8% → barely into warning territory
        losses = [0.5] * 20
        losses[5] = 0.35   # spike: 5→6 pair has 30% change from 0.35→0.5
        losses[10] = 0.35  # spike
        losses[15] = 0.35  # spike
        # This creates 6 thrash events (up+down for each spike) in 19 pairs = 31.5%
        # Too many. Let me just use fewer epochs.
        # 10 epochs, 1 spike = 2 events in 9 pairs = 22% → warning
        short_losses = [0.5] * 10
        short_losses[5] = 0.35
        records = [
            {"model_name": "m1", "trained_at": "2026-05-10",
             "epoch_train_loss": short_losses},
        ]
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ext = SignalExtractor(workspace_root=str(ws))
        result = ext._extract_optimizer_thrashing()
        assert len(result) == 1
        assert result[0].severity == "warning"

    def test_below_threshold_skipped(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        # Stable decreasing loss → low thrash ratio
        losses = [0.5 - i * 0.001 for i in range(20)]  # very gradual change
        records = [
            {"model_name": "m1", "trained_at": "2026-05-10",
             "epoch_train_loss": losses},
        ]
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ext = SignalExtractor(workspace_root=str(ws))
        result = ext._extract_optimizer_thrashing()
        assert result == []

    def test_latest_record_per_model(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        # Two records for same model, only latest should be used
        records = [
            {"model_name": "m1", "trained_at": "2026-05-01",
             "epoch_train_loss": [0.5, 0.4, 0.5, 0.4]},  # old, thrashing
            {"model_name": "m1", "trained_at": "2026-05-10",
             "epoch_train_loss": [0.5 - i * 0.001 for i in range(5)]},  # new, stable
        ]
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ext = SignalExtractor(workspace_root=str(ws))
        result = ext._extract_optimizer_thrashing()
        # Latest record is stable → no signal
        assert result == []

    def test_corrupt_history_file(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        (ws / "data" / "training_history.jsonl").write_text("not valid json\n")

        ext = SignalExtractor(workspace_root=str(ws))
        result = ext._extract_optimizer_thrashing()
        assert result == []

    def test_none_values_in_losses(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        losses = [0.5, None, 0.4, 0.3, 0.2]  # None breaks the relative change chain
        records = [
            {"model_name": "m1", "trained_at": "2026-05-10",
             "epoch_train_loss": losses},
        ]
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ext = SignalExtractor(workspace_root=str(ws))
        result = ext._extract_optimizer_thrashing()
        # Only one valid adjacent pair (0.4→0.3: 25%, 0.3→0.2: 33%)
        # 2/2 > 30% → critical
        assert len(result) == 1

    def test_invalid_json_line_skipped(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        lines = [
            '{"model_name": "m1", "trained_at": "2026-05-10", "epoch_train_loss": [0.5, 0.4, 0.5, 0.4, 0.5, 0.4]}\n',
            'not json\n',
            '\n',
        ]
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            f.writelines(lines)

        ext = SignalExtractor(workspace_root=str(ws))
        result = ext._extract_optimizer_thrashing()
        assert len(result) == 1  # only the valid line counted
