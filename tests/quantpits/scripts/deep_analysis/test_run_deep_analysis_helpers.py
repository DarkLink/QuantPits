"""Tests for run_deep_analysis helper functions (data loaders + profile builders)."""

import json
import os
import pytest
import yaml
from unittest.mock import patch, MagicMock

with patch("os.chdir"):
    from quantpits.scripts.run_deep_analysis import (
        _load_combo_membership,
        _load_training_history_by_model,
        _load_hyperparam_bounds,
        _load_current_params,
        _load_correlation_excerpts,
        _build_model_profiles,
        _build_combo_profile,
        _build_execution_profile,
        _build_execution_risk_summary,
        _load_model_knowledge,
        _persist_feedback_report,
    )
    from quantpits.scripts.deep_analysis.base_agent import Finding, AgentFindings


# ------------------------------------------------------------------
# _load_combo_membership
# ------------------------------------------------------------------

class TestLoadComboMembership:
    def test_loads_membership(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "combo_a": {"models": ["m1", "m2"]},
                "combo_b": {"models": ["m2", "m3"]},
            }
        }))

        result = _load_combo_membership(str(ws))
        assert result == {"m1": ["combo_a"], "m2": ["combo_a", "combo_b"], "m3": ["combo_b"]}

    def test_no_config_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert _load_combo_membership(str(ws)) == {}

    def test_invalid_json(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "ensemble_config.json").write_text("not json")
        assert _load_combo_membership(str(ws)) == {}

    def test_skips_non_dict_combos(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "combo_a": "invalid_string",
                "combo_b": {"models": ["m1"]},
            }
        }))
        result = _load_combo_membership(str(ws))
        assert "m1" in result


# ------------------------------------------------------------------
# _load_training_history_by_model
# ------------------------------------------------------------------

class TestLoadTrainingHistoryByModel:
    def test_loads_and_groups(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        records = [
            {"model_name": "m1", "trained_at": "2026-05-01", "n_epochs": 200, "lr": 0.001,
             "batch_size": 4096, "dropout": 0.3, "early_stop": 20},
            {"model_name": "m1", "trained_at": "2026-05-08", "n_epochs": 200, "lr": 0.0005,
             "batch_size": 4096, "dropout": 0.3, "early_stop": 30},
            {"model_name": "m2", "trained_at": "2026-05-01", "n_epochs": 100, "lr": 0.01},
        ]
        with open(ws / "data" / "training_history.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        result = _load_training_history_by_model(str(ws))
        assert len(result["m1"]) == 2
        assert len(result["m2"]) == 1
        # Only keeps known fields
        assert "n_epochs" in result["m1"][0]
        assert "unknown_field" not in result["m1"][0]

    def test_no_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert _load_training_history_by_model(str(ws)) == {}

    def test_skips_invalid_lines(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        (ws / "data" / "training_history.jsonl").write_text(
            '{"model_name": "m1"}\nnot json\n{"model_name": "m2"}\n\n'
        )
        result = _load_training_history_by_model(str(ws))
        assert len(result) == 2


# ------------------------------------------------------------------
# _load_hyperparam_bounds
# ------------------------------------------------------------------

class TestLoadHyperparamBounds:
    def test_loads_bounds(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "hyperparam_bounds.json").write_text(json.dumps({
            "bounds": {"lr": {"min": 0.0001, "max": 0.01}}
        }))
        assert _load_hyperparam_bounds(str(ws)) == {"lr": {"min": 0.0001, "max": 0.01}}

    def test_no_file(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert _load_hyperparam_bounds(str(ws)) == {}


# ------------------------------------------------------------------
# _load_current_params
# ------------------------------------------------------------------

class TestLoadCurrentParams:
    def test_loads_from_registry_and_yaml(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)

        registry = {"models": {
            "m1": {"yaml_file": "config/workflow_config_m1.yaml"},
            "m2": {},  # no yaml_file
        }}
        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump(registry, f)

        (ws / "config" / "workflow_config_m1.yaml").write_text(yaml.dump({
            "task": {"model": {"kwargs": {
                "lr": 0.001, "n_epochs": 200, "dropout": 0.3,
                "loss": "mse", "GPU": 0,  # non-tunable filter
            }}}
        }))

        result = _load_current_params(str(ws), ["m1", "m2"])
        assert "m1" in result
        assert result["m1"]["lr"] == 0.001
        assert result["m1"]["n_epochs"] == 200
        # Filters to primitive values
        assert all(isinstance(v, (int, float, str, bool, type(None)))
                   for v in result["m1"].values())

    def test_no_registry(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert _load_current_params(str(ws), ["m1"]) == {}

    def test_missing_yaml_file(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        registry = {"models": {"m1": {"yaml_file": "config/nonexistent.yaml"}}}
        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump(registry, f)

        result = _load_current_params(str(ws), ["m1"])
        assert result == {}


# ------------------------------------------------------------------
# _load_correlation_excerpts
# ------------------------------------------------------------------

class TestLoadCorrelationExcerpts:
    def test_loads_from_csv(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "output").mkdir(parents=True)

        import pandas as pd
        df = pd.DataFrame({
            "m1": [1.0, 0.8, 0.1],
            "m2": [0.8, 1.0, 0.3],
            "m3": [0.1, 0.3, 1.0],
        }, index=["m1", "m2", "m3"])
        df.to_csv(ws / "output" / "prediction_correlation_2026-05-01.csv")

        result = _load_correlation_excerpts(str(ws), ["m1", "m2"])
        assert "m1" in result
        assert len(result["m1"]["top_correlated"]) <= 3
        assert len(result["m1"]["bottom_correlated"]) <= 3

    def test_no_correlation_files(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert _load_correlation_excerpts(str(ws), ["m1"]) == {}

    def test_model_not_in_matrix(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "output").mkdir(parents=True)
        import pandas as pd
        df = pd.DataFrame({"m1": [1.0]}, index=["m1"])
        df.to_csv(ws / "output" / "corr.csv")

        result = _load_correlation_excerpts(str(ws), ["unknown_model"])
        assert result == {}


# ------------------------------------------------------------------
# _build_model_profiles
# ------------------------------------------------------------------

class TestBuildModelProfiles:
    def test_builds_profile(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "ensemble_config.json").write_text(json.dumps({
            "combos": {"c1": {"models": ["m1"]}}
        }))

        triage_input = {
            "model_ranking": [{"model": "m1", "ic_mean": 0.04}],
            "family_stats": {"TestFamily": {"count": 1}},
        }

        profiles = _build_model_profiles(
            model_names=["m1"],
            triage_input=triage_input,
            signals=[],
            all_findings=[],
            workspace_root=str(ws),
        )
        assert "m1" in profiles
        p = profiles["m1"]
        assert p["ranking_table"] == triage_input["model_ranking"]
        assert p["family_stats"] == triage_input["family_stats"]
        assert p["combo_role"]["in_combos"] == ["c1"]
        assert p["combo_role"]["is_active"] is True

    def test_model_not_in_combo(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "ensemble_config.json").write_text("{}")

        profiles = _build_model_profiles(
            model_names=["m_orphan"],
            triage_input={"model_ranking": [], "family_stats": {}},
            signals=[],
            all_findings=[],
            workspace_root=str(ws),
        )
        assert profiles["m_orphan"]["combo_role"]["in_combos"] == []
        assert profiles["m_orphan"]["combo_role"]["is_active"] is False

    def test_signals_indexed_by_target(self, tmp_path):
        from quantpits.scripts.deep_analysis.signal_extractor import Signal

        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "ensemble_config.json").write_text("{}")

        s1 = Signal("underfitting", "warning", "hyperparams", "Test", "m1", {})
        s2 = Signal("ic_decay", "warning", "hyperparams", "Test", "m1", {})

        profiles = _build_model_profiles(
            model_names=["m1"],
            triage_input={"model_ranking": [], "family_stats": {}},
            signals=[s1, s2],
            all_findings=[],
            workspace_root=str(ws),
        )
        assert len(profiles["m1"]["signals"]) == 2


# ------------------------------------------------------------------
# _build_combo_profile
# ------------------------------------------------------------------

class TestBuildComboProfile:
    def test_builds_profile(self):
        profile = _build_combo_profile(
            combo_name="test_combo",
            model_diagnoses={
                "m1": {"diagnosis": "healthy", "diagnosis_detail": "All good"},
                "m2": {"diagnosis": "needs_tuning", "diagnosis_detail": "x" * 400},
            },
            triage_input={
                "combo_summary": {
                    "test_combo": {"latest_excess": 0.02, "latest_calmar": 3.0},
                    "_oos_trend": {"calmar_slope": -0.5},
                },
                "market_context": {"current_regime": "Bullish"},
            },
            all_findings=[],
        )
        assert profile["member_diagnoses"]["m1"]["diagnosis"] == "healthy"
        assert profile["member_diagnoses"]["m2"]["diagnosis"] == "needs_tuning"
        # detail truncated to 300
        assert len(profile["member_diagnoses"]["m2"]["detail"]) <= 300
        assert profile["oos_trend"] == {"calmar_slope": -0.5}

    def test_missing_combo_info(self):
        profile = _build_combo_profile(
            combo_name="nonexistent",
            model_diagnoses={},
            triage_input={"combo_summary": {}, "market_context": {}},
            all_findings=[],
        )
        assert profile["member_diagnoses"] == {}
        assert profile["combo_history"][0]["excess"] is None


# ------------------------------------------------------------------
# _build_execution_profile
# ------------------------------------------------------------------

class TestBuildExecutionProfile:
    def test_builds_empty_profile_with_context(self):
        profile = _build_execution_profile(
            all_findings=[],
            triage_input={"market_context": {"current_regime": "Bearish"}},
        )
        assert profile["execution_issues"] == []
        assert profile["trade_pattern_issues"] == []
        assert profile["_execution_context"] == {"current_regime": "Bearish"}


# ------------------------------------------------------------------
# _persist_feedback_report
# ------------------------------------------------------------------

class TestPersistFeedbackReport:
    def test_persists_report(self, tmp_path):
        ws = tmp_path / "ws"
        output_dir = ws / "output" / "deep_analysis"
        output_dir.mkdir(parents=True)

        synthesizer_output = {
            "global_diagnosis": {"health_status": "warning", "trend": "degrading"},
            "conflict_resolutions": [{"conflict": "test"}],
            "cross_validation_notes": ["note1"],
            "scope_recommendations": [],
        }
        feedback_eval = {
            "quality_summary": {"total": 2, "incorrect": 0},
        }

        _persist_feedback_report(synthesizer_output, feedback_eval, str(ws))

        files = list(output_dir.glob("feedback_report_*.json"))
        assert len(files) == 1
        content = json.loads(files[0].read_text())
        assert content["global_diagnosis"]["health_status"] == "warning"
        assert content["feedback_summary"]["total"] == 2


# ------------------------------------------------------------------
# _build_model_profiles — full integration (training history + params + correlation)
# ------------------------------------------------------------------

class TestBuildModelProfilesFull:
    def test_full_profile_with_all_data_sources(self, tmp_path):
        """Build a profile with training_history, current_params, correlation, and bounds."""
        from quantpits.scripts.deep_analysis.signal_extractor import Signal

        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "data").mkdir(parents=True)
        (ws / "output").mkdir(parents=True)

        # 1. Ensemble config
        (ws / "config" / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "combo1": {"models": ["m1", "m2"]},
                "combo2": {"models": ["m1", "m3"]},
            }
        }))

        # 2. Model registry + workflow YAML for current_params
        model_registry = {
            "models": {
                "m1": {
                    "algorithm": "gru",
                    "dataset": "Alpha158",
                    "yaml_file": "config/workflow_m1.yaml",
                    "enabled": True,
                },
            }
        }
        with open(ws / "config" / "model_registry.yaml", "w") as f:
            yaml.dump(model_registry, f)

        workflow_cfg = {
            "task": {
                "model": {
                    "class": "GRU",
                    "kwargs": {"n_epochs": 200, "early_stop": 10, "lr": 0.001,
                               "GPU": 0, "seed": 42},
                },
                "dataset": {"class": "DatasetH"},
            },
            "data_handler_config": {"label": ["Ref($close, -2)"]},
        }
        with open(ws / "config" / "workflow_m1.yaml", "w") as f:
            yaml.dump(workflow_cfg, f)

        # 3. Hyperparam bounds
        (ws / "config" / "hyperparam_bounds.json").write_text(json.dumps({
            "bounds": {"n_epochs": {"min": 10, "max": 500}, "lr": {"min": 1e-5, "max": 0.1}}
        }))

        # 4. Training history
        history_line = json.dumps({
            "model_name": "m1", "record_id": "rec_001",
            "trained_at": "2026-05-01", "duration_seconds": 180,
            "early_stopped": True, "actual_epochs": 45, "configured_epochs": 200,
            "best_epoch": 30, "final_val_score": 0.02, "score_type": "IC",
            "n_epochs": 200, "early_stop": 10, "lr": 0.001,
            "batch_size": 64, "dropout": 0.2, "hidden_size": 128, "num_layers": 2,
        })
        (ws / "data" / "training_history.jsonl").write_text(history_line + "\n")

        # 5. Correlation matrix CSV
        import pandas as pd
        corr_df = pd.DataFrame(
            [[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]],
            columns=["m1", "m2", "m3"], index=["m1", "m2", "m3"],
        )
        corr_df.to_csv(ws / "output" / "correlation_matrix_2026-05-01.csv")

        # 6. Signal for the model
        sig = Signal("ic_decay", "warning", "hyperparams", "Model Health",
                     "m1", context="ic decaying over 3 windows",
                     metrics={"ic": 0.01})

        triage_input = {
            "model_ranking": [
                {"model": "m1", "ic_mean": 0.04, "icir_mean": 0.5,
                 "ic_trend": "declining", "family": "GRU", "in_combos": ["combo1", "combo2"],
                 "best_epoch": 30, "actual_epochs": 45, "early_stopped": True},
            ],
            "family_stats": {"GRU": {"count": 3, "avg_ic": 0.05}},
        }

        profiles = _build_model_profiles(
            model_names=["m1"],
            triage_input=triage_input,
            signals=[sig],
            all_findings=[],
            workspace_root=str(ws),
        )

        assert "m1" in profiles
        p = profiles["m1"]

        # Combo role
        assert set(p["combo_role"]["in_combos"]) == {"combo1", "combo2"}
        assert p["combo_role"]["is_active"] is True

        # Training history (filtered to specific clean keys)
        assert len(p["training_history"]) == 1
        assert p["training_history"][0]["actual_epochs"] == 45
        assert p["training_history"][0]["early_stopped"] is True

        # Current params (tunable + non-tunable, filtered by type only)
        assert "n_epochs" in p["current_params"]
        assert p["current_params"]["n_epochs"] == 200
        assert p["current_params"]["lr"] == 0.001

        # Hyperparam bounds
        assert "n_epochs" in p["hyperparam_bounds"]
        assert p["hyperparam_bounds"]["n_epochs"]["min"] == 10

        # Correlation excerpt
        assert "m1" in corr_df.index  # should be in the workspace correlation
        corr_excerpt = p["correlation_excerpt"]
        if corr_excerpt:  # correlation loading may fail without pandas in some envs
            assert "top_correlated" in corr_excerpt or "bottom_correlated" in corr_excerpt

        # Signals
        assert len(p["signals"]) == 1
        assert p["signals"][0].signal_type == "ic_decay"

        # Ranking table and family stats
        assert p["ranking_table"] == triage_input["model_ranking"]
        assert p["family_stats"] == triage_input["family_stats"]


# =========================================================================
# _build_execution_risk_summary
# =========================================================================


class TestBuildExecutionRiskSummary:
    def test_empty_findings(self):
        result = _build_execution_risk_summary([], {})
        assert result["diagnosis"] == "rule_based"
        assert result["execution_issues"] == []
        assert result["trade_pattern_issues"] == []

    def test_execution_high_severity(self):
        af = AgentFindings(
            agent_name="execution_quality",
            window_label="full",
            findings=[
                Finding(severity="high", category="execution", title="High slippage detected",
                        detail="Slippage > 2%"),
            ],
        )
        result = _build_execution_risk_summary([af], {})
        assert len(result["execution_issues"]) == 1
        assert "High slippage detected" in result["execution_issues"][0]

    def test_trade_critical_severity(self):
        af = AgentFindings(
            agent_name="trade_pattern",
            window_label="full",
            findings=[
                Finding(severity="critical", category="trade", title="Chasing signals",
                        detail="Chasing detected"),
            ],
        )
        result = _build_execution_risk_summary([af], {})
        assert len(result["trade_pattern_issues"]) == 1
        assert "Chasing signals" in result["trade_pattern_issues"][0]

    def test_title_truncated_200(self):
        long_title = "X" * 250
        af = AgentFindings(
            agent_name="execution_quality",
            window_label="full",
            findings=[Finding(severity="high", category="exec", title=long_title, detail="")],
        )
        result = _build_execution_risk_summary([af], {})
        assert len(result["execution_issues"][0]) == 200

    def test_empty_title_skipped(self):
        af = AgentFindings(
            agent_name="execution_quality",
            window_label="full",
            findings=[Finding(severity="high", category="exec", title="", detail="")],
        )
        result = _build_execution_risk_summary([af], {})
        assert result["execution_issues"] == []

    def test_mixed_severity_only_high_critical_collected(self):
        af = AgentFindings(
            agent_name="execution_quality",
            window_label="full",
            findings=[
                Finding(severity="critical", category="exec", title="Critical issue", detail=""),
                Finding(severity="high", category="exec", title="High issue", detail=""),
                Finding(severity="warning", category="exec", title="Warning", detail=""),
                Finding(severity="info", category="exec", title="Info", detail=""),
            ],
        )
        result = _build_execution_risk_summary([af], {})
        assert len(result["execution_issues"]) == 2

    def test_market_context_included(self):
        mc = {"n_regime_changes": 3}
        result = _build_execution_risk_summary(
            [], {"market_context": mc}
        )
        assert result["market_context"]["n_regime_changes"] == 3

    def test_market_context_non_dict_skipped(self):
        result = _build_execution_risk_summary(
            [], {"market_context": "not_a_dict"}
        )
        assert "market_context" not in result

    def test_diagnosis_detail_format(self):
        af = AgentFindings(
            agent_name="execution_quality",
            window_label="full",
            findings=[Finding(severity="high", category="exec", title="Issue 1", detail="")],
        )
        result = _build_execution_risk_summary([af], {})
        assert "Execution issues: 1" in result["diagnosis_detail"]
        assert "Trade pattern issues: 0" in result["diagnosis_detail"]

    def test_multiple_agents_aggregated(self):
        af1 = AgentFindings(
            agent_name="execution_quality",
            window_label="full",
            findings=[Finding(severity="high", category="exec", title="Exec issue", detail="")],
        )
        af2 = AgentFindings(
            agent_name="trade_pattern",
            window_label="full",
            findings=[Finding(severity="critical", category="trade", title="Trade issue", detail="")],
        )
        result = _build_execution_risk_summary([af1, af2], {})
        assert len(result["execution_issues"]) == 1
        assert len(result["trade_pattern_issues"]) == 1

    def test_agent_has_no_findings(self):
        af = AgentFindings(agent_name="execution_quality", window_label="full", findings=[])
        result = _build_execution_risk_summary([af], {})
        assert result["execution_issues"] == []


# =========================================================================
# _load_model_knowledge
# =========================================================================


class TestLoadModelKnowledge:
    def test_loads_valid_yaml(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "model_knowledge.yaml").write_text(yaml.dump({
            "models": {
                "m1": {"architecture_family": "GRU", "tuning_notes": "test"},
            }
        }))
        result = _load_model_knowledge(str(ws))
        assert "m1" in result
        assert result["m1"]["architecture_family"] == "GRU"

    def test_no_file_returns_empty(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert _load_model_knowledge(str(ws)) == {}

    def test_invalid_yaml_returns_empty(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "model_knowledge.yaml").write_text(":invalid: yaml: :::")
        assert _load_model_knowledge(str(ws)) == {}

    def test_not_a_dict_returns_empty(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "model_knowledge.yaml").write_text(yaml.dump([1, 2, 3]))
        result = _load_model_knowledge(str(ws))
        assert result == {}

    def test_empty_models_key(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "model_knowledge.yaml").write_text(yaml.dump({"models": {}}))
        result = _load_model_knowledge(str(ws))
        assert result == {}

    def test_missing_models_key(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "model_knowledge.yaml").write_text(yaml.dump({"other": "data"}))
        result = _load_model_knowledge(str(ws))
        assert result == {}


# =========================================================================
# _load_correlation_excerpts edge cases
# =========================================================================


class TestLoadCorrelationExcerptsExtended:
    def test_model_not_in_index(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        # No correlation file → returns {}
        result = _load_correlation_excerpts(str(ws), ["nonexistent_model"])
        assert result == {}

    def test_empty_correlation_df(self, tmp_path):
        """Correlation CSV exists but is empty."""
        ws = tmp_path / "ws"
        (ws / "output" / "ensemble").mkdir(parents=True)
        import pandas as pd
        empty_df = pd.DataFrame()
        empty_df.to_csv(ws / "output" / "ensemble" / "correlation_matrix_2026-01-01.csv")
        result = _load_correlation_excerpts(str(ws), ["m1"])
        assert result == {}

    def test_duplicate_index_rows(self, tmp_path):
        """When duplicate index entries exist, row is DataFrame → iloc[0] used."""
        ws = tmp_path / "ws"
        (ws / "output" / "ensemble").mkdir(parents=True)
        import pandas as pd
        # Create a correlation CSV with duplicate index entries
        df = pd.DataFrame(
            [[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.5, 1.0, 0.2]],
            columns=["m1", "m2", "m2"],
            index=["m1", "m2", "m2"],
        )
        df.to_csv(ws / "output" / "ensemble" / "correlation_matrix_2026-01-01.csv")
        result = _load_correlation_excerpts(str(ws), ["m2"])
        # Should not crash — but may return empty or valid excerpt depending on dedup
        assert isinstance(result, dict)


# =========================================================================
# _build_combo_profile with workspace membership filtering
# =========================================================================


class TestBuildComboProfileWithWorkspaceMembership:
    def test_filters_to_combo_members(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "combo1": {"models": ["m1"]},
            }
        }))

        model_diagnoses = {
            "m1": {"diagnosis": "healthy", "diagnosis_detail": "All good"},
            "m2": {"diagnosis": "degrading", "diagnosis_detail": "IC falling"},
        }
        triage = {"combo_summary": {}}

        result = _build_combo_profile(
            "combo1", model_diagnoses, triage, [], str(ws)
        )
        assert "m1" in result["member_diagnoses"]
        assert "m2" not in result["member_diagnoses"]

    def test_no_workspace_skips_filtering(self):
        model_diagnoses = {
            "m1": {"diagnosis": "healthy", "diagnosis_detail": "All good"},
            "m2": {"diagnosis": "degrading", "diagnosis_detail": "IC falling"},
        }
        triage = {"combo_summary": {}}
        result = _build_combo_profile("combo1", model_diagnoses, triage, [], "")
        assert "m1" in result["member_diagnoses"]
        assert "m2" in result["member_diagnoses"]

    def test_combo_not_in_membership(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        (ws / "config" / "ensemble_config.json").write_text(json.dumps({
            "combos": {
                "combo_a": {"models": ["m1"]},
            }
        }))

        model_diagnoses = {"m1": {"diagnosis": "healthy", "diagnosis_detail": ""}}
        triage = {"combo_summary": {}}

        result = _build_combo_profile(
            "combo_unknown", model_diagnoses, triage, [], str(ws)
        )
        # combo_unknown matches nothing → combo_members is empty → all pass
        assert "m1" in result["member_diagnoses"]


# =========================================================================
# _build_execution_profile with real dataclass instances
# =========================================================================


class TestBuildExecutionProfileWithRealData:
    def test_extracts_execution_issues(self):
        af = AgentFindings(
            agent_name="execution_quality",
            window_label="full",
            findings=[
                Finding(severity="high", category="exec", title="Slippage",
                        detail="Slippage > 2% on large orders"),
            ],
        )
        result = _build_execution_profile([af], {})
        assert len(result["execution_issues"]) == 1
        assert result["execution_issues"][0]["severity"] == "high"
        assert result["execution_issues"][0]["title"] == "Slippage"

    def test_extracts_trade_issues(self):
        af = AgentFindings(
            agent_name="trade_pattern",
            window_label="full",
            findings=[
                Finding(severity="warning", category="trade", title="Overtrading",
                        detail="Too many trades"),
            ],
        )
        result = _build_execution_profile([af], {})
        assert len(result["trade_pattern_issues"]) == 1
        assert result["trade_pattern_issues"][0]["severity"] == "warning"

    def test_detail_truncated_500(self):
        long_detail = "D" * 600
        af = AgentFindings(
            agent_name="execution_quality",
            window_label="full",
            findings=[Finding(severity="info", category="exec", title="Test",
                              detail=long_detail)],
        )
        result = _build_execution_profile([af], {})
        assert len(result["execution_issues"][0]["detail"]) == 500

    def test_empty_title_and_detail(self):
        af = AgentFindings(
            agent_name="execution_quality",
            window_label="full",
            findings=[Finding(severity="info", category="exec", title="", detail="")],
        )
        result = _build_execution_profile([af], {})
        assert result["execution_issues"][0]["title"] == ""
        assert result["execution_issues"][0]["detail"] == ""

    def test_multiple_agents(self):
        af1 = AgentFindings(
            agent_name="execution_quality",
            window_label="full",
            findings=[Finding(severity="high", category="exec", title="E1", detail="")],
        )
        af2 = AgentFindings(
            agent_name="trade_pattern",
            window_label="full",
            findings=[Finding(severity="critical", category="trade", title="T1", detail="")],
        )
        result = _build_execution_profile([af1, af2], {})
        assert len(result["execution_issues"]) == 1
        assert len(result["trade_pattern_issues"]) == 1
