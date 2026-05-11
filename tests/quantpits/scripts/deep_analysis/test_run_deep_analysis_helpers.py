"""Tests for run_deep_analysis helper functions (data loaders + profile builders)."""

import json
import os
import pytest
import yaml

os.environ["QLIB_WORKSPACE_DIR"] = "/tmp"

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
        _persist_feedback_report,
    )


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
        assert profile["training_history"] == []
        assert profile["signals"] == []
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
