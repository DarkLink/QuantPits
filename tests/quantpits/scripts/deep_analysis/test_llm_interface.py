"""Tests for LLM Interface (Phase 3 extended)."""

import sys
import json
import os
import pytest
from unittest.mock import MagicMock, patch

# Mock openai module before importing LLMInterface
mock_openai = MagicMock()
sys.modules["openai"] = mock_openai

from quantpits.scripts.deep_analysis.llm_interface import LLMInterface
from quantpits.scripts.deep_analysis.signal_extractor import Signal


# ------------------------------------------------------------------
# Existing summary mode tests (preserved)
# ------------------------------------------------------------------

def test_llm_interface_template_fallback():
    # Test fallback when no API key
    interface = LLMInterface(api_key=None)
    interface._client = None 
    
    with patch.dict('os.environ', {}, clear=True):
        interface.api_key = None
        assert interface.is_available() is False
    
    # Test with missing data to hit branches in _template_summary
    synthesis_result = {
        'health_status': 'Healthy',
        'executive_summary_data': {
            'windows_analyzed': ['2026-01-01'],
            'agents_run': ['AgentA'],
            'critical_count': 0,
            'warning_count': 1,
            'positive_count': 2,
        },
        'recommendations': [
            {'priority': 'P0', 'text': 'Do something', 'source': 'AgentA'}
        ],
        'cross_findings': [
             MagicMock(severity='critical', title='Major Issue'),
             MagicMock(severity='warning', title='Minor Issue'),
             MagicMock(severity='info', title='Info Issue')
        ]
    }
    
    summary = interface.generate_executive_summary(synthesis_result)
    assert "**System Health:** Healthy" in summary
    assert "Do something" in summary
    assert "🔴 Major Issue" in summary
    assert "🟡 Minor Issue" in summary
    assert "🟢 Info Issue" in summary

def test_llm_interface_openai():
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "LLM Generated Summary"
    mock_client.chat.completions.create.return_value = mock_response
    
    interface = LLMInterface(api_key="fake_key", base_url="http://fake.url")
    interface._get_client()
    
    assert interface.is_available() is True
    
    synthesis_result = {
        'health_status': 'Healthy',
        'executive_summary_data': {
            'windows_analyzed': ['2026-01-01'],
            'agents_run': ['AgentA'],
            'critical_count': 0,
            'warning_count': 0,
            'positive_count': 0,
            'market_regime': 'Bull',
            'cagr_1y': 0.1,
            'sharpe_1y': 1.0
        },
        'cross_findings': [
            MagicMock(severity='critical', title='Major Issue', detail='Detail here')
        ],
        'recommendations': [
            {'priority': 'P0', 'text': 'Rec 1', 'source': 'AgentA'}
        ],
        'change_impact': [
            {'event': {'type': 'retrain', 'date': '2026-01-15', 'model': 'M1'}}
        ],
        'external_notes': 'Some notes'
    }
    
    summary = interface.generate_executive_summary(synthesis_result)
    assert summary == "LLM Generated Summary"
    mock_client.chat.completions.create.assert_called_once()

def test_llm_interface_error_fallback():
    interface = LLMInterface(api_key="fake_key")
    mock_client = MagicMock()
    interface._client = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    
    synthesis_result = {'health_status': 'Error State'}
    
    summary = interface.generate_executive_summary(synthesis_result)
    assert "**System Health:** Error State" in summary

def test_llm_interface_import_error():
    # Test ImportError for openai
    interface = LLMInterface(api_key="fake_key")
    interface._client = None
    
    with patch.dict(sys.modules, {'openai': None}):
        import pytest
        with pytest.raises(RuntimeError, match="openai package not installed"):
            interface._get_client()


# ------------------------------------------------------------------
# Summary prompt loading from workspace
# ------------------------------------------------------------------

class TestSummaryPromptLoading:
    def test_loads_from_workspace_file(self, tmp_path):
        ws = tmp_path / "ws"
        skills_dir = ws / "config" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "summary_system.md").write_text("Custom summary prompt")

        interface = LLMInterface()
        prompt = interface._load_summary_prompt(str(ws))
        assert prompt == "Custom summary prompt"

    def test_falls_back_to_default_when_missing(self, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)

        interface = LLMInterface()
        prompt = interface._load_summary_prompt(ws)
        assert "quantitative finance analyst" in prompt

    def test_falls_back_when_no_workspace(self):
        interface = LLMInterface()
        prompt = interface._load_summary_prompt(None)
        assert "quantitative finance analyst" in prompt


# ------------------------------------------------------------------
# Skills loading
# ------------------------------------------------------------------

class TestSkillsLoading:
    def test_loads_critic_system_and_scope_skills(self, tmp_path):
        ws = tmp_path / "ws"
        skills_dir = ws / "config" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "critic_system.md").write_text("CRITIC SYSTEM")
        (skills_dir / "hyperparam_tuning.md").write_text("HYPERPARAM SKILL")
        (skills_dir / "model_selection.md").write_text("MODEL SELECTION SKILL")

        interface = LLMInterface()
        prompt = interface._load_skills(str(ws), ["hyperparams", "model_selection"])
        assert "CRITIC SYSTEM" in prompt
        assert "HYPERPARAM SKILL" in prompt
        assert "MODEL SELECTION SKILL" in prompt

    def test_uses_default_when_critic_system_missing(self, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(os.path.join(ws, "config", "skills"), exist_ok=True)

        interface = LLMInterface()
        prompt = interface._load_skills(ws, ["hyperparams"])
        assert "quantitative strategy optimization" in prompt.lower()

    def test_skips_missing_scope_skills(self, tmp_path):
        ws = tmp_path / "ws"
        skills_dir = ws / "config" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "critic_system.md").write_text("CRITIC SYSTEM")

        interface = LLMInterface()
        # combo_search skill file doesn't exist — should not error
        prompt = interface._load_skills(str(ws), ["combo_search"])
        assert "CRITIC SYSTEM" in prompt


# ------------------------------------------------------------------
# Workspace LLM config loading
# ------------------------------------------------------------------

class TestWorkspaceLLMConfig:
    def test_loads_config(self, tmp_path):
        ws = tmp_path / "ws"
        config_dir = ws / "config"
        config_dir.mkdir(parents=True)
        with open(config_dir / "llm_config.json", "w") as f:
            json.dump({"critic_model": "deepseek-v4-pro", "temperature": 0.2}, f)

        interface = LLMInterface()
        cfg = interface._load_workspace_llm_config(str(ws))
        assert cfg["critic_model"] == "deepseek-v4-pro"

    def test_returns_empty_when_missing(self, tmp_path):
        interface = LLMInterface()
        cfg = interface._load_workspace_llm_config(str(tmp_path))
        assert cfg == {}


# ------------------------------------------------------------------
# Critic mode (generate_action_items)
# ------------------------------------------------------------------

class TestCriticMode:
    def test_empty_signals_returns_empty(self, tmp_path):
        interface = LLMInterface(api_key="fake")
        items = interface.generate_action_items([], str(tmp_path))
        assert items == []

    def test_no_api_key_returns_empty(self, tmp_path):
        interface = LLMInterface(api_key=None)
        with patch.dict('os.environ', {}, clear=True):
            interface.api_key = None
            signals = [Signal(
                signal_type="underfitting", severity="warning", scope="hyperparams",
                source_agent="Model Health", target="gru", context="test",
            )]
            items = interface.generate_action_items(signals, str(tmp_path))
            assert items == []

    def test_critic_llm_call_success(self, tmp_path):
        ws = tmp_path / "ws"
        config_dir = ws / "config"
        config_dir.mkdir(parents=True)
        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "critic_system.md").write_text("SYSTEM PROMPT")

        with open(config_dir / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)
        with open(config_dir / "llm_config.json", "w") as f:
            json.dump({"critic_model": "test-model", "base_url": "http://test"}, f)

        # Mock the openai API
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([{
            "action_type": "adjust_hyperparam",
            "scope": "hyperparams",
            "target": "gru_Alpha158",
            "params": {"n_epochs": {"from": 100, "to": 150}},
            "reason": "underfitting detected",
            "source_signals": ["underfitting"],
            "confidence": 0.8,
            "risk_level": "low",
        }])
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake_key")
        signals = [Signal(
            signal_type="underfitting", severity="warning", scope="hyperparams",
            source_agent="Model Health", target="gru_Alpha158", context="test",
        )]
        items = interface.generate_action_items(signals, str(ws))
        assert len(items) == 1
        assert items[0].action_type == "adjust_hyperparam"
        assert items[0].target == "gru_Alpha158"

    def test_parse_action_items_with_code_fence(self):
        content = '```json\n[{"action_type": "test", "target": "m1"}]\n```'
        items = LLMInterface._parse_action_items(content)
        assert len(items) == 1
        assert items[0].action_type == "test"

    def test_parse_action_items_single_object(self):
        content = '{"action_type": "test", "target": "m1"}'
        items = LLMInterface._parse_action_items(content)
        assert len(items) == 1

    def test_critic_prompt_contains_signals_and_scopes(self):
        interface = LLMInterface()
        signals = [Signal(
            signal_type="ic_decay", severity="warning", scope="hyperparams",
            source_agent="Model Health", target="gru", context="test",
        )]
        prompt = interface._build_critic_prompt(signals, ["hyperparams"])
        assert "ic_decay" in prompt
        assert "hyperparams" in prompt
        assert "JSON array" in prompt

    def test_critic_llm_json_parse_error_retry(self, tmp_path):
        """LLM returns invalid JSON twice → empty list."""
        ws = tmp_path / "ws"
        config_dir = ws / "config"
        config_dir.mkdir(parents=True)

        with open(config_dir / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)

        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        # First call: invalid JSON (parse will fail)
        mock_response_bad = MagicMock()
        mock_response_bad.choices = [MagicMock()]
        mock_response_bad.choices[0].message.content = "Not valid JSON at all"

        # Second call: also invalid
        mock_client.chat.completions.create.return_value = mock_response_bad

        interface = LLMInterface(api_key="fake_key")
        signals = [Signal(
            signal_type="underfitting", severity="warning", scope="hyperparams",
            source_agent="Test", target="m1", context="test",
        )]
        items = interface.generate_action_items(signals, str(ws))
        assert items == []


class TestLoadCurrentParams:
    """Tests for _load_current_params — reads YAML hyperparams for signals."""

    def test_extracts_params_for_signal_models(self, tmp_path):
        """Should return current hyperparams for each model in signals."""
        ws = tmp_path / "TestWorkspace"
        ws.mkdir()
        config_dir = ws / "config"
        config_dir.mkdir()

        # model_registry.yaml
        import yaml as _yaml
        registry = {
            "models": {
                "gru_Alpha158": {
                    "algorithm": "gru", "dataset": "Alpha158",
                    "yaml_file": "config/workflow_config_gru_Alpha158.yaml",
                },
                "linear_Alpha158": {
                    "algorithm": "linear", "dataset": "Alpha158",
                    "yaml_file": "config/workflow_config_linear_Alpha158.yaml",
                },
            },
        }
        with open(config_dir / "model_registry.yaml", "w") as f:
            _yaml.dump(registry, f)

        # Workflow YAML for gru_Alpha158
        gru_config = {
            "task": {
                "model": {
                    "class": "GRU",
                    "kwargs": {"n_epochs": 200, "dropout": 0.0, "lr": 0.001},
                },
            },
        }
        with open(config_dir / "workflow_config_gru_Alpha158.yaml", "w") as f:
            _yaml.dump(gru_config, f)

        # Workflow YAML for linear_Alpha158 (no kwargs)
        lin_config = {
            "task": {"model": {"class": "LinearModel", "kwargs": {"estimator": "ols"}}},
        }
        with open(config_dir / "workflow_config_linear_Alpha158.yaml", "w") as f:
            _yaml.dump(lin_config, f)

        interface = LLMInterface(api_key="fake_key")
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Test", "gru_Alpha158", context="test"),
            Signal("overfitting", "warning", "hyperparams", "Test", "linear_Alpha158", context="test"),
        ]
        result = interface._load_current_params(str(ws), signals)

        assert "gru_Alpha158" in result
        assert result["gru_Alpha158"]["dropout"] == 0.0
        assert result["gru_Alpha158"]["n_epochs"] == 200
        assert "linear_Alpha158" in result
        assert result["linear_Alpha158"]["estimator"] == "ols"
        assert "dropout" not in result["linear_Alpha158"]

    def test_ignores_non_hyperparam_signals(self, tmp_path):
        """Should skip signals that are not in hyperparams scope."""
        ws = tmp_path / "TestWorkspace"
        ws.mkdir()
        (ws / "config").mkdir()

        interface = LLMInterface(api_key="fake_key")
        signals = [
            Signal("negative_contribution", "warning", "model_selection", "Test", "m1", context="test"),
        ]
        result = interface._load_current_params(str(ws), signals)
        assert result == {}

    def test_empty_signals(self, tmp_path):
        """Should handle empty signal list."""
        ws = tmp_path / "TestWorkspace"
        ws.mkdir()
        interface = LLMInterface(api_key="fake_key")
        assert interface._load_current_params(str(ws), []) == {}

    def test_model_not_in_registry(self, tmp_path):
        """Should skip models not found in model_registry.yaml."""
        ws = tmp_path / "TestWorkspace"
        ws.mkdir()
        (ws / "config").mkdir()

        interface = LLMInterface(api_key="fake_key")
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Test", "ghost_model", context="test"),
        ]
        result = interface._load_current_params(str(ws), signals)
        assert result == {}


# ===================================================================
# Coverage gap tests — Tier 1: historical override + error paths
# ===================================================================

def _make_workspace_with_skills(tmp_path, skills=None):
    """Create a workspace with config/skills/ directory and given skill files."""
    ws = tmp_path / "TestWorkspace"
    ws.mkdir()
    (ws / "config").mkdir(parents=True)
    skills_dir = ws / "config" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    if skills:
        for name, content in skills.items():
            (skills_dir / name).write_text(content)
    # Default triage and critic skills
    (skills_dir / "triage_system.md").write_text("You are a triage specialist.")
    (skills_dir / "critic_system.md").write_text("You are a critic.")
    return str(ws)


def _make_signal(signal_type="underfitting", severity="warning", scope="hyperparams",
                 target="m1", source="TestAgent", context="test context"):
    return Signal(signal_type, severity, scope, source, target, context=context)


# -------------------------------------------------------------------
# generate_triage — historical flags and override
# -------------------------------------------------------------------

class TestGenerateTriage:
    def test_no_api_key_returns_none(self, tmp_path):
        """Line 1361-1363: no API key → return None."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key=None)
        result = interface.generate_triage(
            triage_input={"model_ranking": [], "family_stats": {}, "combo_summary": {}, "market_context": {}},
            signals=[],
            workspace_root=ws,
        )
        assert result is None

    def test_no_triage_skill_returns_none(self, tmp_path):
        """Line 1366-1368: triage_system.md not found → return None."""
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        (ws_path / "config").mkdir()
        (ws_path / "config" / "skills").mkdir()
        # No triage_system.md

        interface = LLMInterface(api_key="fake_key")
        result = interface.generate_triage(
            triage_input={"model_ranking": [], "family_stats": {}, "combo_summary": {}, "market_context": {}},
            signals=[],
            workspace_root=str(ws_path),
        )
        assert result is None

    def test_historical_flags_added_to_model_table(self, tmp_path):
        """Line 1389: model entries with historical_flags get the field."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        with patch.object(interface, '_call_llm_json_object', return_value={
            "prioritized_models": [],
            "healthy_models": [],
            "prioritized_combos": [],
            "needs_execution_risk": False,
            "systemic_observations": [],
        }):
            triage_input = {
                "model_ranking": [
                    {"model": "m1", "ic_mean": 0.05, "icir_mean": 0.5,
                     "ic_trend": "stable", "family": "gru",
                     "in_combos": [], "best_epoch": 200, "actual_epochs": 200,
                     "historical_flags": {"run_count": 2, "historical_signal_types": ["overfitting"]}},
                    {"model": "m2", "ic_mean": 0.08, "icir_mean": 0.8,
                     "ic_trend": "improving", "family": "mlp",
                     "in_combos": ["combo_a"], "best_epoch": 100, "actual_epochs": 100},
                ],
                "family_stats": {},
                "combo_summary": {},
                "market_context": {},
            }
            result = interface.generate_triage(
                triage_input=triage_input,
                signals=[],
                workspace_root=ws,
            )
            assert result is not None

    def test_historical_override_forces_flagged_models_into_prioritized(self, tmp_path):
        """Lines 1453-1483: LLM ignores flagged model → enforced by override."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        # LLM returns m2, m3 only — m1 (historically flagged) NOT in any list
        with patch.object(interface, '_call_llm_json_object', return_value={
            "prioritized_models": [
                {"target": "m2", "priority_score": 8, "primary_signal": "ic_decay"},
                {"target": "m3", "priority_score": 6, "primary_signal": "underfitting"},
            ],
            "healthy_models": [],  # m1 is NOT in healthy either
            "prioritized_combos": [],
            "needs_execution_risk": False,
            "systemic_observations": [],
        }):
            triage_input = {
                "model_ranking": [
                    {"model": "m1", "ic_mean": 0.03, "icir_mean": 0.3, "ic_trend": "declining",
                     "family": "gru", "in_combos": [], "best_epoch": 50, "actual_epochs": 200,
                     "early_stopped": True,
                     "historical_flags": {"run_count": 3, "historical_signal_types": ["overfitting", "ic_decay"]}},
                    {"model": "m2", "ic_mean": 0.08, "icir_mean": 0.8, "ic_trend": "improving",
                     "family": "mlp", "in_combos": ["c1"]},
                    {"model": "m3", "ic_mean": 0.04, "icir_mean": 0.4, "ic_trend": "stable",
                     "family": "lgb", "in_combos": []},
                ],
                "family_stats": {},
                "combo_summary": {},
                "market_context": {},
            }
            result = interface.generate_triage(
                triage_input=triage_input,
                signals=[],
                workspace_root=ws,
            )
            assert result is not None
            # m1 MUST be in prioritized_models despite LLM ignoring it
            models = [p.get("target", p.get("model", "")) for p in result.get("prioritized_models", [])]
            assert "m1" in models, "Historically flagged model must be forced into prioritized_models"
            # Check the override entry
            m1_entry = next(p for p in result["prioritized_models"] if p.get("target") == "m1")
            assert m1_entry["tracking_mode"] is True
            assert m1_entry["priority_score"] == 7
            assert "historical_tracking" in m1_entry["primary_signal"]

    def test_historical_override_skips_when_already_present(self, tmp_path):
        """Line 1463: missing set is empty → no override needed."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        # LLM already includes the flagged model
        with patch.object(interface, '_call_llm_json_object', return_value={
            "prioritized_models": [
                {"target": "m1", "priority_score": 8, "primary_signal": "overfitting"},
            ],
            "healthy_models": [],
            "prioritized_combos": [],
            "needs_execution_risk": False,
            "systemic_observations": [],
        }):
            triage_input = {
                "model_ranking": [
                    {"model": "m1", "ic_mean": 0.03, "icir_mean": 0.3, "ic_trend": "declining",
                     "family": "gru", "in_combos": [],
                     "historical_flags": {"run_count": 2, "historical_signal_types": ["overfitting"]}},
                ],
                "family_stats": {},
                "combo_summary": {},
                "market_context": {},
            }
            result = interface.generate_triage(
                triage_input=triage_input, signals=[], workspace_root=ws,
            )
            assert result is not None
            # Only one entry (no duplicate)
            assert len(result["prioritized_models"]) == 1

    def test_historical_override_no_flagged_models(self, tmp_path):
        """Line 1452: no flagged_models → override block skipped entirely."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        with patch.object(interface, '_call_llm_json_object', return_value={
            "prioritized_models": [{"target": "m1", "priority_score": 5}],
            "healthy_models": [],
            "prioritized_combos": [],
            "needs_execution_risk": False,
            "systemic_observations": [],
        }):
            triage_input = {
                "model_ranking": [
                    {"model": "m1", "ic_mean": 0.08, "icir_mean": 0.8, "ic_trend": "improving",
                     "family": "mlp", "in_combos": []},  # No historical_flags
                ],
                "family_stats": {},
                "combo_summary": {},
                "market_context": {},
            }
            result = interface.generate_triage(
                triage_input=triage_input, signals=[], workspace_root=ws,
            )
            assert result is not None

    def test_history_summary_building(self, tmp_path):
        """Line 1405: history_summary built from loaded history."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        with patch.object(interface, '_load_recent_action_history', return_value=[
            {"_run_date": "2026-05-25", "target": "m1", "action_type": "adjust_hyperparam",
             "params": {"n_epochs": {"from": 100, "to": 150}}},
            {"run_date": "2026-05-20", "target": "m2", "action_type": "disable_model",
             "params": {}},
        ]):
            with patch.object(interface, '_call_llm_json_object', return_value={
                "prioritized_models": [], "healthy_models": [],
                "prioritized_combos": [], "needs_execution_risk": False,
                "systemic_observations": [],
            }):
                triage_input = {
                    "model_ranking": [{"model": "m1", "ic_mean": 0.05}],
                    "family_stats": {}, "combo_summary": {}, "market_context": {},
                }
                result = interface.generate_triage(
                    triage_input=triage_input, signals=[], workspace_root=ws,
                )
                assert result is not None


# -------------------------------------------------------------------
# _run_triage — error paths
# -------------------------------------------------------------------

class TestRunTriage:
    def test_all_signals_out_of_scope(self, tmp_path):
        """Line 788-793: all signals out of active_scopes → return all signals."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        s = _make_signal(scope="combo_search")  # Not in active_scopes
        result = interface._run_triage(
            signals=[s],
            triage_skill="You are a triage specialist.",
            recent_history=[],
            active_scopes=["hyperparams", "model_selection"],
            current_params={},
            combo_membership={},
            model="gpt-4",
            temperature=0.3,
            api_key="fake_key",
            base_url=None,
            workspace_root=ws,
        )
        assert result == [s]

    def test_triage_llm_call_failure(self, tmp_path):
        """Lines 862-864: Triage LLM call raises → return None."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        s = _make_signal(scope="hyperparams")
        with patch.object(interface, '_call_triage_llm', side_effect=Exception("Connection refused")):
            result = interface._run_triage(
                signals=[s],
                triage_skill="You are a triage specialist.",
                recent_history=[],
                active_scopes=["hyperparams"],
                current_params={"m1": {"n_epochs": 100}},
                combo_membership={},
                model="gpt-4",
                temperature=0.3,
                api_key="fake_key",
                base_url=None,
                workspace_root=ws,
            )
            assert result is None

    def test_triage_returns_none(self, tmp_path):
        """Line 866-867: _call_triage_llm returns None → return None."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        s = _make_signal(scope="hyperparams")
        with patch.object(interface, '_call_triage_llm', return_value=None):
            result = interface._run_triage(
                signals=[s],
                triage_skill="You are a triage specialist.",
                recent_history=[],
                active_scopes=["hyperparams"],
                current_params={"m1": {"n_epochs": 100}},
                combo_membership={},
                model="gpt-4",
                temperature=0.3,
                api_key="fake_key",
                base_url=None,
                workspace_root=ws,
            )
            assert result is None

    def test_global_defaults_detection(self, tmp_path):
        """Lines 834-837: global defaults detection across models."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        signals = [_make_signal(target=f"m{i}", scope="hyperparams") for i in range(1, 6)]
        current_params = {}
        for i in range(1, 5):
            current_params[f"m{i}"] = {"n_epochs": 200, "lr": 0.001}
        current_params["m5"] = {"n_epochs": 200, "lr": 0.002}

        with patch.object(interface, '_call_triage_llm', return_value={
            "prioritized_targets": [{"target": "m1", "priority_score": 5, "primary_signal": "test"}],
            "excluded_targets": [],
            "systemic_observations": [],
        }):
            result = interface._run_triage(
                signals=signals, triage_skill="You are a triage specialist.",
                recent_history=[], active_scopes=["hyperparams"],
                current_params=current_params, combo_membership={},
                model="gpt-4", temperature=0.3, api_key="fake_key",
                base_url=None, workspace_root=ws,
            )
            assert result is not None

    def test_oos_counts_tracking(self, tmp_path):
        """Line 818: out-of-scope signal type counting."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        in_scope = _make_signal(scope="hyperparams", signal_type="overfitting", target="m1")
        out_scope = _make_signal(scope="combo_search", signal_type="ic_decay", target="c1")

        with patch.object(interface, '_call_triage_llm', return_value={
            "prioritized_targets": [{"target": "m1", "priority_score": 5, "primary_signal": "test"}],
            "excluded_targets": [],
            "systemic_observations": [],
        }):
            result = interface._run_triage(
                signals=[in_scope, out_scope], triage_skill="You are a triage specialist.",
                recent_history=[], active_scopes=["hyperparams"],
                current_params={"m1": {"n_epochs": 100}}, combo_membership={},
                model="gpt-4", temperature=0.3, api_key="fake_key",
                base_url=None, workspace_root=ws,
            )
            assert result is not None


# -------------------------------------------------------------------
# generate_action_items — error paths
# -------------------------------------------------------------------

class TestGenerateActionItemsGaps:
    def test_critic_llm_failure_returns_empty(self, tmp_path):
        """Lines 493-496: Critic LLM call raises exception → return []."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        s = _make_signal()
        with patch.object(interface, '_call_critic_llm', side_effect=Exception("API timeout")):
            result = interface.generate_action_items(
                signals=[s], workspace_root=ws,
            )
            assert result == []

    def test_no_signals_returns_empty(self, tmp_path):
        """Lines 400-402: empty signals → return []."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")
        result = interface.generate_action_items(signals=[], workspace_root=ws)
        assert result == []

    def test_critic_returns_zero_items(self, tmp_path):
        """Line 1904: Critic parsed successfully but returned 0 ActionItems."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        s = _make_signal()
        with patch.object(interface, '_call_critic_llm', return_value=[]):
            result = interface.generate_action_items(
                signals=[s], workspace_root=ws,
            )
            assert result == []


# -------------------------------------------------------------------
# _load_skill — exception handling
# -------------------------------------------------------------------

class TestLoadSkill:
    def test_exception_swallowed(self, tmp_path):
        """Line 1343-1344: exception during skill file read → return None."""
        ws = tmp_path / "ws"
        ws.mkdir()
        skills_dir = ws / "config" / "skills"
        skills_dir.mkdir(parents=True)
        # Create a directory instead of a file to cause IsADirectoryError
        (skills_dir / "bad_skill.md").mkdir()

        interface = LLMInterface(api_key="fake_key")
        result = interface._load_skill(str(ws), "bad_skill.md")
        assert result is None


# -------------------------------------------------------------------
# _llm_summary — error paths
# -------------------------------------------------------------------

class TestLlmSummaryGaps:
    def test_no_api_key_raises_runtime_error(self, tmp_path):
        """Line 202: no API key → RuntimeError."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key=None)
        with pytest.raises(RuntimeError, match="No API key available"):
            interface._llm_summary({}, str(ws))

    def test_import_error_for_openai(self, tmp_path):
        """Lines 211-212: ImportError when openai not available."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")
        ws_config = {"base_url": "http://custom:8000/v1"}
        with patch.object(interface, '_load_workspace_llm_config', return_value=ws_config):
            with patch.dict('sys.modules', {'openai': None}):
                with pytest.raises(RuntimeError, match="openai package not installed"):
                    interface._llm_summary({}, str(ws))


# -------------------------------------------------------------------
# _load_tunable_param_names — exception
# -------------------------------------------------------------------

class TestLoadTunableParamNames:
    def test_invalid_json_returns_none(self, tmp_path):
        """Line 699-701: JSON parse failure → pass → return None."""
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "hyperparam_bounds.json").write_text("INVALID JSON")

        result = LLMInterface._load_tunable_param_names(str(ws))
        assert result is None

    def test_no_workspace_returns_none(self):
        """Line 688-689: no workspace_root → None."""
        result = LLMInterface._load_tunable_param_names(None)
        assert result is None

    def test_file_not_found_returns_none(self, tmp_path):
        """Line 691-692: file not found → None."""
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        result = LLMInterface._load_tunable_param_names(str(ws))
        assert result is None

    def test_empty_bounds_returns_none(self, tmp_path):
        """Line 697-698: empty bounds → None."""
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "config").mkdir()
        (ws / "config" / "hyperparam_bounds.json").write_text(json.dumps({"bounds": {}}))
        result = LLMInterface._load_tunable_param_names(str(ws))
        assert result is None


# -------------------------------------------------------------------
# _compute_available_interventions — bounds_path branch
# -------------------------------------------------------------------

class TestComputeAvailableInterventions:
    def test_with_bounds_params(self, tmp_path):
        """Line 746: bounds_params is not None → intersection path."""
        ws = _make_workspace_with_skills(tmp_path)
        interface = LLMInterface(api_key="fake_key")

        s = _make_signal(target="m1", scope="hyperparams")
        result = interface._compute_available_interventions(
            in_scope_signals=[s],
            recent_history=[],
            current_params={"m1": {"n_epochs": 200, "lr": 0.001}},
            workspace_root=ws,
        )
        assert "m1" in result
        assert "untouched" in result["m1"]
        assert "exhausted" in result["m1"]


# -------------------------------------------------------------------
# _call_triage_llm — JSON error paths
# -------------------------------------------------------------------

class TestCallTriageLlmGaps:
    def test_reasoning_fallback_json_decode_error(self, tmp_path):
        """Lines 1086-1087: reasoning_content fallback JSONDecodeError → pass."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message.content = ""
        mock_choice.message.reasoning_content = "not valid json {{{"
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake_key")
        with patch.object(interface, '_get_or_create_client', return_value=mock_client):
            result = interface._call_triage_llm(
                system_prompt="You are helpful.",
                user_prompt="{}",
                model="gpt-4",
                temperature=0.3,
                api_key="fake_key",
                base_url=None,
            )
            assert result is None

    def test_empty_content_no_reasoning(self, tmp_path):
        """Empty content with no reasoning → return None."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = ""
        mock_choice.message.reasoning_content = None
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake_key")
        with patch.object(interface, '_get_or_create_client', return_value=mock_client):
            result = interface._call_triage_llm(
                system_prompt="You are helpful.",
                user_prompt="{}",
                model="gpt-4",
                temperature=0.3,
                api_key="fake_key",
                base_url=None,
            )
            assert result is None
