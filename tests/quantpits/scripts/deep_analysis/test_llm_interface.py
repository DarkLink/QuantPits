"""Tests for LLM Interface (Phase 3 extended)."""

import sys
import json
import os
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
