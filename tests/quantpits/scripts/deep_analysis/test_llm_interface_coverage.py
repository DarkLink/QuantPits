"""Coverage expansion tests for LLMInterface — uncovered functions and paths."""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure openai module is mocked — but don't overwrite an existing mock
# (other test files may have set up their own module-level mock first)
if "openai" not in sys.modules:
    sys.modules["openai"] = MagicMock()

from quantpits.scripts.deep_analysis.llm_interface import LLMInterface
from quantpits.scripts.deep_analysis.signal_extractor import Signal


# ------------------------------------------------------------------
# _assess_convergence
# ------------------------------------------------------------------

class TestAssessConvergence:
    def test_no_convergence_data(self):
        """None convergence -> first round message."""
        result = LLMInterface._assess_convergence(None)
        assert "NO TRAINING DATA" in result
        assert "first round" in result

    def test_empty_dict(self):
        """Empty dict -> first round message."""
        result = LLMInterface._assess_convergence({})
        assert "NO TRAINING DATA" in result

    def test_severe_overfitting(self):
        """best_epoch <= 1, early_stopped=True, actual > 10."""
        result = LLMInterface._assess_convergence({
            "best_epoch": 1, "actual_epochs": 20, "early_stopped": True,
        })
        assert "SEVERE OVERFITTING" in result
        assert "MUST try regularization" in result

    def test_mild_overfitting(self):
        """best_epoch <= 3 but not severe."""
        result = LLMInterface._assess_convergence({
            "best_epoch": 2, "actual_epochs": 20, "early_stopped": False,
        })
        assert "MILD OVERFITTING" in result

    def test_healthy(self):
        """Normal convergence."""
        result = LLMInterface._assess_convergence({
            "best_epoch": 50, "actual_epochs": 200, "early_stopped": True,
        })
        assert "Healthy" in result
        assert "trustworthy" in result

    def test_best_epoch_none_but_has_actual(self):
        """best_epoch None but actual > 10 → falls through to Healthy (best_epoch is None, cond fails)."""
        result = LLMInterface._assess_convergence({
            "best_epoch": None, "actual_epochs": 200, "early_stopped": False,
        })
        # best_epoch is None, so "best_epoch <= 1" and "best_epoch <= 3" both fail
        # Falls to the default Healthy path
        assert "Healthy" in result

    def test_low_actual_epochs_no_overfit_judgment(self):
        """actual_epochs <= 10 should not trigger overfitting."""
        result = LLMInterface._assess_convergence({
            "best_epoch": 1, "actual_epochs": 5, "early_stopped": True,
        })
        assert "NO TRAINING DATA" not in result
        # actual_epochs <= 10 so overfitting check is skipped
        # But best_epoch is set, so goes to Healthy path
        assert "Healthy" in result


# ------------------------------------------------------------------
# _extract_json_object
# ------------------------------------------------------------------

class TestExtractJsonObject:
    def test_empty_text(self):
        assert LLMInterface._extract_json_object("") is None
        assert LLMInterface._extract_json_object(None) is None

    def test_code_fence_with_json_tag(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = LLMInterface._extract_json_object(text)
        assert result == '{"key": "value"}'

    def test_code_fence_without_json_tag(self):
        text = '```\n{"key": "value"}\n```'
        result = LLMInterface._extract_json_object(text)
        assert result == '{"key": "value"}'

    def test_code_fence_generic(self):
        text = '```\n{"a": 1}\n```'
        result = LLMInterface._extract_json_object(text)
        assert result == '{"a": 1}'

    def test_no_fences_pure_json(self):
        text = '{"a": 1, "b": [2, 3]}'
        result = LLMInterface._extract_json_object(text)
        assert result == '{"a": 1, "b": [2, 3]}'

    def test_no_fences_with_surrounding_text(self):
        text = 'Here is the result:\n{"a": 1}\nThat is all.'
        result = LLMInterface._extract_json_object(text)
        assert result == '{"a": 1}'

    def test_nested_braces(self):
        text = '{"a": {"b": {"c": 1}}, "d": [1, 2]}'
        result = LLMInterface._extract_json_object(text)
        assert result == '{"a": {"b": {"c": 1}}, "d": [1, 2]}'

    def test_no_braces_at_all(self):
        text = 'No JSON here, just text.'
        assert LLMInterface._extract_json_object(text) is None

    def test_unbalanced_braces(self):
        text = '{"a": 1'
        assert LLMInterface._extract_json_object(text) is None


# ------------------------------------------------------------------
# _parse_action_items
# ------------------------------------------------------------------

class TestParseActionItems:
    def test_json_array(self):
        content = json.dumps([{"action_type": "test", "target": "m1", "scope": "s1"}])
        items = LLMInterface._parse_action_items(content)
        assert len(items) == 1
        assert items[0].action_type == "test"
        assert items[0].target == "m1"

    def test_json_single_object(self):
        content = json.dumps({"action_type": "test", "target": "m1", "scope": "s1"})
        items = LLMInterface._parse_action_items(content)
        assert len(items) == 1
        assert items[0].target == "m1"

    def test_with_markdown_fence(self):
        content = '```json\n[{"action_type": "test", "target": "m1", "scope": "s1"}]\n```'
        items = LLMInterface._parse_action_items(content)
        assert len(items) == 1
        assert items[0].action_type == "test"

    def test_with_triple_backtick_no_lang(self):
        content = '```\n[{"action_type": "test", "target": "m1", "scope": "s1"}]\n```'
        items = LLMInterface._parse_action_items(content)
        assert len(items) == 1

    def test_without_fence_closing(self):
        content = '```json\n[{"action_type": "test", "target": "m1", "scope": "s1"}]'
        items = LLMInterface._parse_action_items(content)
        assert len(items) == 1

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            LLMInterface._parse_action_items("not json at all")


# ------------------------------------------------------------------
# _load_hyperparam_bounds
# ------------------------------------------------------------------

class TestLoadHyperparamBounds:
    def test_loads_bounds(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({
                "bounds": {
                    "n_epochs": {"min": 10, "max": 500},
                    "lr": {"min": 1e-5, "max": 1e-2},
                }
            }, f)

        interface = LLMInterface()
        bounds = interface._load_hyperparam_bounds(str(ws))
        assert bounds["n_epochs"]["min"] == 10
        assert bounds["n_epochs"]["max"] == 500
        assert bounds["lr"]["min"] == 1e-5

    def test_missing_file_returns_empty(self, tmp_path):
        interface = LLMInterface()
        bounds = interface._load_hyperparam_bounds(str(tmp_path))
        assert bounds == {}

    def test_invalid_json_returns_empty(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            f.write("not json")

        interface = LLMInterface()
        bounds = interface._load_hyperparam_bounds(str(ws))
        assert bounds == {}

    def test_no_bounds_key_returns_empty(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        with open(ws / "config" / "hyperparam_bounds.json", "w") as f:
            json.dump({"other_key": "value"}, f)

        interface = LLMInterface()
        bounds = interface._load_hyperparam_bounds(str(ws))
        assert bounds == {}


# ------------------------------------------------------------------
# _load_combo_membership
# ------------------------------------------------------------------

class TestLoadComboMembership:
    def test_loads_membership(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        config = {
            "default_combo": "combo_A",
            "combos": {
                "combo_A": {"models": ["m1", "m2"]},
                "combo_B": {"models": ["m2", "m3"]},
            }
        }
        with open(ws / "config" / "ensemble_config.json", "w") as f:
            json.dump(config, f)

        interface = LLMInterface()
        membership = interface._load_combo_membership(str(ws))
        assert "m1" in membership
        assert membership["m1"] == ["combo_A"]
        assert membership["m2"] == ["combo_A", "combo_B"]
        assert "m3" in membership

    def test_missing_file_returns_empty(self, tmp_path):
        interface = LLMInterface()
        assert interface._load_combo_membership(str(tmp_path)) == {}

    def test_invalid_json_returns_empty(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        with open(ws / "config" / "ensemble_config.json", "w") as f:
            f.write("invalid")

        interface = LLMInterface()
        assert interface._load_combo_membership(str(ws)) == {}

    def test_no_combos_key(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        with open(ws / "config" / "ensemble_config.json", "w") as f:
            json.dump({"other": "data"}, f)

        interface = LLMInterface()
        assert interface._load_combo_membership(str(ws)) == {}

    def test_non_dict_combo_skipped(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        config = {
            "combos": {
                "combo_A": "not_a_dict",
                "combo_B": {"models": ["m1"]},
            }
        }
        with open(ws / "config" / "ensemble_config.json", "w") as f:
            json.dump(config, f)

        interface = LLMInterface()
        membership = interface._load_combo_membership(str(ws))
        assert "m1" in membership
        # combo_A is skipped because it's not a dict


# ------------------------------------------------------------------
# _load_recent_action_history
# ------------------------------------------------------------------

class TestLoadRecentActionHistory:
    def test_loads_history(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        entries = [
            {"_run_date": "2026-05-01", "target": "m1", "params": {"lr": {"from": 0.001, "to": 0.01}}},
            {"_run_date": "2026-05-02", "target": "m2", "params": {"n_epochs": {"from": 100, "to": 150}}},
            {"_run_date": "2026-05-03", "target": "m3", "params": {"dropout": {"from": 0.0, "to": 0.1}}},
        ]
        with open(ws / "data" / "action_item_history.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        interface = LLMInterface()
        history = interface._load_recent_action_history(str(ws), limit=2)
        assert len(history) == 2
        # Newest first (reversed)
        assert history[0]["_run_date"] == "2026-05-03"
        assert history[1]["_run_date"] == "2026-05-02"

    def test_missing_file_returns_empty(self, tmp_path):
        interface = LLMInterface()
        assert interface._load_recent_action_history(str(tmp_path)) == []

    def test_corrupt_lines_skipped(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "data").mkdir(parents=True)
        with open(ws / "data" / "action_item_history.jsonl", "w") as f:
            f.write('{"_run_date": "2026-05-01", "target": "m1"}\n')
            f.write('not valid json\n')
            f.write('{"_run_date": "2026-05-02", "target": "m2"}\n')

        interface = LLMInterface()
        history = interface._load_recent_action_history(str(ws), limit=10)
        assert len(history) == 2  # bad line skipped


# ------------------------------------------------------------------
# _aggregate_signals_for_triage
# ------------------------------------------------------------------

class TestAggregateSignalsForTriage:
    def test_groups_signals_by_type(self):
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="ctx1"),
            Signal("underfitting", "warning", "hyperparams", "Agent1", "m2", context="ctx2"),
            Signal("underfitting", "warning", "hyperparams", "Agent1", "m3", context="ctx3"),
            Signal("underfitting", "warning", "hyperparams", "Agent1", "m4", context="ctx4"),
            Signal("ic_decay", "critical", "hyperparams", "Agent2", "m1", context="ctx_ic"),
        ]

        interface = LLMInterface()
        result = interface._aggregate_signals_for_triage(signals)
        assert "underfitting" in result
        assert result["underfitting"]["count"] == 4
        assert len(result["underfitting"]["models"]) == 4
        assert len(result["underfitting"]["sample_contexts"]) == 3  # first 3 only
        assert "ic_decay" in result
        assert result["ic_decay"]["count"] == 1

    def test_empty_signals(self):
        interface = LLMInterface()
        result = interface._aggregate_signals_for_triage([])
        assert result == {}

    def test_signals_with_metrics(self):
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent1", "m1",
                   context="test", metrics={"ic": 0.01, "icir": 0.1}),
        ]
        interface = LLMInterface()
        result = interface._aggregate_signals_for_triage(signals)
        assert result["underfitting"]["count"] == 1


# ------------------------------------------------------------------
# _compute_available_interventions
# ------------------------------------------------------------------

class TestComputeAvailableInterventions:
    def test_computes_untouched_params(self):
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="test"),
        ]
        current_params = {
            "m1": {"n_epochs": 200, "lr": 0.001, "dropout": 0.0, "GPU": 0, "seed": 42},
        }
        recent_history = []  # no prior adjustments

        interface = LLMInterface()
        result = interface._compute_available_interventions(signals, recent_history, current_params)
        assert "m1" in result
        # Non-tunable params (GPU, seed) should be excluded
        untouched = result["m1"]["untouched"]
        assert "n_epochs" in untouched
        assert "lr" in untouched
        assert "dropout" in untouched
        assert "GPU" not in untouched
        assert "seed" not in untouched
        assert not result["m1"]["exhausted"]

    def test_computes_recently_adjusted(self):
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="test"),
        ]
        current_params = {"m1": {"n_epochs": 200, "lr": 0.001, "dropout": 0.0}}
        recent_history = [
            {"_run_date": "2026-05-06", "target": "m1", "params": {"lr": {"from": 0.001, "to": 0.01}}},
        ]

        interface = LLMInterface()
        result = interface._compute_available_interventions(signals, recent_history, current_params)
        assert "lr" in result["m1"]["recently_adjusted"]
        assert "n_epochs" in result["m1"]["untouched"]
        assert "dropout" in result["m1"]["untouched"]

    def test_exhausted_when_all_tunable_adjusted(self):
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="test"),
        ]
        current_params = {"m1": {"n_epochs": 200, "lr": 0.001}}
        recent_history = [
            {"_run_date": "2026-05-06", "target": "m1", "params": {"lr": {}, "n_epochs": {}}},
        ]

        interface = LLMInterface()
        result = interface._compute_available_interventions(signals, recent_history, current_params)
        assert result["m1"]["exhausted"] is True

    def test_old_history_ignored(self):
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="test"),
        ]
        current_params = {"m1": {"n_epochs": 200, "lr": 0.001}}
        recent_history = [
            {"_run_date": "2026-01-01", "target": "m1", "params": {"lr": {"from": 0.001, "to": 0.01}}},
        ]

        interface = LLMInterface()
        result = interface._compute_available_interventions(signals, recent_history, current_params)
        # Old history (>30 days) should be ignored, so all params are untouched
        assert "lr" in result["m1"]["untouched"]
        assert result["m1"]["recently_adjusted"] == []

    def test_duplicate_model_signals_deduplicated(self):
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="ctx1"),
            Signal("overfitting", "warning", "hyperparams", "Agent2", "m1", context="ctx2"),
        ]
        current_params = {"m1": {"n_epochs": 200}}
        interface = LLMInterface()
        result = interface._compute_available_interventions(signals, [], current_params)
        assert len(result) == 1  # only one model entry


# ------------------------------------------------------------------
# _build_critic_prompt with extended params
# ------------------------------------------------------------------

class TestBuildCriticPromptExtended:
    def test_with_hyperparam_bounds(self):
        signals = [Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="test")]
        bounds = {
            "n_epochs": {"min": 10, "max": 500, "max_change_pct": 50},
            "lr": {"min": 1e-5, "max": 1e-2, "max_change_pct": None},
        }

        interface = LLMInterface()
        prompt = interface._build_critic_prompt(signals, ["hyperparams"], hyperparam_bounds=bounds)
        assert "Hyperparameter Bounds" in prompt
        assert "max_change_pct" in prompt
        assert "null" in prompt  # None serialized as null

    def test_with_current_params(self):
        signals = [Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="test")]
        current_params = {"m1": {"n_epochs": 200, "lr": 0.001}}

        interface = LLMInterface()
        prompt = interface._build_critic_prompt(signals, ["hyperparams"], current_params=current_params)
        assert "Current Hyperparameter Values" in prompt
        assert "n_epochs" in prompt

    def test_with_recent_history(self):
        signals = [Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="test")]
        history = [
            {"_run_date": "2026-05-01", "target": "m1",
             "params": {"lr": {"from": 0.001, "to": 0.01}}, "action_type": "adjust_hyperparam"},
        ]

        interface = LLMInterface()
        prompt = interface._build_critic_prompt(signals, ["hyperparams"], recent_history=history)
        assert "Recent Action History" in prompt

    def test_with_recent_history_other_target_filtered(self):
        signals = [Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="test")]
        history = [
            {"_run_date": "2026-05-01", "target": "other_model", "params": {}},
        ]

        interface = LLMInterface()
        prompt = interface._build_critic_prompt(signals, ["hyperparams"], recent_history=history)
        # History should be filtered to current signal targets only
        assert "Recent Action History" not in prompt  # relevant_history is empty

    def test_minimal_prompt(self):
        signals = [Signal("underfitting", "warning", "hyperparams", "Agent1", "m1", context="test")]

        interface = LLMInterface()
        prompt = interface._build_critic_prompt(signals, ["hyperparams"])
        assert "Active Scopes" in prompt
        assert "Signals" in prompt
        assert "JSON array" in prompt


# ------------------------------------------------------------------
# _load_triage_skill
# ------------------------------------------------------------------

class TestLoadTriageSkill:
    def test_loads_when_present(self, tmp_path):
        ws = tmp_path / "ws"
        skills_dir = ws / "config" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "triage_system.md").write_text("TRIAGE PROMPT")

        interface = LLMInterface()
        result = interface._load_triage_skill(str(ws))
        assert result == "TRIAGE PROMPT"

    def test_returns_none_when_missing(self, tmp_path):
        interface = LLMInterface()
        assert interface._load_triage_skill(str(tmp_path)) is None

    def test_returns_none_when_empty(self, tmp_path):
        ws = tmp_path / "ws"
        skills_dir = ws / "config" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "triage_system.md").write_text("")

        interface = LLMInterface()
        assert interface._load_triage_skill(str(ws)) is None

    def test_exception_returns_none(self, tmp_path):
        ws = tmp_path / "ws"
        skills_dir = ws / "config" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "triage_system.md").write_text("content")

        interface = LLMInterface()
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            result = interface._load_triage_skill(str(ws))
            assert result is None


# ------------------------------------------------------------------
# _load_active_scopes
# ------------------------------------------------------------------

class TestLoadActiveScopes:
    def test_loads_scopes(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        with open(ws / "config" / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams", "model_selection"]}, f)

        interface = LLMInterface()
        scopes = interface._load_active_scopes(str(ws))
        assert scopes == ["hyperparams", "model_selection"]

    def test_missing_file_returns_empty(self, tmp_path):
        interface = LLMInterface()
        assert interface._load_active_scopes(str(tmp_path)) == []

    def test_exception_returns_empty(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        with open(ws / "config" / "feedback_scope.json", "w") as f:
            f.write("invalid")

        interface = LLMInterface()
        assert interface._load_active_scopes(str(ws)) == []

    def test_no_active_scopes_key(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        with open(ws / "config" / "feedback_scope.json", "w") as f:
            json.dump({"other": "data"}, f)

        interface = LLMInterface()
        assert interface._load_active_scopes(str(ws)) == []


# ------------------------------------------------------------------
# _load_workspace_llm_config
# ------------------------------------------------------------------

class TestLoadWorkspaceLLMConfigExtended:
    def test_exception_returns_empty(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        with open(ws / "config" / "llm_config.json", "w") as f:
            f.write("not json")

        interface = LLMInterface()
        assert interface._load_workspace_llm_config(str(ws)) == {}


# ------------------------------------------------------------------
# _build_triage_prompt
# ------------------------------------------------------------------

class TestBuildTriagePrompt:
    def test_minimal(self):
        interface = LLMInterface()
        prompt = interface._build_triage_prompt(
            signal_summary={"underfitting": {"count": 3, "models": ["m1", "m2", "m3"], "sample_contexts": ["ctx"]}},
            history_summary=[],
            global_defaults=[],
            active_scopes=["hyperparams"],
            total_signals=3,
        )
        assert "Active Scopes" in prompt
        assert "underfitting" in prompt
        assert "JSON object" in prompt

    def test_with_combo_membership(self):
        interface = LLMInterface()
        prompt = interface._build_triage_prompt(
            signal_summary={"underfitting": {"count": 1, "models": ["m1"], "sample_contexts": ["ctx"]}},
            history_summary=[],
            global_defaults=[],
            active_scopes=["hyperparams"],
            total_signals=1,
            combo_membership={"m1": ["combo_A"]},
        )
        assert "Combo Membership" in prompt
        assert "combo_A" in prompt

    def test_with_available_interventions(self):
        interface = LLMInterface()
        interventions = {"m1": {"recently_adjusted": ["lr"], "untouched": ["n_epochs", "dropout"], "exhausted": False}}
        prompt = interface._build_triage_prompt(
            signal_summary={"underfitting": {"count": 1, "models": ["m1"], "sample_contexts": ["ctx"]}},
            history_summary=[],
            global_defaults=[],
            active_scopes=["hyperparams"],
            total_signals=1,
            available_interventions=interventions,
        )
        assert "Intervention Availability" in prompt
        assert "recently_adjusted" in prompt
        assert "untouched" in prompt

    def test_with_global_defaults(self):
        interface = LLMInterface()
        prompt = interface._build_triage_prompt(
            signal_summary={"underfitting": {"count": 1, "models": ["m1"], "sample_contexts": ["ctx"]}},
            history_summary=[],
            global_defaults=["early_stop=10 appears in 15/20 models"],
            active_scopes=["hyperparams"],
            total_signals=1,
        )
        assert "Global Defaults" in prompt
        assert "early_stop=10" in prompt

    def test_with_out_of_scope_summary(self):
        interface = LLMInterface()
        prompt = interface._build_triage_prompt(
            signal_summary={"underfitting": {"count": 1, "models": ["m1"], "sample_contexts": ["ctx"]}},
            history_summary=[],
            global_defaults=[],
            active_scopes=["hyperparams"],
            total_signals=1,
            out_of_scope_summary={"combo_search": 5, "model_selection": 2},
        )
        assert "Out-of-Scope Signals" in prompt
        assert "combo_search: 5" in prompt
        assert "model_selection: 2" in prompt
        assert "DO NOT prioritize" in prompt

    def test_with_history_summary(self):
        interface = LLMInterface()
        prompt = interface._build_triage_prompt(
            signal_summary={"underfitting": {"count": 1, "models": ["m1"], "sample_contexts": ["ctx"]}},
            history_summary=[
                {"date": "2026-05-01", "target": "m2", "action_type": "adjust_hyperparam",
                 "params": {"lr": {}}, "status": "in_scope"},
            ],
            global_defaults=[],
            active_scopes=["hyperparams"],
            total_signals=1,
        )
        assert "Recent Action Item History" in prompt


# ------------------------------------------------------------------
# _call_critic_llm
# ------------------------------------------------------------------

class TestCallCriticLLM:
    def test_successful_call(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([{
            "action_type": "adjust_hyperparam",
            "scope": "hyperparams",
            "target": "m1",
            "params": {},
        }])
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake")
        items = interface._call_critic_llm(
            system_prompt="test",
            user_prompt="test",
            model="gpt-4",
            temperature=0.3,
            max_tokens=8192,
            api_key="fake",
            base_url=None,
        )
        assert len(items) == 1
        assert items[0].target == "m1"

    def test_empty_response(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake")
        items = interface._call_critic_llm(
            system_prompt="test", user_prompt="test",
            model="gpt-4", temperature=0.3, max_tokens=8192,
            api_key="fake", base_url=None,
        )
        assert items == []  # empty content = empty list

    def test_parse_retry_on_first_failure(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client

        # First call returns invalid JSON
        mock_bad = MagicMock()
        mock_bad.choices = [MagicMock()]
        mock_bad.choices[0].message.content = "not json"

        # Second call returns valid JSON
        mock_good = MagicMock()
        mock_good.choices = [MagicMock()]
        mock_good.choices[0].message.content = json.dumps([{
            "action_type": "adjust_hyperparam",
            "scope": "hyperparams",
            "target": "m2",
            "params": {},
        }])

        mock_client.chat.completions.create.side_effect = [mock_bad, mock_good]

        interface = LLMInterface(api_key="fake")
        items = interface._call_critic_llm(
            system_prompt="test", user_prompt="test",
            model="gpt-4", temperature=0.3, max_tokens=8192,
            api_key="fake", base_url=None,
        )
        assert len(items) == 1
        assert items[0].target == "m2"
        assert mock_client.chat.completions.create.call_count == 2

    def test_both_attempts_fail(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client

        mock_bad = MagicMock()
        mock_bad.choices = [MagicMock()]
        mock_bad.choices[0].message.content = "not json"

        mock_client.chat.completions.create.side_effect = [mock_bad, mock_bad]

        interface = LLMInterface(api_key="fake")
        items = interface._call_critic_llm(
            system_prompt="test", user_prompt="test",
            model="gpt-4", temperature=0.3, max_tokens=8192,
            api_key="fake", base_url=None,
        )
        assert items == []

    def test_non_parse_error_does_not_retry(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = ValueError("API connection error")

        interface = LLMInterface(api_key="fake")
        items = interface._call_critic_llm(
            system_prompt="test", user_prompt="test",
            model="gpt-4", temperature=0.3, max_tokens=8192,
            api_key="fake", base_url=None,
        )
        assert items == []
        assert mock_client.chat.completions.create.call_count == 1

    def test_with_base_url(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([{
            "action_type": "test", "scope": "s", "target": "t", "params": {},
        }])
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake")
        items = interface._call_critic_llm(
            system_prompt="test", user_prompt="test",
            model="gpt-4", temperature=0.3, max_tokens=8192,
            api_key="fake", base_url="http://custom.url",
        )
        assert len(items) == 1
        # Verify base_url was passed to OpenAI
        call_kwargs = sys.modules["openai"].OpenAI.call_args[1]
        assert call_kwargs["base_url"] == "http://custom.url"


# ------------------------------------------------------------------
# _call_triage_llm
# ------------------------------------------------------------------

class TestCallTriageLLM:
    def test_successful_call(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "systemic_observations": ["obs1"],
            "prioritized_targets": [{"target": "m1", "priority_score": 8}],
            "excluded_targets": [],
        })
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake")
        result = interface._call_triage_llm(
            system_prompt="test", user_prompt="test",
            model="gpt-4", temperature=0.3,
            api_key="fake", base_url=None,
        )
        assert result["systemic_observations"] == ["obs1"]

    def test_empty_content(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].finish_reason = "stop"
        type(mock_response.choices[0].message).reasoning_content = None
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake")
        result = interface._call_triage_llm(
            system_prompt="test", user_prompt="test",
            model="gpt-4", temperature=0.3,
            api_key="fake", base_url=None,
        )
        assert result is None

    def test_no_json_in_response(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Here is my analysis but no JSON object..."
        mock_response.choices[0].finish_reason = "stop"
        type(mock_response.choices[0].message).reasoning_content = None
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake")
        result = interface._call_triage_llm(
            system_prompt="test", user_prompt="test",
            model="gpt-4", temperature=0.3,
            api_key="fake", base_url=None,
        )
        assert result is None

    def test_content_is_json_code_fence(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        content = '```json\n{"systemic_observations": [], "prioritized_targets": [], "excluded_targets": []}\n```'
        mock_response.choices[0].message.content = content
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake")
        result = interface._call_triage_llm(
            system_prompt="test", user_prompt="test",
            model="gpt-4", temperature=0.3,
            api_key="fake", base_url=None,
        )
        assert result is not None

    def test_api_call_exception(self):
        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        interface = LLMInterface(api_key="fake")
        result = interface._call_triage_llm(
            system_prompt="test", user_prompt="test",
            model="gpt-4", temperature=0.3,
            api_key="fake", base_url=None,
        )
        assert result is None


# ------------------------------------------------------------------
# _run_triage
# ------------------------------------------------------------------

class TestRunTriage:
    def test_all_signals_out_of_scope(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        interface = LLMInterface(api_key="fake")
        signals = [
            Signal("ic_decay", "warning", "combo_search", "Agent", "m1", context="test"),
            Signal("ic_decay", "warning", "strategy_params", "Agent", "m2", context="test"),
        ]

        result = interface._run_triage(
            signals=signals,
            triage_skill="TRIAGE SKILL",
            recent_history=[],
            active_scopes=["hyperparams"],  # none of the signals are in this scope
            current_params={"m1": {"n_epochs": 200}},
            combo_membership={},
            model="gpt-4", temperature=0.3,
            api_key="fake", base_url=None,
        )
        # All out of scope -> returns original signals unmodified
        assert result == signals

    def test_no_in_scope_signals_after_triage_llm_fails(self, tmp_path):
        """When triage LLM returns None, fallback to unfiltered signals."""
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)

        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        interface = LLMInterface(api_key="fake")
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent", "m1", context="test"),
        ]
        result = interface._run_triage(
            signals=signals,
            triage_skill="TRIAGE SKILL",
            recent_history=[],
            active_scopes=["hyperparams"],
            current_params={"m1": {"n_epochs": 200}},
            combo_membership={},
            model="gpt-4", temperature=0.3,
            api_key="fake", base_url=None,
        )
        assert result is None  # LLM failure returns None

    def test_triage_result_none_from_llm(self, tmp_path):
        """When _call_triage_llm returns None."""
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)

        interface = LLMInterface(api_key="fake")
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent", "m1", context="test"),
        ]
        with patch.object(interface, "_call_triage_llm", return_value=None):
            result = interface._run_triage(
                signals=signals,
                triage_skill="TRIAGE SKILL",
                recent_history=[],
                active_scopes=["hyperparams"],
                current_params={"m1": {"n_epochs": 200}},
                combo_membership={},
                model="gpt-4", temperature=0.3,
                api_key="fake", base_url=None,
            )
            assert result is None

    def test_triage_returns_no_prioritized_targets(self, tmp_path):
        """When triage returns empty prioritized_targets, fallback to in_scope signals."""
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)

        interface = LLMInterface(api_key="fake")
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent", "m1", context="test"),
            Signal("underfitting", "warning", "hyperparams", "Agent", "m2", context="test"),
        ]
        triage_result = {
            "systemic_observations": ["systemic issue"],
            "prioritized_targets": [],  # empty
            "excluded_targets": [{"target": "m1", "reason": "recently adjusted"}],
        }
        with patch.object(interface, "_call_triage_llm", return_value=triage_result):
            result = interface._run_triage(
                signals=signals,
                triage_skill="TRIAGE SKILL",
                recent_history=[],
                active_scopes=["hyperparams"],
                current_params={"m1": {"n_epochs": 200}, "m2": {"n_epochs": 200}},
                combo_membership={},
                model="gpt-4", temperature=0.3,
                api_key="fake", base_url=None,
            )
            assert result is not None  # returns in_scope signals as fallback

    def test_triage_filters_to_prioritized_targets(self, tmp_path):
        """Triage returns specific targets -> signals filtered."""
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)

        interface = LLMInterface(api_key="fake")
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent", "m1", context="test"),
            Signal("underfitting", "warning", "hyperparams", "Agent", "m2", context="test"),
            Signal("underfitting", "warning", "hyperparams", "Agent", "m3", context="test"),
        ]
        triage_result = {
            "systemic_observations": [],
            "prioritized_targets": [
                {"target": "m1", "priority_score": 8, "primary_signal": "underfitting",
                 "investigation_direction": "check n_epochs"},
            ],
            "excluded_targets": [
                {"target": "m2", "reason": "recently adjusted"},
                {"target": "m3", "reason": "weak signal"},
            ],
        }
        with patch.object(interface, "_call_triage_llm", return_value=triage_result):
            result = interface._run_triage(
                signals=signals,
                triage_skill="TRIAGE SKILL",
                recent_history=[],
                active_scopes=["hyperparams"],
                current_params={},
                combo_membership={},
                model="gpt-4", temperature=0.3,
                api_key="fake", base_url=None,
            )
            assert result is not None
            assert len(result) == 1
            assert result[0].target == "m1"


# ------------------------------------------------------------------
# _llm_summary (workspace config path)
# ------------------------------------------------------------------

class TestLLMSummaryWorkspaceConfig:
    def test_uses_workspace_config(self, tmp_path):
        ws = tmp_path / "ws"
        config_dir = ws / "config"
        config_dir.mkdir(parents=True)
        with open(config_dir / "llm_config.json", "w") as f:
            json.dump({
                "summary_model": "deepseek-v4-pro",
                "base_url": "http://custom.url",
                "api_key_env": "CUSTOM_KEY",
            }, f)

        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "LLM Generated Summary"
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake_key")
        with patch.dict("os.environ", {"CUSTOM_KEY": "fake_key"}, clear=True):
            result = interface.generate_executive_summary(
                {"health_status": "Healthy"}, str(ws),
            )
        assert result == "LLM Generated Summary"


# ------------------------------------------------------------------
# analyze_experiment_result
# ------------------------------------------------------------------

class TestAnalyzeExperimentResult:
    def test_no_api_key_returns_none(self, tmp_path):
        ws = tmp_path / "ws"
        interface = LLMInterface(api_key=None)
        with patch.dict("os.environ", {}, clear=True):
            result = interface.analyze_experiment_result(
                model_name="m1",
                baseline_ic=0.05,
                playground_ic=0.06,
                changes_tried=[],
                convergence={},
                current_params={"m1": {"n_epochs": 200}},
                available_interventions={"m1": {"untouched": ["lr"], "recently_adjusted": [], "exhausted": False}},
                hyperparam_bounds={"n_epochs": {"min": 10, "max": 500}},
                max_rounds_remaining=2,
                workspace_root=str(ws),
            )
            assert result is None

    def test_no_skill_file_returns_none(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "config").mkdir(parents=True)
        with open(ws / "config" / "llm_config.json", "w") as f:
            json.dump({"api_key_env": "OPENAI_API_KEY"}, f)

        interface = LLMInterface(api_key="fake_key")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"}, clear=True):
            result = interface.analyze_experiment_result(
                model_name="m1",
                baseline_ic=0.05,
                playground_ic=0.06,
                changes_tried=[],
                convergence={},
                current_params={"m1": {"n_epochs": 200}},
                available_interventions={"m1": {"untouched": ["lr"], "recently_adjusted": [], "exhausted": False}},
                hyperparam_bounds={},
                max_rounds_remaining=2,
                workspace_root=str(ws),
            )
            assert result is None

    def test_successful_analysis_first_round(self, tmp_path):
        ws = tmp_path / "ws"
        config_dir = ws / "config"
        config_dir.mkdir(parents=True)
        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "experiment_analyzer.md").write_text("EXPERIMENT ANALYZER SKILL")
        with open(config_dir / "llm_config.json", "w") as f:
            json.dump({"api_key_env": "OPENAI_API_KEY"}, f)

        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "decision": "retry",
            "reason": "Try dropout",
            "next_param": "dropout",
            "next_from": 0.0,
            "next_to": 0.2,
            "rationale": "Overfitting detected, try regularization",
        })
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake_key")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"}, clear=True):
            result = interface.analyze_experiment_result(
                model_name="m1",
                baseline_ic=0.05,
                playground_ic=0.05,
                changes_tried=[],
                convergence=None,
                current_params={"m1": {"n_epochs": 200, "dropout": 0.0}},
                available_interventions={"m1": {"untouched": ["dropout"], "recently_adjusted": [], "exhausted": False}},
                hyperparam_bounds={"dropout": {"min": 0.0, "max": 0.8}},
                max_rounds_remaining=2,
                workspace_root=str(ws),
                is_first_round=True,
            )
            assert result is not None
            assert result["decision"] == "retry"
            assert result["next_param"] == "dropout"

    def test_give_up_decision(self, tmp_path):
        ws = tmp_path / "ws"
        config_dir = ws / "config"
        config_dir.mkdir(parents=True)
        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "experiment_analyzer.md").write_text("EXPERIMENT ANALYZER SKILL")
        with open(config_dir / "llm_config.json", "w") as f:
            json.dump({"api_key_env": "OPENAI_API_KEY"}, f)

        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "decision": "give_up",
            "reason": "IC improved enough",
            "rationale": "No further changes needed",
        })
        mock_client.chat.completions.create.return_value = mock_response

        interface = LLMInterface(api_key="fake_key")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"}, clear=True):
            result = interface.analyze_experiment_result(
                model_name="m1",
                baseline_ic=0.05,
                playground_ic=0.07,
                changes_tried=[{"param": "n_epochs", "from": 200, "to": 250, "ic_result": 0.07}],
                convergence={"best_epoch": 50, "actual_epochs": 200},
                current_params={"m1": {"n_epochs": 250, "dropout": 0.0}},
                available_interventions={"m1": {"untouched": ["dropout"], "recently_adjusted": ["n_epochs"], "exhausted": False}},
                hyperparam_bounds={},
                max_rounds_remaining=0,
                workspace_root=str(ws),
            )
            assert result is not None
            assert result["decision"] == "give_up"

    def test_api_call_exception(self, tmp_path):
        ws = tmp_path / "ws"
        config_dir = ws / "config"
        config_dir.mkdir(parents=True)
        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "experiment_analyzer.md").write_text("SKILL")
        with open(config_dir / "llm_config.json", "w") as f:
            json.dump({"api_key_env": "OPENAI_API_KEY"}, f)

        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        interface = LLMInterface(api_key="fake_key")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"}, clear=True):
            result = interface.analyze_experiment_result(
                model_name="m1",
                baseline_ic=0.05,
                playground_ic=0.06,
                changes_tried=[],
                convergence={},
                current_params={"m1": {"n_epochs": 200}},
                available_interventions={"m1": {"untouched": ["lr"], "recently_adjusted": [], "exhausted": False}},
                hyperparam_bounds={},
                max_rounds_remaining=2,
                workspace_root=str(ws),
            )
            assert result is None


# ------------------------------------------------------------------
# generate_action_items — two-stage path
# ------------------------------------------------------------------

class TestGenerateActionItemsTwoStage:
    def test_uses_triage_when_skill_present_and_many_signals(self, tmp_path):
        ws = tmp_path / "ws"
        config_dir = ws / "config"
        config_dir.mkdir(parents=True)
        skills_dir = config_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "triage_system.md").write_text("TRIAGE PROMPT")
        (skills_dir / "critic_system.md").write_text("CRITIC PROMPT")
        with open(config_dir / "feedback_scope.json", "w") as f:
            json.dump({"active_scopes": ["hyperparams"]}, f)
        with open(config_dir / "llm_config.json", "w") as f:
            json.dump({"critic_model": "test-model"}, f)

        # Create 7 signals (more than 5, so triage is used)
        signals = [
            Signal("underfitting", "warning", "hyperparams", "Agent", f"m{i}", context="test")
            for i in range(7)
        ]

        interface = LLMInterface(api_key="fake_key")

        # Mock triage to return filtered signals
        triage_result = {
            "systemic_observations": [],
            "prioritized_targets": [
                {"target": "m0", "priority_score": 8, "primary_signal": "underfitting",
                 "investigation_direction": "check"},
            ],
            "excluded_targets": [],
        }

        mock_client = MagicMock()
        sys.modules["openai"].OpenAI.return_value = mock_client

        # First call: triage
        mock_triage_response = MagicMock()
        mock_triage_response.choices = [MagicMock()]
        mock_triage_response.choices[0].message.content = json.dumps(triage_result)

        # Second call: critic
        mock_critic_response = MagicMock()
        mock_critic_response.choices = [MagicMock()]
        mock_critic_response.choices[0].message.content = json.dumps([{
            "action_type": "adjust_hyperparam",
            "scope": "hyperparams",
            "target": "m0",
            "params": {"n_epochs": {"from": 100, "to": 150}},
            "reason": "test",
            "confidence": 0.8,
            "risk_level": "low",
        }])

        mock_client.chat.completions.create.side_effect = [
            mock_triage_response,  # triage call
            mock_critic_response,  # critic call
        ]

        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"}, clear=True):
            items = interface.generate_action_items(signals, str(ws))
            assert len(items) == 1
            assert items[0].target == "m0"
