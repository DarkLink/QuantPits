"""Tests for window_critic.py — focused LLM reasoning for training window analysis."""

from unittest.mock import MagicMock, patch

import pytest

from quantpits.scripts.deep_analysis.window_critic import WindowCritic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(json_response=None, should_raise=None):
    """Build a mock LLM interface with configurable behaviour."""
    llm = MagicMock()
    llm._call_llm_json_object.return_value = json_response
    if should_raise:
        llm._call_llm_json_object.side_effect = should_raise
    llm._load_workspace_llm_config.return_value = {}
    llm._resolve_effective_api_key.return_value = None
    return llm


def _make_benchmark_data(overrides=None):
    """Minimal valid benchmark_data dict."""
    data = {
        "current_window": {
            "config": {"train": 5, "valid": 2, "test": 2, "mode": "slide"},
            "segments": {
                "train": {
                    "date_range": {"start": "2021-01-01", "end": "2025-12-31"},
                    "regimes": ["bull", "bear"],
                    "regime_switch_count": 3,
                    "volatility": {"annualized": 0.22},
                    "drawdown_stats": {"major_dd_count": 2},
                    "cum_return": 0.45,
                },
                "valid": {
                    "date_range": {"start": "2026-01-01", "end": "2026-03-31"},
                    "regimes": ["bull"],
                    "regime_switch_count": 1,
                    "volatility": {"annualized": 0.18},
                    "drawdown_stats": {"major_dd_count": 0},
                    "cum_return": 0.05,
                },
                "test": {
                    "date_range": {"start": "2026-04-01", "end": "2026-06-15"},
                    "regimes": ["bear"],
                    "regime_switch_count": 2,
                    "volatility": {"annualized": 0.28},
                    "drawdown_stats": {"major_dd_count": 1},
                    "cum_return": -0.03,
                },
            },
            "cross_segment": {
                "train_vs_test_vol_ratio": 1.27,
                "train_vs_test_ks_statistic": 0.12,
                "train_to_valid_boundary": {"changed": True},
                "valid_to_test_boundary": {"changed": False},
                "regime_coverage_pct": 0.67,
                "missing_regimes": ["crash"],
            },
            "full_history": {
                "regimes": ["bull", "bear", "crash"],
                "regime_switch_count": 8,
                "volatility": {"annualized": 0.24},
                "drawdown_stats": {"major_dd_count": 4},
            },
        },
        "sliding_dynamics": {
            "stability_score": 0.85,
            "trend": "stable",
            "cliff_edges": [],
        },
        "what_if": {
            "top_candidates": [
                {
                    "config": {"train": 5, "valid": 2, "test": 2},
                    "quality_scores": {
                        "coverage": 0.80,
                        "similarity": 0.75,
                        "recency": 0.90,
                        "boundary_quality": 0.85,
                        "stability": 0.88,
                        "composite": 0.836,
                    },
                    "is_current": True,
                    "strength": "current",
                },
                {
                    "config": {"train": 6, "valid": 2, "test": 2},
                    "quality_scores": {
                        "coverage": 0.85,
                        "similarity": 0.72,
                        "recency": 0.88,
                        "boundary_quality": 0.82,
                        "stability": 0.86,
                        "composite": 0.826,
                    },
                    "strength": "better coverage",
                },
            ],
            "pareto_frontier": [],
            "search_space": {},
        },
    }
    if overrides:
        _deep_update(data, overrides)
    return data


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


# ---------------------------------------------------------------------------
# TestFormatFindings
# ---------------------------------------------------------------------------


class TestFormatFindings:
    """Tests for WindowCritic._format_findings() static method."""

    def test_empty_list(self):
        assert WindowCritic._format_findings([]) == "  (none)"

    def test_dict_findings(self):
        findings = [
            {"finding_type": "regime_coverage_gap", "severity": "warning",
             "context": "Missing crash regime in training data"},
        ]
        result = WindowCritic._format_findings(findings)
        assert "[warning] regime_coverage_gap:" in result
        assert "Missing crash regime" in result

    def test_object_findings_with_to_dict(self):
        class FakeFinding:
            def to_dict(self):
                return {"finding_type": "boundary_mismatch", "severity": "critical",
                        "context": "Boundary crosses regime transition"}

        result = WindowCritic._format_findings([FakeFinding()])
        assert "[critical] boundary_mismatch:" in result
        assert "Boundary crosses regime transition" in result

    def test_context_truncated_at_150_chars(self):
        long_context = "x" * 200
        findings = [{"finding_type": "test", "severity": "info", "context": long_context}]
        result = WindowCritic._format_findings(findings)
        # Should not contain the full 200 chars
        assert "x" * 151 not in result
        assert "x" * 150 in result

    def test_missing_fields_get_defaults(self):
        findings = [{}]
        result = WindowCritic._format_findings(findings)
        assert "[?] ?:" in result

    def test_object_without_to_dict(self):
        findings = [object()]  # has no to_dict()
        result = WindowCritic._format_findings(findings)
        assert "[?] ?:" in result

    def test_multiple_findings(self):
        findings = [
            {"finding_type": "a", "severity": "critical", "context": "ctx_a"},
            {"finding_type": "b", "severity": "warning", "context": "ctx_b"},
        ]
        result = WindowCritic._format_findings(findings)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "[critical] a:" in lines[0]
        assert "[warning] b:" in lines[1]


# ---------------------------------------------------------------------------
# TestFallbackDiagnose
# ---------------------------------------------------------------------------


class TestFallbackDiagnose:
    """Tests for WindowCritic._fallback_diagnose()."""

    def _make_critic(self):
        llm = MagicMock()
        return WindowCritic(llm, "/tmp/ws")

    def test_empty_findings(self):
        critic = self._make_critic()
        result = critic._fallback_diagnose([])
        assert result["problems"] == []
        assert result["urgency"] == "watch"
        assert result["root_cause"] == "No critical issues detected"

    def test_critical_severity(self):
        critic = self._make_critic()
        findings = [{"finding_type": "volatility_regime_shift", "severity": "critical",
                     "context": "Test vol 2x train vol"}]
        result = critic._fallback_diagnose(findings)
        assert result["urgency"] == "now"
        assert "volatility" in result["root_cause"].lower()
        assert len(result["problems"]) == 1
        assert result["problems"][0]["type"] == "volatility_regime_shift"
        assert result["problems"][0]["severity"] == "critical"

    def test_warning_severity(self):
        critic = self._make_critic()
        findings = [{"finding_type": "regime_coverage_gap", "severity": "warning",
                     "context": "Missing 2 regimes"}]
        result = critic._fallback_diagnose(findings)
        assert result["urgency"] == "soon"
        assert "regime" in result["root_cause"].lower()

    def test_info_severity_stays_watch(self):
        critic = self._make_critic()
        findings = [{"finding_type": "minor_issue", "severity": "info",
                     "context": "Minor thing"}]
        result = critic._fallback_diagnose(findings)
        assert result["urgency"] == "watch"
        assert result["root_cause"] == "No critical issues detected"

    def test_first_critical_wins(self):
        critic = self._make_critic()
        findings = [
            {"finding_type": "type_a", "severity": "warning", "context": ""},
            {"finding_type": "type_b", "severity": "critical", "context": ""},
            {"finding_type": "type_c", "severity": "critical", "context": ""},
        ]
        result = critic._fallback_diagnose(findings)
        assert result["urgency"] == "now"
        # First critical (type_b) sets the root cause
        assert result["root_cause"] == "type_b"

    def test_first_warning_wins_when_no_critical(self):
        critic = self._make_critic()
        findings = [
            {"finding_type": "type_x", "severity": "info", "context": ""},
            {"finding_type": "regime_coverage_gap", "severity": "warning", "context": ""},
            {"finding_type": "type_y", "severity": "warning", "context": ""},
        ]
        result = critic._fallback_diagnose(findings)
        assert result["urgency"] == "soon"
        assert "regime" in result["root_cause"].lower()

    def test_object_findings(self):
        class FakeFinding:
            def to_dict(self):
                return {"finding_type": "impending_regime_loss", "severity": "critical",
                        "context": "Regime about to drop"}

        critic = self._make_critic()
        result = critic._fallback_diagnose([FakeFinding()])
        assert result["urgency"] == "now"
        assert result["problems"][0]["type"] == "impending_regime_loss"

    def test_unknown_finding_type_maps_literally(self):
        critic = self._make_critic()
        findings = [{"finding_type": "weird_custom_problem", "severity": "warning",
                     "context": "Custom context"}]
        result = critic._fallback_diagnose(findings)
        assert result["root_cause"] == "weird_custom_problem"

    def test_known_root_cause_mapping(self):
        critic = self._make_critic()
        findings = [{"finding_type": "insufficient_drawdown_coverage", "severity": "critical",
                     "context": ""}]
        result = critic._fallback_diagnose(findings)
        assert "drawdown" in result["root_cause"].lower()

    def test_boundary_regime_mismatch_mapping(self):
        critic = self._make_critic()
        findings = [{"finding_type": "boundary_regime_mismatch", "severity": "critical",
                     "context": ""}]
        result = critic._fallback_diagnose(findings)
        assert "window boundaries" in result["root_cause"].lower()

    def test_evidence_truncated_at_200(self):
        critic = self._make_critic()
        long_ctx = "y" * 300
        findings = [{"finding_type": "test_type", "severity": "warning", "context": long_ctx}]
        result = critic._fallback_diagnose(findings)
        assert len(result["problems"][0]["evidence"]) == 200

    def test_finding_without_type_skipped(self):
        critic = self._make_critic()
        findings = [{"severity": "critical", "context": "no type here"}]
        result = critic._fallback_diagnose(findings)
        assert result["problems"] == []
        assert result["urgency"] == "watch"

    def test_implicated_segment_always_cross(self):
        critic = self._make_critic()
        findings = [{"finding_type": "x", "severity": "warning", "context": ""}]
        result = critic._fallback_diagnose(findings)
        assert result["problems"][0]["implicated_segment"] == "cross"

    def test_object_without_to_dict_uses_empty_dict(self):
        """Finding object with no to_dict() method — uses empty dict, so no type, skipped."""
        critic = self._make_critic()
        result = critic._fallback_diagnose([object()])
        assert result["problems"] == []


# ---------------------------------------------------------------------------
# TestFallbackRecommend
# ---------------------------------------------------------------------------


class TestFallbackRecommend:
    """Tests for WindowCritic._fallback_recommend()."""

    def _make_critic(self):
        return WindowCritic(MagicMock(), "/tmp/ws")

    def _make_candidates(self, count=3):
        return [
            {
                "config": {"train": 5 + i, "valid": 2, "test": 2},
                "quality_scores": {
                    "coverage": 0.80 - i * 0.05,
                    "similarity": 0.75 - i * 0.03,
                    "recency": 0.90 - i * 0.02,
                    "boundary_quality": 0.85 - i * 0.03,
                    "stability": 0.88 - i * 0.02,
                    "composite": 0.836 - i * 0.01,
                },
                "strength": f"candidate_{i}",
            }
            for i in range(count)
        ]

    def test_empty_candidates_returns_default(self):
        critic = self._make_critic()
        result = critic._fallback_recommend([])
        assert result["recommended_config"] == {"train": 5, "valid": 2, "test": 2}
        assert result["rationale"] == "No candidates available."
        assert result["tradeoffs"] == []
        assert result["alternatives"] == []

    def test_single_candidate(self):
        critic = self._make_critic()
        candidates = self._make_candidates(1)
        result = critic._fallback_recommend(candidates)
        assert result["recommended_config"] == {"train": 5, "valid": 2, "test": 2}
        assert "0.836" in result["rationale"]
        assert len(result["tradeoffs"]) == 1
        assert result["alternatives"] == []

    def test_multiple_candidates_produces_alternatives(self):
        critic = self._make_critic()
        candidates = self._make_candidates(5)
        result = critic._fallback_recommend(candidates)
        assert result["recommended_config"]["train"] == 5
        assert len(result["alternatives"]) == 3  # candidates[1:4]
        assert result["alternatives"][0]["config"]["train"] == 6
        assert result["alternatives"][1]["config"]["train"] == 7
        assert result["alternatives"][2]["config"]["train"] == 8

    def test_uses_top_candidate(self):
        critic = self._make_critic()
        candidates = self._make_candidates(3)
        result = critic._fallback_recommend(candidates)
        assert result["recommended_config"]["train"] == 5

    def test_alternative_strength_fallback(self):
        critic = self._make_critic()
        candidates = self._make_candidates(3)
        # Remove 'strength' key from the first alt
        del candidates[1]["strength"]
        result = critic._fallback_recommend(candidates)
        assert result["alternatives"][0]["when_to_use"] == "alternative"


# ---------------------------------------------------------------------------
# TestDiagnose — with mocked LLM
# ---------------------------------------------------------------------------


class TestDiagnose:
    """Tests for WindowCritic.diagnose() with mocked LLM interface."""

    def test_llm_returns_valid_json(self):
        llm = _make_mock_llm(json_response={
            "problems": [{"type": "regime_coverage_gap", "severity": "warning",
                          "evidence": "Missing crash regime", "implicated_segment": "train"}],
            "root_cause": "Training window misses crash regime",
            "urgency": "soon",
        })
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        result = critic.diagnose(bd, [])
        assert result["urgency"] == "soon"
        assert result["root_cause"] == "Training window misses crash regime"
        assert len(result["problems"]) == 1

    def test_llm_returns_none_falls_back(self):
        llm = _make_mock_llm(json_response=None)
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        findings = [{"finding_type": "test_type", "severity": "warning", "context": ""}]
        result = critic.diagnose(bd, findings)
        assert result["urgency"] == "soon"  # fallback rule triggers on warning

    def test_llm_returns_empty_dict_falls_back(self):
        llm = _make_mock_llm(json_response={})
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        findings = [{"finding_type": "test_type", "severity": "warning", "context": ""}]
        result = critic.diagnose(bd, findings)
        assert result["urgency"] == "soon"

    def test_llm_raises_exception_falls_back(self):
        llm = _make_mock_llm(should_raise=RuntimeError("API down"))
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        findings = [{"finding_type": "test_type", "severity": "warning", "context": ""}]
        result = critic.diagnose(bd, findings)
        assert result["urgency"] == "soon"

    def test_llm_returns_non_dict_falls_back(self):
        llm = _make_mock_llm(json_response="not a dict")
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        findings = [{"finding_type": "test_type", "severity": "warning", "context": ""}]
        result = critic.diagnose(bd, findings)
        assert result["urgency"] == "soon"

    def test_missing_benchmark_data_keys_handled_gracefully(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "healthy", "urgency": "watch",
        })
        critic = WindowCritic(llm, "/tmp/ws")
        result = critic.diagnose({}, [])
        assert result["urgency"] == "watch"
        assert result["root_cause"] == "healthy"

    def test_prompt_includes_formatted_values(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        critic.diagnose(bd, [])
        call_args = llm._call_llm_json_object.call_args
        user_prompt = call_args[1]["user_prompt"]
        assert "5 years" in user_prompt
        assert "2021-01-01" in user_prompt
        assert "1.27" in user_prompt
        assert "0.12" in user_prompt

    def test_resolves_model_from_workspace_config(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        llm._load_workspace_llm_config.return_value = {"critic_model": "gpt-4-turbo"}
        critic = WindowCritic(llm, "/tmp/ws")
        # Pass model="" so the falsy default lets workspace config take effect
        critic.diagnose(_make_benchmark_data(), [], model="")
        assert llm._call_llm_json_object.call_args[1]["model"] == "gpt-4-turbo"

    def test_explicit_model_overrides_workspace(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        llm._load_workspace_llm_config.return_value = {"critic_model": "gpt-4-turbo"}
        critic = WindowCritic(llm, "/tmp/ws")
        critic.diagnose(_make_benchmark_data(), [], model="gpt-4o")
        assert llm._call_llm_json_object.call_args[1]["model"] == "gpt-4o"

    def test_explicit_api_key_used(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        critic = WindowCritic(llm, "/tmp/ws")
        critic.diagnose(_make_benchmark_data(), [], api_key="sk-test")
        assert llm._call_llm_json_object.call_args[1]["api_key"] == "sk-test"

    def test_explicit_base_url_used(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        critic = WindowCritic(llm, "/tmp/ws")
        critic.diagnose(_make_benchmark_data(), [], base_url="https://custom.api")
        assert llm._call_llm_json_object.call_args[1]["base_url"] == "https://custom.api"

    def test_base_url_from_workspace_config(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        llm._load_workspace_llm_config.return_value = {"base_url": "https://ws.api"}
        critic = WindowCritic(llm, "/tmp/ws")
        critic.diagnose(_make_benchmark_data(), [])
        assert llm._call_llm_json_object.call_args[1]["base_url"] == "https://ws.api"

    def test_api_key_from_llm_resolver(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        llm._resolve_effective_api_key.return_value = "sk-resolved"
        critic = WindowCritic(llm, "/tmp/ws")
        critic.diagnose(_make_benchmark_data(), [])
        assert llm._call_llm_json_object.call_args[1]["api_key"] == "sk-resolved"

    def test_default_model_when_nothing_configured(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        llm._load_workspace_llm_config.return_value = {}
        critic = WindowCritic(llm, "/tmp/ws")
        critic.diagnose(_make_benchmark_data(), [], model="")
        assert llm._call_llm_json_object.call_args[1]["model"] == "gpt-4"

    def test_temperature_passed_through(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        critic = WindowCritic(llm, "/tmp/ws")
        critic.diagnose(_make_benchmark_data(), [], temperature=0.7)
        assert llm._call_llm_json_object.call_args[1]["temperature"] == 0.7

    def test_label_passed_to_llm_call(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        critic = WindowCritic(llm, "/tmp/ws")
        critic.diagnose(_make_benchmark_data(), [])
        assert llm._call_llm_json_object.call_args[1]["label"] == "Window Diagnosis"

    def test_missing_regimes_none_when_empty(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        bd["current_window"]["cross_segment"]["missing_regimes"] = []
        critic.diagnose(bd, [])
        user_prompt = llm._call_llm_json_object.call_args[1]["user_prompt"]
        assert "none" in user_prompt

    def test_missing_regimes_truncated_to_5(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        bd["current_window"]["cross_segment"]["missing_regimes"] = [
            "rA", "rB", "rC", "rD", "rE", "rF", "rG",
        ]
        critic.diagnose(bd, [])
        user_prompt = llm._call_llm_json_object.call_args[1]["user_prompt"]
        # First 5 present, 6th+ excluded
        assert "rA, rB, rC, rD, rE" in user_prompt
        assert "rF" not in user_prompt
        assert "rG" not in user_prompt

    def test_cliff_count_from_sliding_dynamics(self):
        llm = _make_mock_llm(json_response={
            "problems": [], "root_cause": "ok", "urgency": "watch",
        })
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        bd["sliding_dynamics"]["cliff_edges"] = [1, 2, 3]
        critic.diagnose(bd, [])
        user_prompt = llm._call_llm_json_object.call_args[1]["user_prompt"]
        assert "3" in user_prompt  # cliff_count = 3


# ---------------------------------------------------------------------------
# TestRecommend — with mocked LLM
# ---------------------------------------------------------------------------


class TestRecommend:
    """Tests for WindowCritic.recommend() with mocked LLM interface."""

    def test_empty_candidates_early_return(self):
        llm = _make_mock_llm()
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        bd["what_if"]["top_candidates"] = []
        result = critic.recommend(bd)
        assert result["recommended_config"] == {"train": 5, "valid": 2, "test": 2}
        assert result["rationale"] == "No candidates available for comparison."
        assert result["tradeoffs"] == []
        assert result["alternatives"] == []
        # LLM should NOT be called
        llm._call_llm_json_object.assert_not_called()

    def test_llm_returns_valid_recommendation(self):
        llm = _make_mock_llm(json_response={
            "recommended_config": {"train": 6, "valid": 2, "test": 2},
            "rationale": "Better coverage without sacrificing recency",
            "tradeoffs": [{"gains": "Better regime coverage", "costs": "Slightly older data"}],
            "alternatives": [
                {"config": {"train": 7, "valid": 2, "test": 2},
                 "when_to_use": "If regime diversity is highest priority"},
            ],
        })
        critic = WindowCritic(llm, "/tmp/ws")
        result = critic.recommend(_make_benchmark_data())
        assert result["recommended_config"]["train"] == 6
        assert "Better coverage" in result["rationale"]
        assert len(result["tradeoffs"]) == 1
        assert len(result["alternatives"]) == 1

    def test_llm_raises_exception_falls_back(self):
        llm = _make_mock_llm(should_raise=RuntimeError("API error"))
        critic = WindowCritic(llm, "/tmp/ws")
        result = critic.recommend(_make_benchmark_data())
        assert result["recommended_config"]["train"] == 5  # top candidate
        assert "Top composite score" in result["rationale"]

    def test_llm_returns_none_falls_back(self):
        llm = _make_mock_llm(json_response=None)
        critic = WindowCritic(llm, "/tmp/ws")
        result = critic.recommend(_make_benchmark_data())
        assert result["recommended_config"]["train"] == 5

    def test_llm_returns_non_dict_falls_back(self):
        llm = _make_mock_llm(json_response="not a dict")
        critic = WindowCritic(llm, "/tmp/ws")
        result = critic.recommend(_make_benchmark_data())
        assert result["recommended_config"]["train"] == 5

    def test_llm_missing_recommended_config_key_falls_back(self):
        llm = _make_mock_llm(json_response={"some": "other keys"})
        critic = WindowCritic(llm, "/tmp/ws")
        result = critic.recommend(_make_benchmark_data())
        assert result["recommended_config"]["train"] == 5

    def test_diagnosis_none_handled(self):
        llm = _make_mock_llm(json_response={
            "recommended_config": {"train": 5, "valid": 2, "test": 2},
            "rationale": "Current config is optimal",
            "tradeoffs": [],
            "alternatives": [],
        })
        critic = WindowCritic(llm, "/tmp/ws")
        result = critic.recommend(_make_benchmark_data(), diagnosis=None)
        assert result["recommended_config"]["train"] == 5

    def test_candidates_table_limited_to_15(self):
        llm = _make_mock_llm(json_response={
            "recommended_config": {"train": 5, "valid": 2, "test": 2},
            "rationale": "ok", "tradeoffs": [], "alternatives": [],
        })
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        bd["what_if"]["top_candidates"] = [
            {
                "config": {"train": i, "valid": 2, "test": 2},
                "quality_scores": {
                    "coverage": 0.8, "similarity": 0.7, "recency": 0.9,
                    "boundary_quality": 0.8, "stability": 0.8, "composite": 0.8,
                },
            }
            for i in range(1, 25)
        ]
        critic.recommend(bd)
        user_prompt = llm._call_llm_json_object.call_args[1]["user_prompt"]
        # First 15 candidates present, 16th excluded
        assert "| 15 |" in user_prompt
        assert "| 16 |" not in user_prompt

    def test_candidates_table_marks_current(self):
        llm = _make_mock_llm(json_response={
            "recommended_config": {"train": 5, "valid": 2, "test": 2},
            "rationale": "ok", "tradeoffs": [], "alternatives": [],
        })
        critic = WindowCritic(llm, "/tmp/ws")
        critic.recommend(_make_benchmark_data())
        user_prompt = llm._call_llm_json_object.call_args[1]["user_prompt"]
        assert "CURRENT" in user_prompt

    def test_resolves_model_from_workspace_config(self):
        llm = _make_mock_llm(json_response={
            "recommended_config": {"train": 5, "valid": 2, "test": 2},
            "rationale": "ok", "tradeoffs": [], "alternatives": [],
        })
        llm._load_workspace_llm_config.return_value = {"critic_model": "gpt-4-turbo"}
        critic = WindowCritic(llm, "/tmp/ws")
        # Pass model="" so the falsy default lets workspace config take effect
        critic.recommend(_make_benchmark_data(), model="")
        assert llm._call_llm_json_object.call_args[1]["model"] == "gpt-4-turbo"

    def test_label_passed_to_llm_call(self):
        llm = _make_mock_llm(json_response={
            "recommended_config": {"train": 5, "valid": 2, "test": 2},
            "rationale": "ok", "tradeoffs": [], "alternatives": [],
        })
        critic = WindowCritic(llm, "/tmp/ws")
        critic.recommend(_make_benchmark_data())
        assert llm._call_llm_json_object.call_args[1]["label"] == "Window Recommendation"

    def test_current_composite_resolved_from_candidates(self):
        llm = _make_mock_llm(json_response={
            "recommended_config": {"train": 5, "valid": 2, "test": 2},
            "rationale": "ok", "tradeoffs": [], "alternatives": [],
        })
        critic = WindowCritic(llm, "/tmp/ws")
        critic.recommend(_make_benchmark_data())
        user_prompt = llm._call_llm_json_object.call_args[1]["user_prompt"]
        assert "0.836" in user_prompt  # composite from current

    def test_problems_summary_truncated_to_5(self):
        llm = _make_mock_llm(json_response={
            "recommended_config": {"train": 5, "valid": 2, "test": 2},
            "rationale": "ok", "tradeoffs": [], "alternatives": [],
        })
        critic = WindowCritic(llm, "/tmp/ws")
        diagnosis = {
            "problems": [
                {"type": f"p{i}", "severity": "warning"} for i in range(10)
            ],
            "root_cause": "many issues",
            "urgency": "soon",
        }
        critic.recommend(_make_benchmark_data(), diagnosis=diagnosis)
        user_prompt = llm._call_llm_json_object.call_args[1]["user_prompt"]
        # Only 5 problems should appear
        assert "p5" not in user_prompt

    def test_no_current_marker_when_no_is_current(self):
        llm = _make_mock_llm(json_response={
            "recommended_config": {"train": 5, "valid": 2, "test": 2},
            "rationale": "ok", "tradeoffs": [], "alternatives": [],
        })
        critic = WindowCritic(llm, "/tmp/ws")
        bd = _make_benchmark_data()
        for c in bd["what_if"]["top_candidates"]:
            c.pop("is_current", None)
        critic.recommend(bd)
        user_prompt = llm._call_llm_json_object.call_args[1]["user_prompt"]
        assert "CURRENT" not in user_prompt
        assert "0.000" in user_prompt  # cur_composite = 0


# ---------------------------------------------------------------------------
# TestConstructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for WindowCritic.__init__()."""

    def test_stores_llm_and_workspace(self):
        llm = MagicMock()
        critic = WindowCritic(llm, "/some/workspace")
        assert critic._llm is llm
        assert critic._workspace_root == "/some/workspace"
