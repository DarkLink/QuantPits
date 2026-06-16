"""
Window Critic — focused LLM reasoning for training window analysis.

Two independent, narrow-scope LLM tasks:
  1. Diagnosis: interpret per-segment stats + findings → structured assessment
  2. Recommendation: evaluate what-if candidates → ranked config recommendation

Each task produces structured JSON. Outputs flow into the existing
SignalExtractor → Triage → Synthesizer pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DIAGNOSIS_SYSTEM = """\
You are a quantitative market analyst specializing in training data quality \
for machine learning models. Your job is to diagnose problems with a model's \
training/validation/test data split based on market regime statistics.

Rules:
1. Focus on REGIME-LEVEL problems, not parameter tuning.
2. Cite specific evidence from the data provided.
3. Be concise and precise in your assessment.
4. Output valid JSON only — no markdown, no commentary outside the JSON.
5. If everything looks healthy, say so with urgency="watch"."""

DIAGNOSIS_USER_TEMPLATE = """\
## Current Window Configuration
train: {train_w} years ({train_start} to {train_end})
valid: {valid_w} years ({valid_start} to {valid_end})
test: {test_w} years ({test_start} to {test_end})
mode: {mode}

## Per-Segment Market Analysis
| Metric | Train | Valid | Test | Full History |
|--------|-------|-------|------|-------------|
| Regime types covered | {train_regimes} | {valid_regimes} | {test_regimes} | {full_regimes} |
| Regime switches | {train_switches} | {valid_switches} | {test_switches} | {full_switches} |
| Ann. volatility | {train_vol:.1%} | {valid_vol:.1%} | {test_vol:.1%} | {full_vol:.1%} |
| Major drawdowns (>15%) | {train_dd} | {valid_dd} | {test_dd} | {full_dd} |
| Cum. return | {train_cumret:.1%} | {valid_cumret:.1%} | {test_cumret:.1%} | n/a |

## Cross-Segment Comparisons
- Train vs test volatility ratio: {vol_ratio:.2f}
- Train vs test KS statistic: {ks_stat:.3f}
- Train→valid boundary regime change: {tv_change}
- Valid→test boundary regime change: {vt_change}
- Regime coverage: {coverage_pct:.0%} of full-history regime types (missing: {missing_regimes})

## Detector Findings
{findings_text}

## Sliding Dynamics (52 weeks)
- Coverage stability: {stability:.2f} (1.0 = identical every week)
- Trend: {trend}
- Cliff edges detected: {cliff_count}

Based on the above data, diagnose the current window's problems.

Output a JSON object:
{{
  "problems": [
    {{
      "type": "string (e.g., regime_coverage_gap, boundary_regime_mismatch, volatility_shift)",
      "severity": "critical|warning|info",
      "evidence": "specific data point supporting this problem",
      "implicated_segment": "train|valid|test|cross"
    }}
  ],
  "root_cause": "one-sentence summary of the primary issue",
  "urgency": "now|soon|watch"
}}
"""

RECOMMENDATION_SYSTEM = """\
You are a portfolio optimization engineer. Your job is to recommend the best \
training window configuration given quality scores for candidate configs.

Rules:
1. Balance coverage (regime diversity) against similarity (train-test alignment) against recency.
2. The composite score is a weighted guide, not a mandate — use your judgment.
3. Explain tradeoffs clearly: what is gained and what is sacrificed.
4. Output valid JSON only — no markdown, no commentary outside the JSON."""

RECOMMENDATION_USER_TEMPLATE = """\
## Current Config: train={cur_train}, valid={cur_valid}, test={cur_test} — composite={cur_composite}

## Pareto-Optimal Candidates (ranked by composite score)
| Rank | Train | Valid | Test | Coverage | Similarity | Recency | Boundary | Stability | Composite |
|------|-------|-------|------|----------|------------|---------|----------|-----------|-----------|
{candidates_table}

## Diagnosis
Root cause: {root_cause}
Urgency: {urgency}
Problems: {problems_summary}

## Context
- Full history covers {n_regimes} distinct market regimes from 2005 to present.
- Current training covers {cur_coverage:.0%} of these regimes.
- Test data is the most recent {cur_test} window(s).

Output a JSON object:
{{
  "recommended_config": {{"train": <number>, "valid": <number>, "test": <number>}},
  "rationale": "2-3 sentence explanation of why this config is best",
  "tradeoffs": [
    {{"gains": "what improves", "costs": "what is sacrificed"}}
  ],
  "alternatives": [
    {{"config": {{"train": N, "valid": N, "test": N}}, "when_to_use": "scenario description"}}
  ]
}}
"""

# ---------------------------------------------------------------------------
# WindowCritic
# ---------------------------------------------------------------------------


class WindowCritic:
    """Focused LLM reasoning for training window analysis.

    Two independent tasks:
    1. Diagnosis: interpret stats + findings → structured problem assessment
    2. Recommendation: evaluate candidates → ranked config recommendation
    """

    def __init__(self, llm_interface, workspace_root: str):
        self._llm = llm_interface
        self._workspace_root = workspace_root

    # ------------------------------------------------------------------
    # Task A: Diagnosis
    # ------------------------------------------------------------------

    def diagnose(
        self,
        benchmark_data: dict,
        window_findings: list,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
    ) -> dict:
        """Diagnose current window problems from market regime statistics.

        Args:
            benchmark_data: Output from BenchmarkDataLoader.
            window_findings: List of WindowAnalysisFinding (or their to_dict() output).
            model, api_key, base_url, temperature: LLM config.

        Returns:
            dict with keys: problems, root_cause, urgency.
            Falls back to a rule-based diagnosis if LLM is unavailable.
        """
        cw = benchmark_data.get("current_window", {})
        sd = benchmark_data.get("sliding_dynamics", {})
        config = cw.get("config", {})
        segs = cw.get("segments", {})
        cross = cw.get("cross_segment", {})
        full_hist = cw.get("full_history", {})

        # Format findings
        findings_text = self._format_findings(window_findings)

        # Build user prompt
        user_prompt = DIAGNOSIS_USER_TEMPLATE.format(
            train_w=config.get("train", "?"),
            train_start=segs.get("train", {}).get("date_range", {}).get("start", "?"),
            train_end=segs.get("train", {}).get("date_range", {}).get("end", "?"),
            valid_w=config.get("valid", "?"),
            valid_start=segs.get("valid", {}).get("date_range", {}).get("start", "?"),
            valid_end=segs.get("valid", {}).get("date_range", {}).get("end", "?"),
            test_w=config.get("test", "?"),
            test_start=segs.get("test", {}).get("date_range", {}).get("start", "?"),
            test_end=segs.get("test", {}).get("date_range", {}).get("end", "?"),
            mode=config.get("mode", "?"),
            train_regimes=len(segs.get("train", {}).get("regimes", [])),
            valid_regimes=len(segs.get("valid", {}).get("regimes", [])),
            test_regimes=len(segs.get("test", {}).get("regimes", [])),
            full_regimes=len(full_hist.get("regimes", [])),
            train_switches=segs.get("train", {}).get("regime_switch_count", 0),
            valid_switches=segs.get("valid", {}).get("regime_switch_count", 0),
            test_switches=segs.get("test", {}).get("regime_switch_count", 0),
            full_switches=full_hist.get("regime_switch_count", 0),
            train_vol=segs.get("train", {}).get("volatility", {}).get("annualized", 0),
            valid_vol=segs.get("valid", {}).get("volatility", {}).get("annualized", 0),
            test_vol=segs.get("test", {}).get("volatility", {}).get("annualized", 0),
            full_vol=full_hist.get("volatility", {}).get("annualized", 0),
            train_dd=segs.get("train", {}).get("drawdown_stats", {}).get("major_dd_count", 0),
            valid_dd=segs.get("valid", {}).get("drawdown_stats", {}).get("major_dd_count", 0),
            test_dd=segs.get("test", {}).get("drawdown_stats", {}).get("major_dd_count", 0),
            full_dd=full_hist.get("drawdown_stats", {}).get("major_dd_count", 0),
            train_cumret=segs.get("train", {}).get("cum_return", 0),
            valid_cumret=segs.get("valid", {}).get("cum_return", 0),
            test_cumret=segs.get("test", {}).get("cum_return", 0),
            vol_ratio=cross.get("train_vs_test_vol_ratio", 1.0),
            ks_stat=cross.get("train_vs_test_ks_statistic", 0.0),
            tv_change=cross.get("train_to_valid_boundary", {}).get("changed", False),
            vt_change=cross.get("valid_to_test_boundary", {}).get("changed", False),
            coverage_pct=cross.get("regime_coverage_pct", 0),
            missing_regimes=", ".join(cross.get("missing_regimes", [])[:5]) or "none",
            findings_text=findings_text,
            stability=sd.get("stability_score", 1.0),
            trend=sd.get("trend", "stable"),
            cliff_count=len(sd.get("cliff_edges", [])),
        )

        # Resolve config from workspace (same path as rest of Critic pipeline)
        ws_config = self._llm._load_workspace_llm_config(self._workspace_root)
        effective_key = (api_key
                         or self._llm._resolve_effective_api_key(self._workspace_root))
        effective_base_url = base_url or ws_config.get("base_url")
        effective_model = model or ws_config.get("critic_model") or "gpt-4"

        # Try LLM call
        try:
            result = self._llm._call_llm_json_object(
                system_prompt=DIAGNOSIS_SYSTEM,
                user_prompt=user_prompt,
                model=effective_model,
                api_key=effective_key,
                base_url=effective_base_url,
                temperature=temperature,
                label="Window Diagnosis",
            )
            if result and isinstance(result, dict):
                return {
                    "problems": result.get("problems", []),
                    "root_cause": result.get("root_cause", "Unable to determine"),
                    "urgency": result.get("urgency", "watch"),
                }
        except Exception as e:
            logger.warning("Window diagnosis LLM call failed: %s", e)

        # Fallback: rule-based diagnosis from findings
        return self._fallback_diagnose(window_findings)

    # ------------------------------------------------------------------
    # Task B: Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        benchmark_data: dict,
        diagnosis: Optional[dict] = None,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
    ) -> dict:
        """Recommend the best window config from what-if candidates.

        Args:
            benchmark_data: Output from BenchmarkDataLoader.
            diagnosis: Output from diagnose() (optional, for context).
            model, api_key, base_url, temperature: LLM config.

        Returns:
            dict with keys: recommended_config, rationale, tradeoffs, alternatives.
        """
        wi = benchmark_data.get("what_if", {})
        cw = benchmark_data.get("current_window", {})
        config = cw.get("config", {})

        candidates = wi.get("top_candidates", [])
        if not candidates:
            return {
                "recommended_config": {
                    "train": config.get("train", 5),
                    "valid": config.get("valid", 2),
                    "test": config.get("test", 2),
                },
                "rationale": "No candidates available for comparison.",
                "tradeoffs": [],
                "alternatives": [],
            }

        # Build candidates table
        current_cfg = f"{config.get('train', '?')}/{config.get('valid', '?')}/{config.get('test', '?')}"
        rows = []
        for i, c in enumerate(candidates[:15]):
            cfg = c["config"]
            scores = c["quality_scores"]
            cur_marker = " ← CURRENT" if c.get("is_current") else ""
            rows.append(
                f"| {i+1} | {cfg['train']} | {cfg['valid']} | {cfg['test']} | "
                f"{scores['coverage']:.2f} | {scores['similarity']:.2f} | "
                f"{scores['recency']:.2f} | {scores['boundary_quality']:.2f} | "
                f"{scores['stability']:.2f} | {scores['composite']:.3f}{cur_marker} |"
            )
        candidates_table = "\n".join(rows)

        # Current composite score
        cur_composite = 0
        for c in candidates:
            if c.get("is_current"):
                cur_composite = c["quality_scores"]["composite"]
                break

        # Diagnosis context
        diag = diagnosis or {}
        problems = diag.get("problems", [])
        problems_summary = "; ".join(
            f"{p.get('type', '?')} ({p.get('severity', '?')})"
            for p in problems[:5]
        ) if problems else "none"

        # Regime summary
        regime_summary = benchmark_data.get("regime_summary", {})
        n_regimes = len(regime_summary.get("unique_regimes", []))
        cur_coverage = cw.get("cross_segment", {}).get("regime_coverage_pct", 0)

        user_prompt = RECOMMENDATION_USER_TEMPLATE.format(
            cur_train=config.get("train", "?"),
            cur_valid=config.get("valid", "?"),
            cur_test=config.get("test", "?"),
            cur_composite=f"{cur_composite:.3f}",
            candidates_table=candidates_table,
            root_cause=diag.get("root_cause", "unknown"),
            urgency=diag.get("urgency", "watch"),
            problems_summary=problems_summary,
            n_regimes=n_regimes,
            cur_coverage=cur_coverage,
        )

        # Resolve config from workspace (same path as rest of Critic pipeline)
        ws_config = self._llm._load_workspace_llm_config(self._workspace_root)
        effective_key = (api_key
                         or self._llm._resolve_effective_api_key(self._workspace_root))
        effective_base_url = base_url or ws_config.get("base_url")
        effective_model = model or ws_config.get("critic_model") or "gpt-4"

        # Try LLM call
        try:
            result = self._llm._call_llm_json_object(
                system_prompt=RECOMMENDATION_SYSTEM,
                user_prompt=user_prompt,
                model=effective_model,
                api_key=effective_key,
                base_url=effective_base_url,
                temperature=temperature,
                label="Window Recommendation",
            )
            if result and isinstance(result, dict) and "recommended_config" in result:
                return {
                    "recommended_config": result.get("recommended_config", {}),
                    "rationale": result.get("rationale", ""),
                    "tradeoffs": result.get("tradeoffs", []),
                    "alternatives": result.get("alternatives", []),
                }
        except Exception as e:
            logger.warning("Window recommendation LLM call failed: %s", e)

        # Fallback: return top candidate
        return self._fallback_recommend(candidates)

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _fallback_diagnose(self, findings: list) -> dict:
        """Rule-based fallback diagnosis when LLM is unavailable."""
        problems = []
        for f in findings:
            f_dict = f if isinstance(f, dict) else f.to_dict() if hasattr(f, 'to_dict') else {}
            f_type = f_dict.get("finding_type", "")
            severity = f_dict.get("severity", "warning")
            context = f_dict.get("context", "")
            if f_type:
                problems.append({
                    "type": f_type,
                    "severity": severity,
                    "evidence": context[:200],
                    "implicated_segment": "cross",
                })

        # Determine root cause
        root_causes = {
            "regime_coverage_gap": "Training window misses key market regimes",
            "boundary_regime_mismatch": "Window boundaries cross regime transitions",
            "volatility_regime_shift": "Test volatility differs significantly from training",
            "insufficient_drawdown_coverage": "Training lacks major drawdown experience",
            "impending_regime_loss": "A regime is about to drop from the training window",
        }
        root_cause = "No critical issues detected"
        urgency = "watch"
        for p in problems:
            if p["severity"] == "critical":
                root_cause = root_causes.get(p["type"], p["type"])
                urgency = "now"
                break
        if urgency != "now":
            for p in problems:
                if p["severity"] == "warning":
                    root_cause = root_causes.get(p["type"], p["type"])
                    urgency = "soon"
                    break

        return {"problems": problems, "root_cause": root_cause, "urgency": urgency}

    def _fallback_recommend(self, candidates: list) -> dict:
        """Return the top candidate as fallback recommendation."""
        if not candidates:
            return {
                "recommended_config": {"train": 5, "valid": 2, "test": 2},
                "rationale": "No candidates available.",
                "tradeoffs": [],
                "alternatives": [],
            }
        top = candidates[0]
        cfg = top["config"]
        scores = top["quality_scores"]
        return {
            "recommended_config": {"train": cfg["train"], "valid": cfg["valid"], "test": cfg["test"]},
            "rationale": (
                f"Top composite score ({scores['composite']:.3f}) with coverage={scores['coverage']:.2f}, "
                f"similarity={scores['similarity']:.2f}, recency={scores['recency']:.2f}."
            ),
            "tradeoffs": [{"gains": "Highest composite quality score", "costs": "Not validated by LLM"}],
            "alternatives": [
                {"config": alt["config"], "when_to_use": alt.get("strength", "alternative")}
                for alt in candidates[1:4]
            ],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_findings(findings: list) -> str:
        """Format findings into a readable text block."""
        if not findings:
            return "  (none)"

        lines = []
        for f in findings:
            f_dict = f if isinstance(f, dict) else f.to_dict() if hasattr(f, 'to_dict') else {}
            f_type = f_dict.get("finding_type", "?")
            severity = f_dict.get("severity", "?")
            context = f_dict.get("context", "")[:150]
            lines.append(f"  - [{severity}] {f_type}: {context}")
        return "\n".join(lines)
