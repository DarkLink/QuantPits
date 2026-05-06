"""
LLM Interface for the MAS Deep Analysis System.

Provides OpenAI API integration for:
1. Executive summary synthesis (existing)
2. Critic mode: generating ActionItems from structured Signals (Phase 3)

Falls back to template-based generation if no API key or API failure.
Skills and system prompts are loaded from workspace config files.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from .action_items import ActionItem
from .signal_extractor import Signal

logger = logging.getLogger(__name__)

# Default system prompt — used when config/skills/summary_system.md is absent
_DEFAULT_SUMMARY_PROMPT = """\
You are an expert quantitative finance analyst reviewing a systematic 
trading strategy's live performance. You are given structured findings from a multi-agent 
analysis system that covers: market regime, model health (IC/ICIR), ensemble composition, 
execution quality (slippage/friction), portfolio risk (factor exposure, drawdowns), 
prediction accuracy (hit rates), and trade behavior patterns.

Your task is to write a concise, insightful executive summary that:
1. Highlights the 2-3 most important findings
2. Explains the ROOT CAUSE behind observed patterns (not just symptoms)
3. Connects cross-domain observations (e.g., market regime → model performance → execution)
4. Provides actionable, specific recommendations ranked by priority
5. Uses professional but accessible language

Write in English. Be direct and data-driven. Avoid generic advice.
Keep the summary under 500 words."""

_DEFAULT_CRITIC_PROMPT = """\
You are a quantitative strategy optimization expert. Based on structured signals, \
output a JSON array of ActionItem objects with action_type, scope, target, params, \
reason, source_signals, expected_outcome, confidence, and risk_level."""

_DEFAULT_TRIAGE_PROMPT = """\
You are a quantitative strategy triage specialist. Your job is to prioritize which \
models need intervention, NOT to decide what the intervention should be.

Given aggregated signals across many models, you must:
1. Identify systemic patterns (e.g., "all models share the same early_stop default")
2. Prioritize at most 5 models that most need attention
3. Flag models that should be excluded (already had same intervention recently, \
   signal too weak, or wait for prior experiment results)
4. For each prioritized model, suggest an investigation direction (NOT a specific \
   parameter change — that's the Critic's job)

Key principles:
- Variety matters: if 12 models all have the same signal, pick the 2-3 WORST ones, \
  not all 12. The others can wait for the next cycle.
- History-aware: if a model was already adjusted for the same issue recently, \
  exclude it unless the situation has significantly worsened.
- Architecture-aware: NN models and tree models need different approaches. \
  Don't suggest the same fix for different architectures.
- Be specific in your rationale: mention exact epoch ratios, IC values, and trends.

Output a JSON object (not an array) with this structure:
{
  "systemic_observations": ["observation 1", "observation 2"],
  "prioritized_targets": [
    {
      "target": "model_name",
      "priority_score": 0-10,
      "primary_signal": "signal_type",
      "investigation_direction": "what to investigate (not exact params)",
      "rationale": "why this model was chosen over others"
    }
  ],
  "excluded_targets": [
    {
      "target": "model_name",
      "reason": "why excluded"
    }
  ]
}"""


class LLMInterface:
    """
    LLM integration layer for synthesis and critic decision-making.

    Supports OpenAI-compatible APIs with template-based fallback.
    """

    # Keep class-level reference for backward compatibility in existing tests
    SYSTEM_PROMPT = _DEFAULT_SUMMARY_PROMPT

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model
        self.base_url = base_url
        self._client = None

    def is_available(self) -> bool:
        """Check if LLM backend is available."""
        return bool(self.api_key)

    def _get_client(self):
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            try:
                import openai
                kwargs = {'api_key': self.api_key}
                if self.base_url:
                    kwargs['base_url'] = self.base_url
                self._client = openai.OpenAI(**kwargs)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    # ------------------------------------------------------------------
    # Executive Summary (existing, with optional workspace skill loading)
    # ------------------------------------------------------------------

    def generate_executive_summary(self, synthesis_result: dict,
                                   workspace_root: Optional[str] = None) -> str:
        """
        Generate an executive summary from synthesis results.

        Args:
            synthesis_result: Output from Synthesizer.synthesize()
            workspace_root: Optional workspace path for loading summary_system.md

        Returns:
            Formatted executive summary string
        """
        if not self.is_available():
            return self._template_summary(synthesis_result)

        try:
            return self._llm_summary(synthesis_result, workspace_root)
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}. Falling back to template.")
            return self._template_summary(synthesis_result)

    def _llm_summary(self, synthesis_result: dict,
                     workspace_root: Optional[str] = None) -> str:
        """Generate summary using OpenAI API, reading workspace config when available."""
        # Load system prompt from workspace skill file (fallback to default)
        system_prompt = self._load_summary_prompt(workspace_root)

        # Build user prompt from structured data
        user_prompt = self._build_prompt(synthesis_result)

        # Resolve model/endpoint: CLI args > workspace config > built-in defaults
        model = self.model or "gpt-4"
        client = self._get_client()

        if workspace_root:
            ws_config = self._load_workspace_llm_config(workspace_root)
            if ws_config:
                model = self.model or ws_config.get("summary_model") or "gpt-4"
                base_url = self.base_url or ws_config.get("base_url")
                api_key_env = ws_config.get("api_key_env", "OPENAI_API_KEY")
                api_key = self.api_key or os.environ.get(api_key_env, "")

                print(f"   Summary model: {model}")
                print(f"   Endpoint: {base_url or '(OpenAI default)'}")

                # Create a dedicated client when config overrides are active
                try:
                    import openai as _openai
                except ImportError:
                    raise RuntimeError("openai package not installed. Run: pip install openai")
                _kwargs = {"api_key": api_key}
                if base_url:
                    _kwargs["base_url"] = base_url
                client = _openai.OpenAI(**_kwargs)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        return response.choices[0].message.content.strip()

    def _load_summary_prompt(self, workspace_root: Optional[str]) -> str:
        """Load summary system prompt from workspace, fallback to default."""
        if workspace_root:
            path = os.path.join(workspace_root, "config", "skills", "summary_system.md")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        return content
                except Exception as e:
                    logger.warning("Failed to load summary_system.md: %s", e)
            else:
                logger.warning(
                    "summary_system.md not found at %s — using default prompt", path
                )
        return _DEFAULT_SUMMARY_PROMPT

    def _build_prompt(self, synthesis_result: dict) -> str:
        """Build a structured prompt from synthesis results."""
        parts = []

        # Health status
        parts.append(f"## System Health: {synthesis_result.get('health_status', 'Unknown')}")

        # Executive summary data
        exec_data = synthesis_result.get('executive_summary_data', {})
        parts.append(f"\n## Analysis Scope")
        parts.append(f"- Windows analyzed: {exec_data.get('windows_analyzed', [])}")
        parts.append(f"- Agents: {exec_data.get('agents_run', [])}")
        parts.append(f"- Critical findings: {exec_data.get('critical_count', 0)}")
        parts.append(f"- Warnings: {exec_data.get('warning_count', 0)}")
        parts.append(f"- Positive findings: {exec_data.get('positive_count', 0)}")

        if exec_data.get('market_regime'):
            parts.append(f"- Market regime: {exec_data['market_regime']}")
        if exec_data.get('cagr_1y') is not None:
            parts.append(f"- 1Y CAGR: {exec_data['cagr_1y']*100:.2f}%")
        if exec_data.get('sharpe_1y') is not None:
            parts.append(f"- 1Y Sharpe: {exec_data['sharpe_1y']:.3f}")

        # Cross-agent findings
        cross = synthesis_result.get('cross_findings', [])
        if cross:
            parts.append("\n## Cross-Agent Findings")
            for f in cross:
                parts.append(f"- [{f.severity.upper()}] {f.title}: {f.detail}")

        # Recommendations
        recs = synthesis_result.get('recommendations', [])
        if recs:
            parts.append("\n## Recommendations (by priority)")
            for r in recs[:10]:  # Limit to 10
                parts.append(f"- [{r['priority']}] ({r['source']}): {r['text']}")

        # Change events
        changes = synthesis_result.get('change_impact', [])
        if changes:
            parts.append(f"\n## Change Events: {len(changes)} detected")
            for c in changes[:5]:
                event = c.get('event', {})
                parts.append(f"- {event.get('type', '?')}: {event.get('date', '?')} — "
                           f"{event.get('model', event.get('combo', '?'))}")

        # External notes
        notes = synthesis_result.get('external_notes', '')
        if notes:
            parts.append(f"\n## Operator Notes\n{notes}")

        parts.append("\n---\nPlease write a concise executive summary of the above findings.")

        return "\n".join(parts)

    def _template_summary(self, synthesis_result: dict) -> str:
        """Generate template-based summary without LLM."""
        exec_data = synthesis_result.get('executive_summary_data', {})
        health = synthesis_result.get('health_status', 'Unknown')

        lines = [
            f"**System Health:** {health}",
            "",
            f"Analysis covered {len(exec_data.get('windows_analyzed', []))} time windows "
            f"using {len(exec_data.get('agents_run', []))} specialist agents. "
            f"Found {exec_data.get('critical_count', 0)} critical issues, "
            f"{exec_data.get('warning_count', 0)} warnings, and "
            f"{exec_data.get('positive_count', 0)} positive indicators.",
        ]

        if exec_data.get('market_regime'):
            lines.append(f"\nMarket regime: **{exec_data['market_regime']}**.")

        if exec_data.get('cagr_1y') is not None:
            lines.append(
                f"1-year performance: CAGR {exec_data['cagr_1y']*100:.2f}%, "
                f"Sharpe {exec_data.get('sharpe_1y', 0):.3f}."
            )

        # Key cross-findings
        cross = synthesis_result.get('cross_findings', [])
        if cross:
            lines.append("\n**Key Cross-Agent Findings:**")
            for f in cross[:3]:
                icon = "🔴" if f.severity == 'critical' else "🟡" if f.severity == 'warning' else "🟢"
                lines.append(f"- {icon} {f.title}")

        # Top recommendations
        recs = synthesis_result.get('recommendations', [])
        if recs:
            lines.append("\n**Priority Actions:**")
            for r in recs[:5]:
                lines.append(f"- **[{r['priority']}]** {r['text']}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Critic Mode (Phase 3) — ActionItem generation
    # ------------------------------------------------------------------

    def generate_action_items(
        self,
        signals: List[Signal],
        workspace_root: str,
    ) -> List[ActionItem]:
        """
        Two-stage Critic: Triage → Focused ActionItem generation.

        Stage 1 (Triage): Group signals, check history, prioritize max 5 models.
        Stage 2 (Focused Critic): Deep per-model analysis for prioritized targets.

        Falls back to single-stage when triage_system.md is absent or signals <= 5.

        Args:
            signals: List of Signals from SignalExtractor.
            workspace_root: Workspace path for config loading.

        Returns:
            List of ActionItem (unvalidated — caller should run Validator).
        """
        if not signals:
            logger.info("No signals to process — skipping Critic.")
            return []

        # Load workspace LLM config
        ws_config = self._load_workspace_llm_config(workspace_root)

        # Resolve effective settings: CLI args > config file > built-in defaults
        critic_model = ws_config.get("critic_model") or self.model or "gpt-4"
        temperature = ws_config.get("temperature", 0.3)
        max_tokens = ws_config.get("max_tokens", 4000)
        base_url = self.base_url or ws_config.get("base_url")
        api_key_env = ws_config.get("api_key_env", "OPENAI_API_KEY")

        # Resolve API key: CLI arg > env var named by config
        api_key = self.api_key or os.environ.get(api_key_env, "")

        # Diagnostics to stdout so the user can verify config loading
        print(f"   Critic model: {critic_model}")
        print(f"   Endpoint: {base_url or '(OpenAI default)'}")
        print(f"   API key env: {api_key_env} — {'found' if api_key else 'NOT FOUND'}")
        if not api_key:
            print("   ⚠️  No API key available. Set the env var or use --api-key.")
            return []

        # Load active scopes
        active_scopes = self._load_active_scopes(workspace_root)

        # Load hyperparam bounds (if available) so the LLM knows the limits
        hyperparam_bounds = self._load_hyperparam_bounds(workspace_root)

        # Load current hyperparameter values so the LLM has accurate 'from' values
        current_params = self._load_current_params(workspace_root, signals)

        # Load recent action history for dedup
        recent_history = self._load_recent_action_history(workspace_root, limit=20)

        # --- Decide: two-stage or single-stage ---
        triage_skill = self._load_triage_skill(workspace_root)
        use_triage = (
            triage_skill is not None
            and len(signals) > 5
        )

        if use_triage:
            print("   Mode: two-stage (Triage → Focused Critic)")
            combo_membership = self._load_combo_membership(workspace_root)
            prioritized_signals = self._run_triage(
                signals=signals,
                triage_skill=triage_skill,
                recent_history=recent_history,
                active_scopes=active_scopes,
                current_params=current_params,
                combo_membership=combo_membership,
                model=critic_model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
            )
            if prioritized_signals is None:
                # Triage failed — fall back to all signals with history context
                print("   ⚠️  Triage failed, falling back to single-stage with history.")
                prioritized_signals = signals
        else:
            if len(signals) <= 5:
                print("   Mode: single-stage (≤5 signals, skipping triage)")
            else:
                print("   Mode: single-stage (triage_system.md not found)")
            prioritized_signals = signals

        # Build system prompt from skills
        system_prompt = self._load_skills(workspace_root, active_scopes)

        # Build user prompt with only prioritized signals + history context
        user_prompt = self._build_critic_prompt(
            prioritized_signals, active_scopes, hyperparam_bounds, current_params,
            recent_history=recent_history,
        )

        # Call LLM
        try:
            items = self._call_critic_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=critic_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url,
            )
            if not items:
                print("   ⚠️  Critic returned 0 ActionItems "
                      f"(from {len(prioritized_signals)} signals)")
            return items
        except Exception as e:
            logger.error("Critic LLM call failed: %s", e)
            print(f"   ❌ Critic LLM call failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Critic internals
    # ------------------------------------------------------------------

    def _load_workspace_llm_config(self, workspace_root: str) -> dict:
        """Load llm_config.json from workspace, returning empty dict on failure."""
        path = os.path.join(workspace_root, "config", "llm_config.json")
        if not os.path.exists(path):
            logger.warning("llm_config.json not found at %s — using defaults", path)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load llm_config.json: %s", e)
            return {}

    def _load_active_scopes(self, workspace_root: str) -> List[str]:
        """Load active_scopes from feedback_scope.json."""
        path = os.path.join(workspace_root, "config", "feedback_scope.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("active_scopes", [])
        except Exception:
            return []

    def _load_skills(self, workspace_root: str, active_scopes: List[str]) -> str:
        """
        Load and concatenate skill files from config/skills/.

        Always loads critic_system.md first, then scope-matching skills.
        """
        skills_dir = os.path.join(workspace_root, "config", "skills")
        parts = []

        # 1. Always load critic_system.md
        critic_path = os.path.join(skills_dir, "critic_system.md")
        if os.path.exists(critic_path):
            try:
                with open(critic_path, "r", encoding="utf-8") as f:
                    parts.append(f.read().strip())
            except Exception as e:
                logger.warning("Failed to load critic_system.md: %s", e)
                parts.append(_DEFAULT_CRITIC_PROMPT)
        else:
            logger.warning(
                "critic_system.md not found at %s — using default", critic_path
            )
            parts.append(_DEFAULT_CRITIC_PROMPT)

        # 2. Load scope-matching skills
        scope_skill_map = {
            "hyperparams": "hyperparam_tuning.md",
            "model_selection": "model_selection.md",
            "combo_search": "combo_search.md",
            "strategy_params": "strategy_params.md",
        }

        for scope in active_scopes:
            skill_file = scope_skill_map.get(scope)
            if skill_file:
                skill_path = os.path.join(skills_dir, skill_file)
                if os.path.exists(skill_path):
                    try:
                        with open(skill_path, "r", encoding="utf-8") as f:
                            parts.append(f.read().strip())
                    except Exception as e:
                        logger.warning("Failed to load skill %s: %s", skill_file, e)

        return "\n\n---\n\n".join(parts)

    def _load_triage_skill(self, workspace_root: str) -> Optional[str]:
        """Load triage_system.md from workspace skills. Returns None if absent."""
        path = os.path.join(
            workspace_root, "config", "skills", "triage_system.md",
        )
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    return content
            except Exception as e:
                logger.warning("Failed to load triage_system.md: %s", e)
        return None

    def _load_combo_membership(self, workspace_root: str) -> Dict[str, List[str]]:
        """Load ensemble_config.json and return {model_name: [combo_names]}.

        Only includes models that are members of at least one active combo.
        """
        path = os.path.join(workspace_root, "config", "ensemble_config.json")
        if not os.path.exists(path):
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            return {}

        combos = config.get("combos", {})
        default_combo = config.get("default_combo", "")

        membership: Dict[str, List[str]] = {}
        for combo_name, combo_info in combos.items():
            if not isinstance(combo_info, dict):
                continue
            models = combo_info.get("models", [])
            for m in models:
                if isinstance(m, str):
                    membership.setdefault(m, []).append(combo_name)

        return membership

    def _load_recent_action_history(
        self, workspace_root: str, limit: int = 20,
    ) -> List[dict]:
        """Load recent ActionItems from data/action_item_history.jsonl.

        Returns the most recent ``limit`` entries as a list of dicts,
        newest first.  Returns an empty list if the file does not exist.
        """
        path = os.path.join(workspace_root, "data", "action_item_history.jsonl")
        if not os.path.exists(path):
            return []

        entries: List[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning("Failed to read action_item_history: %s", e)
            return []

        # Return newest first (assumes chronological append; reverse for safety)
        entries.reverse()
        return entries[:limit]

    def _aggregate_signals_for_triage(
        self, signals: List[Signal],
    ) -> dict:
        """Group signals by type for a compact triage summary."""
        by_type: Dict[str, List[dict]] = {}
        for s in signals:
            by_type.setdefault(s.signal_type, []).append({
                "target": s.target,
                "severity": s.severity,
                "context": s.context,
                "metrics": s.metrics,
            })

        return {
            signal_type: {
                "count": len(items),
                "models": [item["target"] for item in items],
                "sample_contexts": [
                    item["context"] for item in items[:3]
                ],  # first 3 only
            }
            for signal_type, items in by_type.items()
        }

    def _compute_available_interventions(
        self,
        in_scope_signals: List[Signal],
        recent_history: List[dict],
        current_params: dict,
    ) -> Dict[str, dict]:
        """For each model with in-scope signals, compute which params have been
        recently adjusted vs which are still untouched — purely rule-based.

        Returns {model_name: {recently_adjusted: [...], untouched: [...], exhausted: bool}}
        Exhausted means every known param was already adjusted within 30 days.
        """
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        adjusted: Dict[str, set] = {}
        for entry in recent_history:
            date = (
                entry.get("_run_date", "")
                or entry.get("run_date", "")
                or entry.get("timestamp", "")
            )
            target = entry.get("target", "")
            params = entry.get("params", {})
            if date and date >= cutoff and target and params:
                adjusted.setdefault(target, set()).update(params.keys())

        result: Dict[str, dict] = {}
        seen: set = set()
        for s in in_scope_signals:
            model = s.target
            if model in seen:
                continue
            seen.add(model)

            model_params = set(current_params.get(model, {}).keys())
            adj = adjusted.get(model, set()) & model_params
            untouched = sorted(model_params - adj)
            recently = sorted(adj)

            result[model] = {
                "recently_adjusted": recently,
                "untouched": untouched,
                "exhausted": len(untouched) == 0,
            }

        return result

    def _run_triage(
        self,
        signals: List[Signal],
        triage_skill: str,
        recent_history: List[dict],
        active_scopes: List[str],
        current_params: dict,
        combo_membership: Dict[str, List[str]],
        model: str,
        temperature: float,
        api_key: str,
        base_url: Optional[str],
    ) -> Optional[List[Signal]]:
        """Run Stage 1 triage. Returns filtered signals or None on failure.

        Only signals whose scope is in *active_scopes* are eligible for
        prioritization.  Out-of-scope signals are counted and noted but
        the triage LLM is instructed not to prioritize them.
        """
        active_scope_set = set(active_scopes)

        # Split signals: only in-scope signals can be acted on
        in_scope = [s for s in signals if s.scope in active_scope_set]
        out_of_scope = [s for s in signals if s.scope not in active_scope_set]

        if not in_scope:
            print(
                f"   ⚠️  Triage: all {len(signals)} signals are outside "
                f"active_scopes={active_scopes}. Skipping triage."
            )
            return signals

        # Rule-based: compute which params are still available per model
        interventions = self._compute_available_interventions(
            in_scope, recent_history, current_params,
        )

        # Build history summary for the prompt
        history_summary = []
        for entry in recent_history[:15]:
            history_summary.append({
                "date": entry.get("_run_date", entry.get("run_date", entry.get("timestamp", ""))),
                "target": entry.get("target", ""),
                "action_type": entry.get("action_type", ""),
                "params": entry.get("params", {}),
                "status": entry.get("scope_status", entry.get("status", "")),
            })

        # Build compact signal type summary from IN-SCOPE signals only
        signal_summary = self._aggregate_signals_for_triage(in_scope)

        # Count out-of-scope signals by type for awareness
        oos_counts: Dict[str, int] = {}
        for s in out_of_scope:
            oos_counts[s.signal_type] = oos_counts.get(s.signal_type, 0) + 1

        # Note any params that appear to be global defaults
        param_value_counts: Dict[str, Dict[str, int]] = {}
        for model_name, params in current_params.items():
            for param, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    key = str(value)
                    param_value_counts.setdefault(param, {}).setdefault(
                        key, 0
                    )
                    param_value_counts[param][key] += 1

        global_defaults = []
        total_models = len(current_params)
        if total_models >= 5:
            for param, counts in param_value_counts.items():
                for value, count in counts.items():
                    if count >= total_models * 0.7:
                        global_defaults.append(
                            f"{param}={value} appears in {count}/{total_models} "
                            f"models — likely a global default, not per-model tuning"
                        )

        triage_prompt = self._build_triage_prompt(
            signal_summary=signal_summary,
            history_summary=history_summary,
            global_defaults=global_defaults,
            active_scopes=active_scopes,
            total_signals=len(in_scope),
            out_of_scope_summary=oos_counts,
            combo_membership=combo_membership,
            available_interventions=interventions,
        )

        try:
            triage_result = self._call_triage_llm(
                system_prompt=triage_skill,
                user_prompt=triage_prompt,
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
            )
        except Exception as e:
            logger.warning("Triage LLM call failed: %s", e)
            return None

        if triage_result is None:
            return None

        # Extract prioritized targets from triage result
        prioritized = triage_result.get("prioritized_targets", [])

        # Print systemic observations regardless
        systemic = triage_result.get("systemic_observations", [])
        for obs in systemic:
            print(f"   🔍 Systemic: {obs}")

        excluded = triage_result.get("excluded_targets", [])
        for exc in excluded:
            print(f"   ⏭️  Excluded: {exc.get('target', '?')} — {exc.get('reason', '?')}")

        if not prioritized:
            print(
                "   ⚠️  Triage returned no prioritized targets "
                f"(all {len(in_scope)} in-scope models recently adjusted or "
                "signals too weak). Falling back to in-scope signals."
            )
            return in_scope if in_scope else signals

        target_names = set()
        for entry in prioritized:
            target = entry.get("target", "")
            if target:
                target_names.add(target)

        for entry in prioritized:
            print(
                f"   ✅ Prioritized: {entry.get('target', '?')} "
                f"(score={entry.get('priority_score', '?')}, "
                f"{entry.get('primary_signal', '?')})"
            )

        # Filter signals to only prioritized targets
        filtered = [s for s in signals if s.target in target_names]
        print(f"   Signals: {len(signals)} → {len(filtered)} after triage")
        return filtered

    def _build_triage_prompt(
        self,
        signal_summary: dict,
        history_summary: List[dict],
        global_defaults: List[str],
        active_scopes: List[str],
        total_signals: int,
        out_of_scope_summary: Optional[Dict[str, int]] = None,
        combo_membership: Optional[Dict[str, List[str]]] = None,
        available_interventions: Optional[Dict[str, dict]] = None,
    ) -> str:
        """Build the user prompt for the Triage LLM call."""
        parts = [
            f"## Active Scopes (ONLY these can be acted on)\n{json.dumps(active_scopes)}",
        ]

        if combo_membership:
            combo_models: Dict[str, List[str]] = {}
            for model_name, combos in combo_membership.items():
                for c in combos:
                    combo_models.setdefault(c, []).append(model_name)

            parts.append(
                f"\n## Ensemble Combo Membership\n"
                f"Models in active combos have HIGHER IMPACT — prioritize them.\n"
                f"```json\n{json.dumps(combo_models, indent=2, ensure_ascii=False)}\n```"
            )

        if available_interventions:
            parts.append(
                f"\n## Per-Model Intervention Availability (RULE-BASED — authoritative)\n"
                f"For each model with in-scope signals, this shows:\n"
                f"- **recently_adjusted**: params already changed in the last 30 days. "
                f"Suggesting a DIFFERENT value for these SAME params should have a high bar "
                f"(signal significantly worsened since last change).\n"
                f"- **untouched**: params that have NOT been recently changed and are "
                f"SAFE TO EXPERIMENT WITH. These are your primary candidates.\n"
                f"- **exhausted**: true if ALL known params were already adjusted. "
                f"Only exclude these models — partially explored models should still "
                f"be prioritized as long as there are untouched params.\n\n"
                f"A model with ic_decay + 3 untouched params should NOT be excluded "
                f"just because it had 1-2 other params adjusted recently.\n"
                f"```json\n{json.dumps(available_interventions, indent=2, ensure_ascii=False)}\n```"
            )

        parts.append(
            f"\n## Actionable Signals ({total_signals} in-scope)\n"
            f"```json\n{json.dumps(signal_summary, indent=2, ensure_ascii=False)}\n```"
        )

        if out_of_scope_summary:
            oos_lines = [
                f"  {stype}: {count} signals" for stype, count in
                sorted(out_of_scope_summary.items(), key=lambda x: -x[1])
            ]
            parts.append(
                f"\n## Out-of-Scope Signals (REPORT ONLY — DO NOT prioritize)\n"
                f"These signals belong to disabled scopes. You MUST NOT include "
                f"their targets in prioritized_targets.\n"
                + "\n".join(oos_lines)
            )

        if global_defaults:
            parts.append(
                f"\n## ⚠️ Potential Global Defaults\n"
                f"These parameter values appear in >70% of models and may be "
                f"system-wide defaults rather than per-model tuning. "
                f"Batch-adjusting these is rarely the right approach.\n"
                + "\n".join(f"- {d}" for d in global_defaults)
            )

        if history_summary:
            parts.append(
                f"\n## Recent Action Item History\n"
                f"These models were ALREADY adjusted recently. "
                f"Do NOT re-recommend the same change unless the model's "
                f"condition has significantly worsened.\n"
                f"```json\n{json.dumps(history_summary, indent=2, ensure_ascii=False)}\n```"
            )

        parts.append(
            f"\n## Instructions\n"
            f"1. Identify systemic patterns across these {total_signals} signals\n"
            f"2. Pick at most 5 models that most need intervention RIGHT NOW\n"
            f"3. Exclude models that already had the same fix applied recently\n"
            f"4. For global defaults, pick 2-3 worst-affected models to experiment on\n"
            f"5. Output JSON object with keys: systemic_observations, "
            f"prioritized_targets, excluded_targets"
        )

        return "\n".join(parts)

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        """Extract the first valid JSON object from text that may contain
        markdown fences, explanatory prose, or other surrounding content.

        Returns the JSON substring (including outer braces) or None.
        """
        if not text:
            return None

        # Try extracting from markdown code fences first (most common pattern)
        fence_patterns = [
            ("```json\n", "\n```"),
            ("```json", "```"),
            ("```\n", "\n```"),
            ("```", "```"),
        ]
        for start_fence, end_fence in fence_patterns:
            if start_fence in text:
                after_start = text.split(start_fence, 1)[1]
                if end_fence in after_start:
                    return after_start.split(end_fence, 1)[0].strip()

        # No fences found — try to find the outermost JSON object braces
        first_brace = text.find("{")
        if first_brace == -1:
            return None

        # Find matching closing brace
        depth = 0
        for i in range(first_brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[first_brace : i + 1]

        return None

    def _call_triage_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        api_key: str,
        base_url: Optional[str],
    ) -> Optional[dict]:
        """Call the LLM for triage and return parsed JSON dict."""
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        client = openai.OpenAI(**kwargs)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=16000,
            )

            choice = response.choices[0]
            finish = choice.finish_reason if hasattr(choice, "finish_reason") else "?"
            content = choice.message.content
            # Some reasoning models return reasoning_content and leave content empty
            reasoning = getattr(choice.message, "reasoning_content", None)

            if not content:
                # Some reasoning models put the final answer in reasoning_content
                # when max_tokens is exhausted before content can be generated.
                diag = (
                    f"   ⚠️  Triage returned empty content. "
                    f"finish_reason={finish}, "
                    f"has_reasoning={bool(reasoning)}, "
                    f"reasoning_len={len(reasoning) if reasoning else 0}"
                )
                print(diag)
                logger.warning(diag)
                # Fallback: try reasoning_content as the response
                if reasoning and finish == "length":
                    print("   🔄 Triage: using reasoning_content as fallback (token limit hit)")
                    json_str = self._extract_json_object(reasoning)
                    if json_str:
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                return None

            # Extract JSON from potentially noisy response
            json_str = self._extract_json_object(content)
            if json_str is None:
                logger.warning(
                    "Triage response contained no JSON object. "
                    "finish_reason=%s, raw (first 300 chars): %s",
                    finish, content[:300],
                )
                return None

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.warning(
                "Triage JSON parse failed: %s. Raw (first 300 chars): %s",
                e, content[:300] if 'content' in dir() else "N/A",
            )
            return None
        except Exception as e:
            logger.warning("Triage LLM call failed: %s", e)
            return None

    def _load_hyperparam_bounds(self, workspace_root: str) -> dict:
        """Load hyperparam_bounds from workspace config, returning empty dict on failure."""
        path = os.path.join(workspace_root, "config", "hyperparam_bounds.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("bounds", {})
        except Exception:
            return {}

    def _load_current_params(
        self, workspace_root: str, signals: List[Signal],
    ) -> dict:
        """Load current hyperparameter values for each model referenced in signals.

        Reads model_registry.yaml to find the YAML file for each model, then
        extracts task.model.kwargs.  Returns a dict keyed by model name so the
        LLM Critic can populate accurate ``from`` values in ActionItems.
        """
        import yaml

        # Collect unique model targets from signals
        model_names = set()
        for s in signals:
            if s.scope == "hyperparams" and s.target:
                model_names.add(s.target)

        if not model_names:
            return {}

        # Load model registry to map model → yaml_file
        registry_path = os.path.join(
            workspace_root, "config", "model_registry.yaml",
        )
        registry = {}
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = yaml.safe_load(f)
            except Exception:
                pass
        registry = registry.get("models", {}) if isinstance(registry, dict) else {}

        current = {}
        for model_name in sorted(model_names):
            model_info = registry.get(model_name, {})
            yaml_file = model_info.get("yaml_file", "")
            if not yaml_file:
                continue
            yaml_path = os.path.join(workspace_root, yaml_file)
            if not os.path.exists(yaml_path):
                continue
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                kwargs = (
                    cfg.get("task", {}).get("model", {}).get("kwargs", {})
                )
                # Filter to hyperparameter-relevant keys only
                clean = {}
                for k, v in kwargs.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        clean[k] = v
                if clean:
                    current[model_name] = clean
            except Exception:
                pass

        return current

    def _build_critic_prompt(
        self, signals: List[Signal], active_scopes: List[str],
        hyperparam_bounds: Optional[dict] = None,
        current_params: Optional[dict] = None,
        recent_history: Optional[List[dict]] = None,
    ) -> str:
        """Build the user prompt for the Critic LLM call."""
        signals_json = json.dumps(
            [s.to_dict() for s in signals],
            indent=2,
            ensure_ascii=False,
        )

        parts = [
            f"## Active Scopes\n{json.dumps(active_scopes)}",
        ]

        if hyperparam_bounds:
            # Include bounds so the LLM can generate valid suggestions
            bounds_summary = {
                k: {"min": v.get("min"), "max": v.get("max"),
                    "max_change_pct": v.get("max_change_pct")}
                for k, v in hyperparam_bounds.items()
            }
            parts.append(
                f"## Hyperparameter Bounds\n"
                f"Your suggestions must respect these limits. "
                f"`max_change_pct: null` means no percentage-change limit.\n"
                f"```json\n{json.dumps(bounds_summary, indent=2, ensure_ascii=False)}\n```"
            )

        if current_params:
            parts.append(
                f"## Current Hyperparameter Values\n"
                f"These are the CURRENT values for each model. "
                f"Use them EXACTLY as the `from` value in your `params` fields. "
                f"Do NOT guess or invent parameter values.\n"
                f"```json\n{json.dumps(current_params, indent=2, ensure_ascii=False)}\n```"
            )

        if recent_history:
            # Filter history to only targets present in the current signals
            signal_targets = {s.target for s in signals}
            relevant_history = [
                h for h in recent_history
                if h.get("target") in signal_targets
            ]
            if relevant_history:
                parts.append(
                    f"## Recent Action History for These Models\n"
                    f"These models were recently adjusted. Do NOT repeat the same "
                    f"change unless their condition has significantly worsened. "
                    f"If you believe a prior change was insufficient, explain why "
                    f"and suggest a DIFFERENT approach.\n"
                    f"```json\n{json.dumps(relevant_history[:10], indent=2, ensure_ascii=False)}\n```"
                )

        parts.append(
            f"## Signals\n```json\n{signals_json}\n```\n\n"
            f"Based on the above signals, generate a JSON array of ActionItems. "
            f"Generate 2-5 ActionItems — each targeting a different model, each "
            f"using a DIFFERENT untouched parameter. If a model has ic_decay and "
            f"several untouched params, you SHOULD suggest one of them. "
            f"Output ONLY the JSON array, no other text."
        )

        return "\n\n".join(parts)

    def _call_critic_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        base_url: Optional[str],
    ) -> List[ActionItem]:
        """Call the LLM API and parse the response into ActionItems."""
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        client = openai.OpenAI(**kwargs)

        max_retries = 2
        last_error = None
        last_content = None

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                content = response.choices[0].message.content
                if not content:
                    print("   ⚠️  Critic returned empty response")
                    return []
                last_content = content
                items = self._parse_action_items(content.strip())
                if not items:
                    print(
                        "   ⚠️  Critic parsed successfully but returned 0 ActionItems. "
                        f"Raw (first 300 chars): {content.strip()[:300]}"
                    )
                return items

            except json.JSONDecodeError as e:
                last_error = e
                print(
                    f"   ⚠️  Critic JSON parse failed (attempt {attempt+1}/{max_retries}). "
                    f"Raw (first 200 chars): {last_content[:200] if last_content else 'N/A'}"
                )
                logger.warning(
                    "Critic response JSON parse failed (attempt %d/%d): %s",
                    attempt + 1, max_retries, e,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "Critic LLM call failed (attempt %d/%d): %s",
                    attempt + 1, max_retries, e,
                )
                break  # Don't retry on non-parse errors

        logger.error("Critic failed after %d attempts: %s", max_retries, last_error)
        print(f"   ❌ Critic failed after {max_retries} attempts: {last_error}")
        return []

    @staticmethod
    def _parse_action_items(content: str) -> List[ActionItem]:
        """
        Parse LLM text output into ActionItem list.

        Handles JSON arrays, with or without markdown code fences.
        """
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = text.index("\n")
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()

        items_data = json.loads(text)

        if not isinstance(items_data, list):
            items_data = [items_data]

        return [ActionItem.from_dict(d) for d in items_data]
