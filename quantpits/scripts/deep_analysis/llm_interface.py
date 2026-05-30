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
from typing import Any, List, Dict, Optional

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
        self._client_cache: Dict[tuple, Any] = {}  # (api_key, base_url) → client
    # Default timeout for all LLM calls (seconds).
    # 120s is generous for reasoning models which can take 30-60s.
    _DEFAULT_TIMEOUT = 120.0

    def _get_or_create_client(self, api_key: str, base_url: Optional[str] = None):
        """Return a cached OpenAI client for the given (api_key, base_url) pair.

        Thread-safe: concurrent reads of the same key return the same client;
        concurrent first-writes may create a duplicate, but that is harmless.
        """
        try:
            import openai
            import httpx
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        cache_key = (api_key, base_url or "")
        client = self._client_cache.get(cache_key)
        if client is None:
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            kwargs["timeout"] = httpx.Timeout(
                self._DEFAULT_TIMEOUT, connect=10.0,
            )
            client = openai.OpenAI(**kwargs)
            self._client_cache[cache_key] = client
        return client

    def _resolve_effective_api_key(self, workspace_root: Optional[str] = None) -> Optional[str]:
        """Resolve API key: explicit arg > workspace config env var > OPENAI_API_KEY."""
        if self.api_key:
            return self.api_key
        if workspace_root:
            ws_config = self._load_workspace_llm_config(workspace_root)
            api_key_env = ws_config.get("api_key_env", "OPENAI_API_KEY")
            key = os.environ.get(api_key_env)
            if key:
                return key
        return os.environ.get("OPENAI_API_KEY")

    def is_available(self, workspace_root: Optional[str] = None) -> bool:
        """Check if LLM backend is available."""
        return bool(self._resolve_effective_api_key(workspace_root))

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
        if not self.is_available(workspace_root):
            return self._template_summary(synthesis_result)

        try:
            return self._llm_summary(synthesis_result, workspace_root)
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}. Falling back to template.")
            return self._template_summary(synthesis_result)

    def _llm_summary(self, synthesis_result: dict,
                     workspace_root: Optional[str] = None) -> str:
        """Generate summary using OpenAI API, reading workspace config when available."""
        system_prompt = self._load_summary_prompt(workspace_root)
        user_prompt = self._build_prompt(synthesis_result)

        # Resolve effective model/endpoint/api_key
        ws_config = self._load_workspace_llm_config(workspace_root) if workspace_root else {}
        model = self.model or ws_config.get("summary_model") or "gpt-4"
        base_url = self.base_url or ws_config.get("base_url")
        api_key = self._resolve_effective_api_key(workspace_root)

        if not api_key:
            raise RuntimeError("No API key available")

        print(f"   Summary model: {model}")
        print(f"   Endpoint: {base_url or '(OpenAI default)'}")

        # Only create a dedicated client when endpoint/key differs from defaults
        if base_url or api_key != self.api_key:
            try:
                import openai as _openai
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
            _kwargs = {"api_key": api_key}
            if base_url:
                _kwargs["base_url"] = base_url
            client = _openai.OpenAI(**_kwargs)
        else:
            client = self._get_client()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=8192,
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

        # Layered Critic Pipeline output (when available)
        critic_out = synthesis_result.get('_critic_pipeline_output')
        if critic_out:
            ai_list = critic_out.get('action_items', [])
            if ai_list:
                parts.append("\n## Critic Pipeline — Final ActionItems")
                parts.append(
                    "These ActionItems reflect the layered Triage → Per-Model → "
                    "Synthesizer pipeline. They are the authoritative diagnosis; "
                    "do NOT contradict them in the summary."
                )
                for ai in ai_list:
                    parts.append(
                        f"- [{ai.get('action_type', '?')}] {ai.get('target', '?')}: "
                        f"{ai.get('reason', '')}"
                    )
            conflicts = critic_out.get('conflict_resolutions', [])
            if conflicts:
                parts.append("\n## Critic Pipeline — Conflict Resolutions")
                for c in conflicts:
                    parts.append(
                        f"- {c.get('conflict', '')[:150]}\n"
                        f"  Resolution: {c.get('resolution', '')[:150]}"
                    )
            scopes = critic_out.get('scope_recommendations', [])
            if scopes:
                parts.append("\n## Critic Pipeline — Scope Recommendations")
                for s in scopes:
                    parts.append(f"- Open scope '{s.get('scope', '?')}': {s.get('reason', '')[:150]}")

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
        max_tokens = ws_config.get("max_tokens", 32768)
        base_url = self.base_url or ws_config.get("base_url")
        api_key_env = ws_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = self._resolve_effective_api_key(workspace_root)

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
                workspace_root=workspace_root,
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

    # Fallback set of non-tunable params — used only when hyperparam_bounds.json
    # is unavailable.  When bounds are loaded, tunable params are inferred as
    # the intersection of a model's current params with bounds.keys(), which is
    # always up-to-date.
    _FALLBACK_NON_TUNABLE = {
        "metric", "loss", "loss_type", "optimizer", "GPU", "device",
        "seed", "rnn_type", "d_feat", "class", "module_path", "estimator",
        "n_jobs", "kernel_type", "num_class",
    }

    @staticmethod
    def _load_tunable_param_names(workspace_root: Optional[str]) -> Optional[set]:
        """Load the set of tunable param names from hyperparam_bounds.json.

        Returns None if the file doesn't exist or can't be parsed, signalling
        the caller to fall back to _FALLBACK_NON_TUNABLE.
        """
        if not workspace_root:
            return None
        path = os.path.join(workspace_root, "config", "hyperparam_bounds.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            bounds = data.get("bounds", {})
            if bounds:
                return set(bounds.keys())
        except Exception:
            pass
        return None

    def _compute_available_interventions(
        self,
        in_scope_signals: List[Signal],
        recent_history: List[dict],
        current_params: dict,
        workspace_root: Optional[str] = None,
    ) -> Dict[str, dict]:
        """For each model with in-scope signals, compute which params have been
        recently adjusted vs which are still untouched — purely rule-based.

        Tunable params are determined from hyperparam_bounds.json when available;
        otherwise falls back to excluding _FALLBACK_NON_TUNABLE.

        Returns {model_name: {recently_adjusted: [...], untouched: [...], exhausted: bool}}
        Exhausted means every TUNABLE param was already adjusted within 30 days.
        """
        # Determine which params are tunable
        bounds_params = self._load_tunable_param_names(workspace_root)

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
            if bounds_params is not None:
                # Tunable = params that exist in both the model's config AND bounds
                tunable = model_params & bounds_params
            else:
                # Fallback: tunable = model params minus known non-tunable
                tunable = model_params - self._FALLBACK_NON_TUNABLE
            adj = adjusted.get(model, set()) & tunable
            untouched = sorted(tunable - adj)
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
        workspace_root: Optional[str] = None,
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
            workspace_root=workspace_root,
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
        client = self._get_or_create_client(api_key, base_url)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=32768,
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

    # ------------------------------------------------------------------
    # Layered Pipeline — triage → per-model → per-combo → synthesizer
    # ------------------------------------------------------------------

    def _call_llm_json_object(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 32768,
        label: str = "LLM",
    ) -> Optional[dict]:
        """Generic LLM call returning a parsed JSON object (not array)."""
        client = self._get_or_create_client(api_key, base_url)

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

            choice = response.choices[0]
            finish = getattr(choice, "finish_reason", "?")
            content = choice.message.content
            reasoning = getattr(choice.message, "reasoning_content", None)

            if not content:
                diag = (
                    f"   ⚠️  {label} returned empty content. "
                    f"finish_reason={finish}, "
                    f"has_reasoning={bool(reasoning)}, "
                    f"reasoning_len={len(reasoning) if reasoning else 0}"
                )
                print(diag)
                if reasoning and finish == "length":
                    print(f"   🔄 {label}: using reasoning_content as fallback")
                    json_str = self._extract_json_object(reasoning)
                    if json_str:
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                return None

            json_str = self._extract_json_object(content)
            if json_str is None:
                logger.warning(
                    "%s response contained no JSON object. finish_reason=%s, raw (300 chars): %s",
                    label, finish, content[:300],
                )
                return None

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.warning("%s JSON parse failed: %s", label, e)
            return None
        except Exception as e:
            logger.warning("%s call failed: %s", label, e)
            return None

    def _resolve_llm_config(self, workspace_root: str) -> dict:
        """Resolve effective LLM config: CLI args > workspace config > defaults."""
        ws_config = self._load_workspace_llm_config(workspace_root)
        return {
            "model": self.model or ws_config.get("critic_model") or "gpt-4",
            "temperature": ws_config.get("temperature", 0.3),
            "triage_temperature": ws_config.get(
                "triage_temperature",
                max(0.1, ws_config.get("temperature", 0.3) * 0.3)
            ),
            "base_url": self.base_url or ws_config.get("base_url"),
        }

    def _load_skill(self, workspace_root: str, filename: str) -> Optional[str]:
        """Load a single skill file. Returns None if not found."""
        path = os.path.join(workspace_root, "config", "skills", filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    return content
            except Exception as e:
                logger.warning("Failed to load skill %s: %s", filename, e)
        return None

    def generate_triage(
        self,
        triage_input: dict,
        signals: List[Signal],
        workspace_root: str,
    ) -> Optional[dict]:
        """
        Layered Triage — route models/combos/execution to appropriate downstream LLMs.

        Returns a dict with keys: prioritized_models, healthy_models, prioritized_combos,
        needs_execution_risk, systemic_observations.
        """
        cfg = self._resolve_llm_config(workspace_root)
        api_key = self._resolve_effective_api_key(workspace_root)
        if not api_key:
            print("   ⚠️  No API key available for Triage.")
            return None

        system_prompt = self._load_skill(workspace_root, "triage_system.md")
        if not system_prompt:
            print("   ⚠️  triage_system.md not found — cannot run Triage.")
            return None

        # Build compact signal summary
        signal_summary = self._aggregate_signals_for_triage(signals)

        # Build model summary table from triage_input
        model_table = []
        for m in triage_input.get("model_ranking", []):
            entry = {
                "model": m["model"],
                "ic_mean": m.get("ic_mean"),
                "icir_mean": m.get("icir_mean"),
                "ic_trend": m.get("ic_trend"),
                "family": m.get("family"),
                "in_combos": m.get("in_combos", []),
                "best_epoch": m.get("best_epoch"),
                "actual_epochs": m.get("actual_epochs"),
                "early_stopped": m.get("early_stopped"),
            }
            hf = m.get("historical_flags")
            if hf:
                entry["historical_flags"] = hf
            model_table.append(entry)

        n_flagged = sum(1 for e in model_table if "historical_flags" in e)
        if n_flagged:
            flagged_names = [e["model"] for e in model_table if "historical_flags" in e]
            print(f"   Triage: {n_flagged} models with historical flags: {', '.join(flagged_names[:8])}")

        family_stats = triage_input.get("family_stats", {})
        combo_summary = triage_input.get("combo_summary", {})
        market_ctx = triage_input.get("market_context", {})

        # Load history for dedup
        recent_history = self._load_recent_action_history(workspace_root, limit=20)
        history_summary = []
        for entry in recent_history[:15]:
            history_summary.append({
                "date": entry.get("_run_date", entry.get("run_date", entry.get("timestamp", ""))),
                "target": entry.get("target", ""),
                "action_type": entry.get("action_type", ""),
                "params": entry.get("params", {}),
                "executed": bool(entry.get("executed", False)),
            })

        user_prompt = json.dumps({
            "market_context": market_ctx,
            "model_ranking_table": model_table,
            "family_statistics": family_stats,
            "combo_summary": combo_summary,
            "signal_summary": signal_summary,
            "recent_action_history": history_summary,
            "instructions": (
                "Based on the above data, decide: "
                "1) Which models need Per-Model LLM deep analysis (prioritized_models, max 8)? "
                "2) Which models are healthy and can be skipped (healthy_models)? "
                "3) Which combos need Per-Combo LLM analysis (prioritized_combos, max 3)? "
                "4) Is Execution/Risk LLM needed (needs_execution_risk, boolean)? "
                "5) What systemic patterns do you observe (systemic_observations)? "
                "\n\nCRITICAL RULE — Historical Tracking: "
                "Any model with a historical_flags field MUST be routed to Per-Model, "
                "even if it has no new training data or current signals. "
                "Set tracking_mode: true for such models. "
                "No new data does NOT equal healthy — historical problems need follow-up. "
                "\n\nOutput a JSON object with keys: systemic_observations, prioritized_models, "
                "healthy_models, prioritized_combos, needs_execution_risk."
            ),
        }, indent=2, ensure_ascii=False)

        print(f"   Triage: analyzing {len(model_table)} models, {len(signals)} signals")
        result = self._call_llm_json_object(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=cfg["model"],
            api_key=api_key,
            base_url=cfg["base_url"],
            temperature=cfg["triage_temperature"],
            label="Triage",
        )

        if result:
            # --- Deterministic override: force historically flagged models in ---
            # LLMs can ignore CRITICAL RULE instructions, so we enforce this
            # outside the LLM call.
            flagged_models = {e["model"] for e in model_table if "historical_flags" in e}
            if flagged_models:
                existing = set()
                for item in result.get("prioritized_models", []):
                    name = item.get("target", item.get("model", ""))
                    if name:
                        existing.add(name)
                for item in result.get("healthy_models", []):
                    name = item if isinstance(item, str) else item.get("model", item.get("target", ""))
                    if name:
                        existing.add(name)

                missing = flagged_models - existing
                if missing:
                    prioritized = result.setdefault("prioritized_models", [])
                    for name in sorted(missing):
                        hf = next(
                            (e["historical_flags"] for e in model_table
                             if e.get("model") == name), {}
                        )
                        prioritized.append({
                            "target": name,
                            "priority_score": 7,
                            "primary_signal": "historical_tracking",
                            "tracking_mode": True,
                            "investigation_direction": (
                                f"History shows {hf.get('historical_signal_types', [])} "
                                f"in {hf.get('run_count', 0)} of last 3 runs"
                            ),
                            "rationale": "Deterministic override: historically flagged, "
                                         "must be re-evaluated.",
                        })
                    print(f"   Triage: +{len(missing)} historical override models "
                          f"({', '.join(sorted(missing))})")

            # --- Deterministic override: force combos with OOS degradation ---
            if not result.get("prioritized_combos"):
                oos_trend = combo_summary.get("_oos_trend", {})
                calmar_slope = oos_trend.get("calmar_slope")
                if calmar_slope is not None and calmar_slope < -0.3:
                    # OOS is degrading but LLM didn't route any combos — inject them
                    forced_combos = []
                    for combo_name, info in combo_summary.items():
                        if combo_name.startswith("_"):
                            continue  # skip meta keys
                        forced_combos.append({
                            "combo": combo_name,
                            "priority_score": 8,
                            "primary_signal": "oos_degradation",
                            "rationale": (
                                f"Deterministic override: OOS Calmar slope "
                                f"{calmar_slope:.3f}, combo '{combo_name}' "
                                f"needs Per-Combo diagnosis."
                            ),
                        })
                    if forced_combos:
                        result["prioritized_combos"] = forced_combos[:3]
                        print(f"   Triage: +{len(forced_combos[:3])} combo override "
                              f"(OOS slope={calmar_slope:.3f})")

            n_models = len(result.get("prioritized_models", []))
            n_combos = len(result.get("prioritized_combos", []))
            n_healthy = len(result.get("healthy_models", []))
            print(f"   Triage: {n_models} models → Per-Model, "
                  f"{n_combos} combos → Per-Combo, "
                  f"{n_healthy} healthy, "
                  f"exec_risk={result.get('needs_execution_risk', False)}")
        return result

    def generate_model_critique(
        self,
        model_name: str,
        model_profile: dict,
        workspace_root: str,
    ) -> Optional[dict]:
        """
        Layered Per-Model Critic — deep single-model diagnosis.

        Args:
            model_name: The model being analyzed.
            model_profile: Dict with keys: training_history, ranking_table, family_stats,
                correlation_excerpt, combo_role, signals, current_params, hyperparam_bounds.
            workspace_root: Workspace path for skill loading.

        Returns a dict with keys: diagnosis, diagnosis_detail, action_items, cross_references.
        """
        cfg = self._resolve_llm_config(workspace_root)
        api_key = self._resolve_effective_api_key(workspace_root)
        if not api_key:
            return None

        # Load skills: model_critic_system.md (primary) + scope skills
        critic_skill = self._load_skill(workspace_root, "model_critic_system.md")
        if not critic_skill:
            print(f"   ⚠️  model_critic_system.md not found — cannot run Per-Model Critic.")
            return None

        hyperparam_skill = self._load_skill(workspace_root, "hyperparam_tuning.md") or ""
        model_sel_skill = self._load_skill(workspace_root, "model_selection.md") or ""

        system_prompt = "\n\n---\n\n".join(
            p for p in [critic_skill, hyperparam_skill, model_sel_skill] if p
        )

        # Load active scopes
        active_scopes = self._load_active_scopes(workspace_root)

        # Inject tuning_knowledge only if non-empty (avoid prompt bloat)
        tuning_knowledge = model_profile.get("tuning_knowledge", {})

        user_prompt = json.dumps({
            "model_name": model_name,
            "active_scopes": active_scopes,
            "training_history": model_profile.get("training_history", []),
            "ranking_context": {
                "full_table": model_profile.get("ranking_table", []),
                "family_stats": model_profile.get("family_stats", {}),
            },
            "correlation_excerpt": model_profile.get("correlation_excerpt", {}),
            "combo_role": model_profile.get("combo_role", {}),
            "diversity_signals": model_profile.get("diversity_signals", {}),
            "tuning_knowledge": tuning_knowledge if tuning_knowledge else None,
            "signals": [s.to_dict() for s in model_profile.get("signals", [])],
            "current_params": model_profile.get("current_params", {}),
            "hyperparam_bounds": model_profile.get("hyperparam_bounds", {}),
            "instructions": (
                f"Diagnose {model_name} based on the profile above. "
                "Consider: training history, ranking context, correlation structure, "
                "combo role, diversity_signals (especially is_diversifier), and all signals. "
                "IMPORTANT: If diversity_signals.is_diversifier is true, this model's low IC "
                "may be acceptable — it provides orthogonal diversification. Do NOT recommend "
                "disabling without clear evidence (e.g. negative LOO delta). "
                + (
                    "TUNING KNOWLEDGE: This model has accumulated experiment knowledge in "
                    "tuning_knowledge. You MUST respect: "
                    "(1) known_ineffective_params — do NOT suggest changes in those directions, "
                    "(2) known_effective_params — prefer those directions, "
                    "(3) preferred_param_ranges — keep suggestions within those ranges, "
                    "(4) regularization_direction — follow its guidance (aggressive/cautious/neutral). "
                    if tuning_knowledge else ""
                )
                + "Output a JSON object with: diagnosis (short label), diagnosis_detail (explanation), "
                "action_items (recommended actions for THIS model), "
                "cross_references (other models/combos that would be affected — use combo_role "
                "and correlation_excerpt to identify them)."
            ),
        }, indent=2, ensure_ascii=False, default=str)

        return self._call_llm_json_object(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=cfg["model"],
            api_key=api_key,
            base_url=cfg["base_url"],
            temperature=cfg["temperature"],
            label=f"Per-Model({model_name})",
        )

    def generate_combo_critique(
        self,
        combo_name: str,
        combo_profile: dict,
        workspace_root: str,
    ) -> Optional[dict]:
        """
        Layered Per-Combo Critic — deep combo analysis.

        Args:
            combo_name: The combo being analyzed.
            combo_profile: Dict with keys: member_diagnoses, combo_history, loo_analysis,
                pairwise_correlation, oos_trend, market_context.
            workspace_root: Workspace path for skill loading.

        Returns a dict with keys: diagnosis, diagnosis_detail, member_assessments, action_items, cross_references.
        """
        cfg = self._resolve_llm_config(workspace_root)
        api_key = self._resolve_effective_api_key(workspace_root)
        if not api_key:
            return None

        system_prompt = self._load_skill(workspace_root, "combo_critic_system.md")
        if not system_prompt:
            print(f"   ⚠️  combo_critic_system.md not found — cannot run Per-Combo Critic.")
            return None

        active_scopes = self._load_active_scopes(workspace_root)

        user_prompt = json.dumps({
            "combo_name": combo_name,
            "active_scopes": active_scopes,
            "member_diagnoses": combo_profile.get("member_diagnoses", {}),
            "combo_history": combo_profile.get("combo_history", []),
            "loo_analysis": combo_profile.get("loo_analysis", {}),
            "pairwise_correlation": combo_profile.get("pairwise_correlation", {}),
            "oos_trend": combo_profile.get("oos_trend", {}),
            "market_context": combo_profile.get("market_context", {}),
            "instructions": (
                f"Analyze combo '{combo_name}' based on the profile above. "
                "Consider: member health (from Per-Model diagnoses), LOO deltas, "
                "pairwise correlations, OOS trend, and market context. "
                "Output a JSON object with: diagnosis (short label), diagnosis_detail (explanation), "
                "member_assessments (evaluate each member's role in THIS combo based on "
                "member_diagnoses and LOO deltas), "
                "action_items (combo-level: trigger_search, replace_member, adjust_weights)."
            ),
        }, indent=2, ensure_ascii=False, default=str)

        return self._call_llm_json_object(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=cfg["model"],
            api_key=api_key,
            base_url=cfg["base_url"],
            temperature=cfg["temperature"],
            label=f"Per-Combo({combo_name})",
        )

    def generate_execution_risk_critique(
        self,
        execution_profile: dict,
        workspace_root: str,
    ) -> Optional[dict]:
        """
        Dedicated Execution/Risk Critic — uses its own skill file
        (execution_risk_system.md) instead of reusing the model_critic prompt.

        Args:
            execution_profile: Dict with keys: execution_issues, trade_pattern_issues,
                market_context (from agent findings + triage).
            workspace_root: Workspace path for skill loading.

        Returns a dict with keys: diagnosis, diagnosis_detail, execution_issues,
        risk_issues, model_linkage, action_items.
        """
        cfg = self._resolve_llm_config(workspace_root)
        api_key = self._resolve_effective_api_key(workspace_root)
        if not api_key:
            return None

        system_prompt = self._load_skill(workspace_root, "execution_risk_system.md")
        if not system_prompt:
            print("   ⚠️  execution_risk_system.md not found — skipping Execution/Risk Critic.")
            return None

        active_scopes = self._load_active_scopes(workspace_root)

        # Build focused prompt from agent findings
        exec_issues = execution_profile.get("execution_issues", [])
        trade_issues = execution_profile.get("trade_pattern_issues", [])
        market_ctx = execution_profile.get("_execution_context", {})

        user_prompt = json.dumps({
            "active_scopes": active_scopes,
            "execution_issues": exec_issues,
            "trade_pattern_issues": trade_issues,
            "market_context": market_ctx,
            "instructions": (
                "Analyze execution quality and portfolio risk based on the agent findings above. "
                "Focus on OPERATIONAL issues (execution, trade patterns, risk exposure) — "
                "do NOT repeat model-level diagnoses. "
                "Output a JSON object with: diagnosis, diagnosis_detail, execution_issues, "
                "risk_issues, model_linkage, action_items."
            ),
        }, indent=2, ensure_ascii=False, default=str)

        return self._call_llm_json_object(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=cfg["model"],
            api_key=api_key,
            base_url=cfg["base_url"],
            temperature=cfg["temperature"],
            label="Execution/Risk",
        )

    def generate_synthesizer_output(
        self,
        model_diagnoses: Dict[str, dict],
        combo_diagnoses: Dict[str, dict],
        execution_risk_output: Optional[dict],
        rule_aggregator_result: dict,
        feedback_eval: Optional[dict],
        triage_result: dict,
        workspace_root: str,
    ) -> Optional[dict]:
        """
        Layered Synthesizer — final arbitration and ActionItem generation.

        Args:
            model_diagnoses: {model_name: diagnosis_dict} from all Per-Model calls.
            combo_diagnoses: {combo_name: diagnosis_dict} from all Per-Combo calls.
            execution_risk_output: Output from Execution/Risk LLM (or None).
            rule_aggregator_result: Pre-processed dedup + scope filter + conflict list.
            feedback_eval: Feedback Evaluator output with self_corrections rules.
            triage_result: Original Triage output for context.
            workspace_root: Workspace path for skill loading.

        Returns a dict with: global_diagnosis, conflict_resolutions, action_items,
        cross_validation_notes, scope_recommendations.
        """
        cfg = self._resolve_llm_config(workspace_root)
        api_key = self._resolve_effective_api_key(workspace_root)
        if not api_key:
            return None

        system_prompt = self._load_skill(workspace_root, "synthesizer_system.md")
        if not system_prompt:
            print("   ⚠️  synthesizer_system.md not found — cannot run Synthesizer.")
            return None

        active_scopes = self._load_active_scopes(workspace_root)

        # Load history for dedup
        recent_history = self._load_recent_action_history(workspace_root, limit=30)

        user_prompt = json.dumps({
            "active_scopes": active_scopes,
            "model_diagnoses": model_diagnoses,
            "combo_diagnoses": combo_diagnoses,
            "execution_risk_output": execution_risk_output,
            "rule_aggregator": rule_aggregator_result,
            "feedback_loop": feedback_eval,
            "triage_summary": {
                "systemic_observations": triage_result.get("systemic_observations", []),
                "healthy_models": triage_result.get("healthy_models", []),
            },
            "recent_action_history": [
                {
                    "date": e.get("_run_date", e.get("run_date", "")),
                    "target": e.get("target", ""),
                    "action_type": e.get("action_type", ""),
                    "params": e.get("params", {}),
                    "executed": bool(e.get("executed", False)),
                    "note": (
                        "SUGGESTED ONLY — never actually applied via adapter"
                        if not e.get("executed", False)
                        else "APPLIED — config was modified and model was retrained"
                    ),
                }
                for e in recent_history[:20]
            ],
            "instructions": (
                "You are the final arbiter. Your PRIMARY job is:\n"
                "1. Resolve conflicts between upstream outputs (Per-Model vs Per-Combo, etc.)\n"
                "2. Produce the final ranked ActionItem list (deduplicated, scope-filtered)\n"
                "3. Apply self_corrections from the feedback loop\n\n"
                "SECONDARY:\n"
                "- global_diagnosis: brief assessment (health_status, trend, systemic_risks)\n"
                "- cross_validation_notes: flag genuine inconsistencies between upstream outputs\n"
                "- scope_recommendations: suggest enabling scopes if blocked items exist\n\n"
                "Output a JSON object with: global_diagnosis, conflict_resolutions, "
                "action_items, cross_validation_notes, scope_recommendations."
            ),
        }, indent=2, ensure_ascii=False, default=str)

        return self._call_llm_json_object(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=cfg["model"],
            api_key=api_key,
            base_url=cfg["base_url"],
            temperature=cfg["temperature"],
            label="Synthesizer",
        )

    # ------------------------------------------------------------------
    # Experiment Analyzer — multi-round Playground decision-making
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_convergence(conv: Optional[dict]) -> str:
        """Pre-compute convergence quality so the LLM can't ignore it."""
        if not conv:
            return "NO TRAINING DATA — this is the first round, pick a param to try."
        best_ep = conv.get("best_epoch")
        actual_ep = conv.get("actual_epochs", 0)
        early = conv.get("early_stopped", False)
        if best_ep is not None and actual_ep > 10:
            if best_ep <= 1 and early:
                return (
                    f"⚠️ SEVERE OVERFITTING: best @ epoch {best_ep}/{actual_ep}. "
                    f"Model memorizes immediately. IC is NOT trustworthy. "
                    f"MUST try regularization (dropout, batch_size↑) or "
                    f"capacity reduction (hidden_size↓, num_layers↓). "
                    f"Do NOT give_up based on IC alone."
                )
            if best_ep <= 3:
                return (
                    f"MILD OVERFITTING: best @ epoch {best_ep}/{actual_ep}. "
                    f"Try dropout or reduce n_epochs."
                )
        return f"Healthy: best @ epoch {best_ep}, actual {actual_ep}. IC is trustworthy."

    def analyze_experiment_result(
        self,
        model_name: str,
        baseline_ic: float,
        playground_ic: float,
        changes_tried: List[dict],
        convergence: Optional[dict],
        current_params: dict,
        available_interventions: dict,
        hyperparam_bounds: dict,
        max_rounds_remaining: int,
        workspace_root: str,
        is_first_round: bool = False,
    ) -> Optional[dict]:
        """
        Analyze a single round of Playground training and decide next action.

        Returns a dict with decision/next_param/next_from/next_to or None.
        """
        ws_config = self._load_workspace_llm_config(workspace_root)
        llm_model = ws_config.get("critic_model") or self.model or "gpt-4"
        temperature = ws_config.get("temperature", 0.3)
        base_url = self.base_url or ws_config.get("base_url")
        api_key = self._resolve_effective_api_key(workspace_root)

        if not api_key:
            return None

        skill_path = os.path.join(
            workspace_root, "config", "skills", "experiment_analyzer.md",
        )
        system_prompt = None
        if os.path.exists(skill_path):
            try:
                with open(skill_path, "r", encoding="utf-8") as f:
                    system_prompt = f.read().strip()
            except Exception:
                pass

        if not system_prompt:
            print("   ⚠️  experiment_analyzer.md not found — skipping experiment loop")
            return None

        model_interventions = available_interventions.get(model_name, {})
        bounds_summary = {
            k: {"min": v.get("min"), "max": v.get("max"),
                "max_change_pct": v.get("max_change_pct")}
            for k, v in hyperparam_bounds.items()
        }

        user_prompt = "\n".join([
            f"## Model\n{model_name}",
        ] + ([
            f"\n## ⚠️ FIRST ROUND — NO TRAINING HAS BEEN DONE YET",
            f"The IC values below are placeholders (baseline only). "
            f"Your job is to suggest the FIRST parameter to try. "
            f"Pick the most promising untouched parameter and suggest a "
            f"moderate first-step change. Do NOT give_up — there is no "
            f"result to evaluate yet.",
        ] if is_first_round else []) + [
            f"\n## Baseline IC (production)\n{baseline_ic:.6f}",
            f"\n## Current Playground IC\n{playground_ic:.6f}",
            f"\n## IC Change\n{(playground_ic - baseline_ic):+.6f} "
            f"({'IMPROVED' if playground_ic > baseline_ic else 'DEGRADED'})" + (
                " (PLACEHOLDER — no training yet)" if is_first_round else ""
            ),
            f"\n## Changes Tried So Far\n```json\n{json.dumps(changes_tried, indent=2, ensure_ascii=False)}\n```",
            f"\n## Convergence Pattern\n```json\n{json.dumps(convergence or {}, indent=2, ensure_ascii=False)}\n```"
            f"\n## Convergence Assessment (pre-computed — TRUST THIS)"
            f"\n{self._assess_convergence(convergence)}",
            f"\n## Current Hyperparameter Values\n```json\n{json.dumps(current_params.get(model_name, {}), indent=2, ensure_ascii=False)}\n```",
            f"\n## Available Interventions\n"
            f"recently_adjusted (DO NOT repeat): {model_interventions.get('recently_adjusted', [])}\n"
            f"untouched (SAFE to try): {model_interventions.get('untouched', [])}\n"
            f"exhausted: {model_interventions.get('exhausted', False)}",
            f"\n## Hyperparameter Bounds\n```json\n{json.dumps(bounds_summary, indent=2, ensure_ascii=False)}\n```",
            f"\n## Constraints\n- Max rounds remaining: {max_rounds_remaining}\n"
            f"- Check convergence FIRST (see assessment above). "
            f"Overfitting overrides IC improvement.\n"
            f"- If convergence is healthy AND IC improved meaningfully → give_up\n"
            f"- If convergence broken → retry with regularization regardless of IC\n"
            f"- Do NOT repeat a param already in changes_tried\n"
            f"- Output ONLY a JSON object: {{\"decision\": \"retry|give_up\", \"reason\": \"...\", "
            f"\"next_param\": \"...\", \"next_from\": value, \"next_to\": value, \"rationale\": \"...\"}}",
        ])

        try:
            client = self._get_or_create_client(api_key, base_url)
        except RuntimeError:
            return None

        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=32768,
            )

            choice = response.choices[0]
            finish = getattr(choice, "finish_reason", "?")
            content = choice.message.content
            reasoning = getattr(choice.message, "reasoning_content", None)
            if not content:
                print(f"   ⚠️  ExperimentAnalyzer empty: finish={finish} "
                      f"reasoning_len={len(reasoning) if reasoning else 0}")
                if reasoning and finish == "length":
                    json_str = self._extract_json_object(reasoning)
                    if json_str:
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass
                return None

            json_str = self._extract_json_object(content)
            if json_str is None:
                print(f"   ⚠️  ExperimentAnalyzer: no JSON found. Raw (200 chars): {content[:200]}")
                return None

            result = json.loads(json_str)
            decision = result.get("decision", "give_up")
            if decision == "retry":
                print(
                    f"   🔄 Experiment: retry → {result.get('next_param', '?')} "
                    f"({result.get('next_from', '?')} → {result.get('next_to', '?')})"
                )
            else:
                print(f"   🛑 Experiment: give_up — {result.get('reason', result.get('rationale', '?'))}")
            return result

        except Exception as e:
            logger.warning("ExperimentAnalyzer call failed: %s", e)
            print(f"   ⚠️  ExperimentAnalyzer failed: {e}")
            return None

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
        client = self._get_or_create_client(api_key, base_url)

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
