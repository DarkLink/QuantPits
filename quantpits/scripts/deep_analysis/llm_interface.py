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
        Critic mode: generate ActionItems from structured Signals.

        Flow:
        1. Load llm_config.json for model/API settings
        2. Load feedback_scope.json for active_scopes
        3. Load matching skill files from config/skills/
        4. Build system prompt (critic_system.md + scope skills)
        5. Build user prompt (signals JSON + active_scopes)
        6. Call LLM
        7. Parse JSON output into ActionItem list

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

        # Build system prompt from skills
        system_prompt = self._load_skills(workspace_root, active_scopes)

        # Build user prompt
        user_prompt = self._build_critic_prompt(signals, active_scopes, hyperparam_bounds)

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

    def _build_critic_prompt(
        self, signals: List[Signal], active_scopes: List[str],
        hyperparam_bounds: Optional[dict] = None,
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

        parts.append(
            f"## Signals\n```json\n{signals_json}\n```\n\n"
            f"Based on the above signals, generate a JSON array of ActionItems. "
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

                content = response.choices[0].message.content.strip()
                return self._parse_action_items(content)

            except json.JSONDecodeError as e:
                last_error = e
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
