"""
LLM Interface for the MAS Deep Analysis System.

Provides OpenAI API integration for natural language synthesis of analysis findings.
Falls back to template-based generation if no API key or API failure.
"""

import os
import json
from typing import List, Dict, Optional


class LLMInterface:
    """
    LLM integration layer for synthesizing analysis results into natural language.
    
    Supports OpenAI API with template-based fallback.
    """

    SYSTEM_PROMPT = """You are an expert quantitative finance analyst reviewing a systematic 
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

    def generate_executive_summary(self, synthesis_result: dict) -> str:
        """
        Generate an executive summary from synthesis results.
        
        Args:
            synthesis_result: Output from Synthesizer.synthesize()
            
        Returns:
            Formatted executive summary string
        """
        if not self.is_available():
            return self._template_summary(synthesis_result)

        try:
            return self._llm_summary(synthesis_result)
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}. Falling back to template.")
            return self._template_summary(synthesis_result)

    def _llm_summary(self, synthesis_result: dict) -> str:
        """Generate summary using OpenAI API."""
        client = self._get_client()

        # Build user prompt from structured data
        user_prompt = self._build_prompt(synthesis_result)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        return response.choices[0].message.content.strip()

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
