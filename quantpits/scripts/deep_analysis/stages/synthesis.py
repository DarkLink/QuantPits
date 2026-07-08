"""
Stage 3: synthesis — cross-reference agent findings into a structured summary.
"""

from ..stage_runner import register_stage


@register_stage(
    name='synthesis',
    depends_on=['agents'],
    provides=['synthesis_result'],
    description='Cross-reference agent findings and produce structured synthesis',
)
def run_synthesis(state, **kwargs):
    """Run cross-agent synthesis on all agent findings.

    Populates ``state.synthesis_result`` with cross-findings,
    recommendations, health status, and executive-summary data.
    """
    from quantpits.scripts.deep_analysis.synthesizer import Synthesizer

    external_notes = kwargs.get('external_notes', state.external_notes)

    print("\n🧠 Running cross-agent synthesis...")
    synthesizer = Synthesizer(state.all_findings, external_notes=external_notes)
    result = synthesizer.synthesize()

    n_cross = len(result.get('cross_findings', []))
    n_recs = len(result.get('recommendations', []))
    print(f"   → {n_cross} cross-agent findings, {n_recs} recommendations")

    state.synthesis_result = result
    return state
