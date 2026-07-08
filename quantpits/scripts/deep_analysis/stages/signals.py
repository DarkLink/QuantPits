"""
Stage 5: signals — extract structured Signals from agent findings and synthesis.
"""

from ..stage_runner import register_stage


@register_stage(
    name='signals',
    depends_on=['agents', 'synthesis', 'window_analysis'],
    provides=['signals'],
    description='Extract structured Signals from agent findings, synthesis, and window analysis',
)
def run_signals(state, **kwargs):
    """Run SignalExtractor on all available upstream outputs.

    Populates ``state.signals`` with a list of ``Signal`` objects.
    """
    from quantpits.scripts.deep_analysis.signal_extractor import SignalExtractor

    workspace_root = kwargs.get('workspace_root', state.workspace_root)
    run_date = kwargs.get('run_date', state.run_date)

    print("\n📡 Extracting structured signals...")
    extractor = SignalExtractor(
        reference_date=run_date,
        workspace_root=workspace_root,
        window_analysis_findings=state.window_findings,
        window_analysis_context=state.window_analysis_context
        if state.window_analysis_context else None,
    )
    signals = extractor.extract(state.all_findings, state.synthesis_result)
    print(f"   → {len(signals)} signals extracted")

    state.signals = signals
    return state
