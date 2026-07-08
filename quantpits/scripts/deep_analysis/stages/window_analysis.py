"""
Stage 4: window_analysis — rule-based training window adequacy analysis (optional).
"""

from ..stage_runner import register_stage


@register_stage(
    name='window_analysis',
    depends_on=['agents'],
    provides=['window_findings', 'window_analysis_context'],
    description='Run rule-based training window adequacy checks (static + data-driven + CPCV/rolling)',
)
def run_window_analysis(state, **kwargs):
    """Run TrainingWindowAnalyzer if ``--window-analysis`` was requested.

    Populates ``state.window_findings`` and ``state.window_analysis_context``.
    Skipped (empty lists) when the flag is off.
    """
    workspace_root = kwargs.get('workspace_root', state.workspace_root)
    run_date = kwargs.get('run_date', state.run_date)

    window_findings = []
    window_analysis_context = {}

    print("\n🔍 Running Training Window Analyzer...")
    from quantpits.scripts.deep_analysis.training_window_analyzer import (
        TrainingWindowAnalyzer,
    )

    window_analyzer = TrainingWindowAnalyzer(
        workspace_root=workspace_root,
        reference_date=run_date,
    )

    # Extract Market Regime metrics from agent findings
    market_regime_metrics = {}
    for af in state.all_findings:
        if getattr(af, 'agent_name', '') == "Market Regime":
            market_regime_metrics = getattr(af, 'raw_metrics', {})
            break

    # Attempt to load benchmark data for data-driven rules (R7-R13)
    benchmark_data = {}
    try:
        from quantpits.scripts.deep_analysis.benchmark_data_loader import (
            load_benchmark_data,
        )
        benchmark_data = load_benchmark_data(workspace_root)
        if benchmark_data.get("error"):
            print(f"   ⚠️  Benchmark data unavailable: {benchmark_data['error']}")
        else:
            print(f"   ✅ {benchmark_data.get('benchmark', '')} data loaded")
    except Exception as e:
        print(f"   ⚠️  Failed to load benchmark data: {e}")

    window_findings = window_analyzer.analyze(
        market_regime_metrics=market_regime_metrics,
        benchmark_data=benchmark_data if not benchmark_data.get("error") else None,
    )
    print(f"   → {len(window_findings)} window analysis findings")
    for f in window_findings:
        print(f"      [{getattr(f, 'severity', '?')}] "
              f"{getattr(f, 'finding_type', '?')}: "
              f"{str(getattr(f, 'context', ''))[:120]}")

    window_analysis_context = {
        "findings": [f.to_dict() if hasattr(f, 'to_dict') else f
                     for f in window_findings],
    }

    state.window_findings = window_findings
    state.window_analysis_context = window_analysis_context
    return state
