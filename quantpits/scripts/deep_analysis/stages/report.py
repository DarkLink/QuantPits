"""
Stage 7: report — render the final Markdown analysis report.
"""

import os

from ..stage_runner import register_stage


@register_stage(
    name='report',
    depends_on=['agents', 'synthesis', 'critic'],
    provides=['report_md', 'output_path'],
    description='Render the final Markdown deep analysis report',
)
def run_report(state, **kwargs):
    """Generate and write the final Markdown report.

    Populates ``state.report_md`` and ``state.output_path``.
    """
    from quantpits.scripts.deep_analysis.report_generator import ReportGenerator

    workspace_root = kwargs.get('workspace_root', state.workspace_root)
    run_date = kwargs.get('run_date', state.run_date)
    run_label = kwargs.get('run_label', state.run_label)
    output_arg = kwargs.get('output', 'output/deep_analysis_report.md')
    shareable = kwargs.get('shareable', False)

    print("\n📝 Generating report...")
    report_gen = ReportGenerator(
        all_findings=state.all_findings,
        synthesis_result=state.synthesis_result,
        executive_summary=state.executive_summary,
    )
    report_md = report_gen.generate()

    # Determine output path
    label_suffix = f"_{run_label}" if run_label else ""
    base = output_arg.replace('.md', '')
    output_path = f"{base}_{run_date}{label_suffix}.md"
    output_path = os.path.join(workspace_root, output_path) if not os.path.isabs(output_path) else output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_md)

    print(f"   → Report saved to: {output_path}")

    state.report_md = report_md
    state.output_path = output_path
    return state
