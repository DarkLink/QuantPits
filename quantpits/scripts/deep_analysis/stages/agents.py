"""
Stage 2: agents — run all registered specialist agents across every time window.
"""

from ..stage_runner import register_stage


@register_stage(
    name='agents',
    depends_on=['discover'],
    provides=['all_findings'],
    description='Execute specialist agents across all time windows',
)
def run_agents(state, **kwargs):
    """Instantiate and run specialist agents on each time window.

    Populates ``state.all_findings`` with a list of ``AgentFindings``.
    """
    from quantpits.scripts.deep_analysis.agents import ALL_AGENTS, load_manifest_agents
    from quantpits.scripts.deep_analysis.coordinator import Coordinator
    from quantpits.scripts.deep_analysis.base_agent import AgentFindings

    workspace_root = kwargs.get('workspace_root', state.workspace_root)
    manifest_path = kwargs.get('manifest_path', None)
    agent_filter = kwargs.get('agent_filter', 'all')

    # Rebuild coordinator from discover outputs
    coordinator = Coordinator(
        workspace_root=workspace_root,
        freq_change_date=state.freq_change_date,
        external_notes=state.external_notes,
        windows=state.windows,
    )
    coordinator._data_start_date = state.data_start_date
    coordinator._data_end_date = state.data_end_date
    coordinator._discovered_files = state.discovered_files
    coordinator._load_shared_dataframes()

    # Load dynamic agents
    manifest_agents = load_manifest_agents(workspace_root, manifest_path)
    ALL_AGENTS.update(manifest_agents)

    # Determine which agents to run
    specific_agents = kwargs.get('specific_agents', None)
    if specific_agents:
        agent_names = specific_agents
    elif agent_filter == 'all':
        agent_names = list(ALL_AGENTS.keys())
    else:
        agent_names = [a.strip() for a in agent_filter.split(',')]

    agents = []
    for name in agent_names:
        if name in ALL_AGENTS:
            agents.append(ALL_AGENTS[name]())
        else:
            print(f"  ⚠️  Unknown agent: {name}. Available: {list(ALL_AGENTS.keys())}")

    if not agents:
        print("❌ No agents to run.")
        return state

    print(f"\n🔧 Running {len(agents)} agents: {[a.name for a in agents]}")

    findings_map = {}
    for window in state.windows:
        ctx = coordinator.build_context(window)
        print(f"\n{'='*60}")
        print(f"  Window: [{window['label']}] {window['start_date']} → {window['end_date']}")
        print(f"{'='*60}")

        for agent in agents:
            try:
                print(f"  🔍 Running {agent.name}...")
                findings = agent.analyze(ctx)
                findings_map[(agent.name, window['label'])] = findings
                n_findings = len(findings.findings)
                n_critical = sum(1 for f in findings.findings if f.severity == 'critical')
                n_warning = sum(1 for f in findings.findings if f.severity == 'warning')
                print(f"     → {n_findings} findings ({n_critical} critical, {n_warning} warning)")
            except Exception as e:
                print(f"  ❌ {agent.name} failed: {e}")
                import traceback
                traceback.print_exc()
                findings_map[(agent.name, window['label'])] = AgentFindings(
                    agent_name=agent.name,
                    window_label=window['label'],
                    findings=[],
                    recommendations=[],
                    raw_metrics={'error': str(e)},
                )

    state.all_findings = list(findings_map.values())
    return state
