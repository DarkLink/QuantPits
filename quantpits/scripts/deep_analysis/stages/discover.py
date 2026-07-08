"""
Stage 1: discover — scan workspace for data files and generate time windows.
"""

from ..stage_runner import register_stage


@register_stage(
    name='discover',
    depends_on=[],
    provides=['data_start_date', 'data_end_date', 'discovered_files', 'windows'],
    description='Scan workspace for data files and generate analysis time windows',
)
def run_discover(state, **kwargs):
    """Discover data files, load shared DataFrames, generate time windows.

    Populates ``state.data_start_date``, ``state.data_end_date``,
    ``state.discovered_files``, and ``state.windows``.
    """
    from quantpits.scripts.deep_analysis.coordinator import Coordinator

    workspace_root = kwargs.get('workspace_root', state.workspace_root)
    freq_change_date = kwargs.get('freq_change_date', state.freq_change_date)
    external_notes = kwargs.get('external_notes', state.external_notes)
    windows = kwargs.get('windows', [])

    # Snapshot config if not disabled
    snapshot = kwargs.get('snapshot_config', True)
    no_snapshot = kwargs.get('no_snapshot', False)
    if snapshot and not no_snapshot:
        print("\n📸 Saving config snapshot...")
        try:
            from quantpits.scripts.deep_analysis.config_ledger import (
                snapshot_configs, save_snapshot,
            )
            snap = snapshot_configs(workspace_root)
            path = save_snapshot(workspace_root, snap)
            print(f"   → Saved to {path}")
        except Exception as e:
            print(f"   ⚠️  Config snapshot failed: {e}")

    coordinator = Coordinator(
        workspace_root=workspace_root,
        freq_change_date=freq_change_date,
        external_notes=external_notes,
        windows=windows,
    )
    coordinator.discover()
    generated_wins = coordinator.generate_windows()

    print(f"  Data range: {coordinator._data_start_date} → {coordinator._data_end_date}")
    print(f"  Windows: {[w['label'] for w in generated_wins]}")

    state.data_start_date = coordinator._data_start_date
    state.data_end_date = coordinator._data_end_date
    state.discovered_files = coordinator._discovered_files
    state.windows = generated_wins

    return state
