#!/usr/bin/env python3
"""
MAS Deep Analysis System — CLI Entry Point

Automated multi-agent post-trade analysis with optional LLM-powered synthesis.

Usage:
    # Rule-based full analysis
    python -m quantpits.scripts.run_deep_analysis

    # With LLM synthesis
    python -m quantpits.scripts.run_deep_analysis --llm openai

    # With external notes
    python -m quantpits.scripts.run_deep_analysis --llm openai \\
        --notes "Retrained catboost last week."

    # Specific agents only
    python -m quantpits.scripts.run_deep_analysis --agents model_health,prediction_audit

    # Custom windows
    python -m quantpits.scripts.run_deep_analysis --windows 1y,3m,1m
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Ensure QuantPits modules are importable
try:
    from quantpits.utils import env
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from quantpits.utils import env

os.chdir(env.ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MAS Deep Analysis System — Automated multi-agent post-trade analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--windows', type=str, default='full,weekly_era,1y,6m,3m,1m',
        help='Comma-separated time windows (default: full,weekly_era,1y,6m,3m,1m)')
    parser.add_argument(
        '--freq-change-date', type=str, default=None,
        help='Date when trading frequency changed (e.g., 2024-10-21). '
             'Auto-read from config/deep_analysis_config.json if not specified.')
    parser.add_argument(
        '--output', type=str, default='output/deep_analysis_report.md',
        help='Output report path (default: output/deep_analysis_report.md)')
    parser.add_argument(
        '--llm', type=str, default='none', choices=['none', 'openai'],
        help='LLM backend for synthesis (default: none)')
    parser.add_argument(
        '--llm-model', type=str, default='gpt-4',
        help='LLM model name (default: gpt-4)')
    parser.add_argument(
        '--api-key', type=str, default=None,
        help='OpenAI API key (default: $OPENAI_API_KEY)')
    parser.add_argument(
        '--base-url', type=str, default=None,
        help='OpenAI API base URL (for compatible endpoints)')
    parser.add_argument(
        '--agents', type=str, default='all',
        help='Comma-separated agent names to run (default: all)')
    parser.add_argument(
        '--notes', type=str, default='',
        help='Free-text external context notes')
    parser.add_argument(
        '--notes-file', type=str, default=None,
        help='Path to file containing external notes')
    parser.add_argument(
        '--shareable', action='store_true',
        help='Redact sensitive data in report')
    parser.add_argument(
        '--snapshot-config', action='store_true', default=True,
        help='Save config snapshot to history ledger (default: True)')
    parser.add_argument(
        '--no-snapshot', action='store_true',
        help='Skip config snapshot')
    return parser.parse_args()


def load_deep_analysis_config(workspace_root: str) -> dict:
    """Load deep analysis configuration if it exists."""
    config_path = os.path.join(workspace_root, 'config', 'deep_analysis_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main():
    args = parse_args()
    workspace_root = env.ROOT_DIR

    print("=" * 60)
    print("  🤖 MAS Deep Analysis System")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Workspace: {workspace_root}")
    print("=" * 60)

    # Load config
    da_config = load_deep_analysis_config(workspace_root)

    # Resolve freq-change-date
    freq_change_date = args.freq_change_date or da_config.get('freq_change_date')

    # Resolve external notes
    external_notes = args.notes
    if args.notes_file and os.path.exists(args.notes_file):
        with open(args.notes_file, 'r') as f:
            external_notes = f.read().strip()

    # Parse windows
    windows = [w.strip() for w in args.windows.split(',')]

    # --- 1. Config Snapshot ---
    if args.snapshot_config and not args.no_snapshot:
        print("\n📸 Saving config snapshot...")
        try:
            from quantpits.scripts.deep_analysis.config_ledger import (
                snapshot_configs, save_snapshot
            )
            snapshot = snapshot_configs(workspace_root)
            path = save_snapshot(workspace_root, snapshot)
            print(f"   → Saved to {path}")
        except Exception as e:
            print(f"   ⚠️  Config snapshot failed: {e}")

    # --- 2. Initialize Coordinator ---
    from quantpits.scripts.deep_analysis.coordinator import Coordinator

    coordinator = Coordinator(
        workspace_root=workspace_root,
        freq_change_date=freq_change_date,
        external_notes=external_notes,
        windows=windows,
    )

    # --- 3. Initialize Agents ---
    from quantpits.scripts.deep_analysis.agents import ALL_AGENTS

    if args.agents == 'all':
        agent_names = list(ALL_AGENTS.keys())
    else:
        agent_names = [a.strip() for a in args.agents.split(',')]

    agents = []
    for name in agent_names:
        if name in ALL_AGENTS:
            agents.append(ALL_AGENTS[name]())
        else:
            print(f"  ⚠️  Unknown agent: {name}. Available: {list(ALL_AGENTS.keys())}")

    if not agents:
        print("❌ No agents to run. Exiting.")
        return 1

    print(f"\n🔧 Running {len(agents)} agents: {[a.name for a in agents]}")

    # --- 4. Run Analysis ---
    all_findings = coordinator.run(agents)

    print(f"\n✅ Analysis complete. {len(all_findings)} agent-window results collected.")

    # --- 5. Synthesize ---
    print("\n🧠 Running cross-agent synthesis...")
    from quantpits.scripts.deep_analysis.synthesizer import Synthesizer

    synthesizer = Synthesizer(all_findings, external_notes=external_notes)
    synthesis_result = synthesizer.synthesize()

    n_cross = len(synthesis_result.get('cross_findings', []))
    n_recs = len(synthesis_result.get('recommendations', []))
    print(f"   → {n_cross} cross-agent findings, {n_recs} recommendations")

    # --- 6. LLM Executive Summary ---
    print("\n📝 Generating executive summary...")
    from quantpits.scripts.deep_analysis.llm_interface import LLMInterface

    if args.llm == 'openai':
        llm = LLMInterface(
            api_key=args.api_key,
            model=args.llm_model,
            base_url=args.base_url,
        )
        if llm.is_available():
            print(f"   Using OpenAI API (model: {args.llm_model})")
        else:
            print("   ⚠️  No API key found. Using template-based summary.")
    else:
        llm = LLMInterface()  # No API key → template mode

    executive_summary = llm.generate_executive_summary(synthesis_result)

    # --- 7. Generate Report ---
    print("\n📊 Generating report...")
    from quantpits.scripts.deep_analysis.report_generator import ReportGenerator

    report_gen = ReportGenerator(all_findings, synthesis_result, executive_summary)
    report_md = report_gen.generate()

    # Write output
    output_path = os.path.join(workspace_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_md)

    print(f"\n✅ Report saved to: {output_path}")
    print(f"   Health: {synthesis_result.get('health_status', 'Unknown')}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
