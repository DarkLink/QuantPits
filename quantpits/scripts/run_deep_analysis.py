#!/usr/bin/env python3
"""
MAS Deep Analysis System — CLI Entry Point

Automated multi-agent post-trade analysis with optional LLM-powered synthesis.

Usage:
    # Rule-based full analysis
    python -m quantpits.scripts.run_deep_analysis

    # With LLM synthesis (reads model/endpoint from config/llm_config.json)
    python -m quantpits.scripts.run_deep_analysis --llm

    # With external notes
    python -m quantpits.scripts.run_deep_analysis --llm \\
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
import re
from datetime import datetime
from typing import Optional

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
        '--llm', action='store_true',
        help='Enable LLM-powered executive summary (reads model/endpoint from config/llm_config.json)')
    parser.add_argument(
        '--llm-model', type=str, default=None,
        help='Override LLM model for summary (default: use llm_config.json summary_model)')
    parser.add_argument(
        '--api-key', type=str, default=None,
        help='API key override (default: reads env var from llm_config.json api_key_env)')
    parser.add_argument(
        '--base-url', type=str, default=None,
        help='API base URL override (default: use llm_config.json base_url)')
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
    parser.add_argument(
        '--critic', action='store_true',
        help='Enable Critic mode: generate ActionItems from signals')
    parser.add_argument(
        '--critic-dry-run', action='store_true',
        help='Generate ActionItems but do not persist to files (preview mode)')
    parser.add_argument(
        '--window-analysis', action='store_true',
        help='Enable rule-based training window analysis (independent of Critic)')
    parser.add_argument(
        '--run-label', type=str, default='',
        help='Label for this run (e.g., "after-retrain"). '
             'When set, output filenames include the label to prevent '
             'collisions when running multiple analyses on the same date.')
    parser.add_argument(
        '--stage', type=str, default='all',
        help='Execution stage: discover | agents | agents:NAME | synthesis | window_analysis | signals | critic | report | all (default: all)')
    parser.add_argument(
        '--resume-from', type=str, default=None,
        help='Path to a stage checkpoint file to resume execution from')
    parser.add_argument(
        '--resume-latest', action='store_true',
        help='Automatically find the latest checkpoint to resume from')
    parser.add_argument(
        '--manifest', type=str, default=None,
        help='Path to agent_manifest.yaml configuration file')
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


# ------------------------------------------------------------------
# Single-Stage Critic (backward compatible)
# ------------------------------------------------------------------

def _run_critic_single_stage(
    critic_llm,
    signals: list,
    workspace_root: str,
    data_date: str = "",
    dry_run: bool = False,
    run_label: str = "",
) -> list:
    """Run the original single-stage Critic flow."""
    from quantpits.scripts.deep_analysis.action_items import (
        ActionItemValidator, persist_action_items,
    )

    print("\n🧪 Running LLM Critic (single-stage)...")
    action_items = critic_llm.generate_action_items(signals, workspace_root)
    print(f"   → {len(action_items)} action items generated")

    print("\n🔍 Validating action items...")
    validator = ActionItemValidator(
        feedback_scope_path=os.path.join(workspace_root, 'config', 'feedback_scope.json'),
        hyperparam_bounds_path=os.path.join(workspace_root, 'config', 'hyperparam_bounds.json'),
        workspace_root=workspace_root,
    )
    action_items = validator.validate(action_items)
    n_in = sum(1 for a in action_items if a.scope_status == 'in_scope')
    n_out = sum(1 for a in action_items if a.scope_status == 'out_of_scope')
    n_rej = sum(1 for a in action_items if a.scope_status == 'rejected')
    print(f"   → {n_in} in-scope, {n_out} out-of-scope, {n_rej} rejected")

    if not dry_run:
        print("\n💾 Persisting action items...")
        snap_path = persist_action_items(action_items, workspace_root, run_date=data_date, run_label=run_label)
        print(f"   → Saved to {snap_path}")
    else:
        print("\n🏜️  Dry-run mode — action items not persisted.")
        for ai in action_items:
            status_icon = {'in_scope': '✅', 'out_of_scope': '⚠️', 'rejected': '❌'}.get(ai.scope_status, '❓')
            print(f"   {status_icon} [{ai.scope_status}] {ai.action_type}: {ai.target} — {ai.reason[:80]}")

    return action_items


# ------------------------------------------------------------------
# Layered Critic (triage → per-model → per-combo → synthesizer)
# ------------------------------------------------------------------

def _run_critic_layered(
    critic_llm,
    signals: list,
    all_findings: list,
    synthesis_result: dict,
    signal_extractor,
    workspace_root: str,
    data_date: str = "",
    dry_run: bool = False,
    run_label: str = "",
) -> list:
    """
    Run the layered pipeline:
    Triage → Per-Model (parallel) → Per-Combo → Exec/Risk → Feedback → Synthesizer
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from quantpits.scripts.deep_analysis.action_items import (
        ActionItemValidator, persist_action_items, ActionItem,
    )
    from quantpits.scripts.deep_analysis.action_aggregator import ActionAggregator
    from quantpits.scripts.deep_analysis.feedback_evaluator import run_feedback_loop

    print("\n🧪 Running LLM Critic (layered pipeline)...")

    # --- Step 1: Triage ---
    print("\n📋 Step 1/6: Triage...")
    triage_input = signal_extractor.extract_triage_input(
        all_findings, synthesis_result, signals,
    )
    triage_result = critic_llm.generate_triage(
        triage_input=triage_input,
        signals=signals,
        workspace_root=workspace_root,
    )
    if triage_result is None:
        print("   ❌ Triage failed. Falling back to single-stage.")
        return _run_critic_single_stage(critic_llm, signals, workspace_root, dry_run)

    # --- Step 2: Per-Model LLM (parallel) ---
    prioritized_models = triage_result.get("prioritized_models", [])
    if isinstance(prioritized_models, list) and prioritized_models and isinstance(prioritized_models[0], dict):
        model_names = [m.get("target", m.get("model", "")) for m in prioritized_models]
    else:
        model_names = prioritized_models if isinstance(prioritized_models, list) else []

    print(f"\n📋 Step 2/6: Per-Model Critic ({len(model_names)} models)...")

    model_diagnoses = {}
    if model_names:
        model_profiles = _build_model_profiles(
            model_names, triage_input, signals, all_findings, workspace_root,
        )

        with ThreadPoolExecutor(max_workers=min(8, len(model_names))) as executor:
            futures = {
                executor.submit(
                    critic_llm.generate_model_critique,
                    model_name=name,
                    model_profile=profile,
                    workspace_root=workspace_root,
                ): name
                for name, profile in model_profiles.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if result:
                        model_diagnoses[name] = result
                        diag = result.get("diagnosis", "?")
                        print(f"   ✅ {name}: {diag}")
                    else:
                        print(f"   ⚠️  {name}: LLM call returned None")
                except Exception as e:
                    print(f"   ❌ {name}: {e}")

    # --- Step 3: Per-Combo LLM + Execution/Risk LLM (parallel) ---
    prioritized_combos = triage_result.get("prioritized_combos", [])
    if isinstance(prioritized_combos, list) and prioritized_combos and isinstance(prioritized_combos[0], dict):
        combo_names = [c.get("combo", c.get("target", "")) for c in prioritized_combos]
    else:
        combo_names = prioritized_combos if isinstance(prioritized_combos, list) else []

    print(f"\n📋 Step 3/6: Per-Combo Critic ({len(combo_names)} combos + Exec/Risk)...")

    combo_diagnoses = {}
    execution_risk_output = None

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit combo tasks
        combo_futures = {}
        for combo_name in combo_names:
            combo_profile = _build_combo_profile(
                combo_name, model_diagnoses, triage_input, all_findings,
                workspace_root=workspace_root,
            )
            combo_futures[
                executor.submit(
                    critic_llm.generate_combo_critique,
                    combo_name=combo_name,
                    combo_profile=combo_profile,
                    workspace_root=workspace_root,
                )
            ] = combo_name

        # Submit execution/risk task (dedicated skill, parallel with combos)
        exec_future = None
        if triage_result.get("needs_execution_risk", False):
            exec_profile = _build_execution_profile(all_findings, triage_input)
            exec_future = executor.submit(
                critic_llm.generate_execution_risk_critique,
                execution_profile=exec_profile,
                workspace_root=workspace_root,
            )

        # Collect combo results
        for future in as_completed(combo_futures):
            name = combo_futures[future]
            try:
                result = future.result()
                if result:
                    combo_diagnoses[name] = result
                    print(f"   ✅ Combo {name}: {result.get('diagnosis', '?')}")
                else:
                    print(f"   ⚠️  Combo {name}: LLM call returned None")
            except Exception as e:
                print(f"   ❌ Combo {name}: {e}")

        # Collect exec/risk result
        if exec_future:
            try:
                execution_risk_output = exec_future.result()
                if execution_risk_output:
                    print(f"   ✅ Execution/Risk: analyzed")
            except Exception as e:
                print(f"   ❌ Execution/Risk: {e}")

    # --- Step 4: Feedback Loop ---
    print("\n📋 Step 4/6: Feedback Evaluator...")
    feedback_eval = run_feedback_loop(
        workspace_root, current_date=data_date, latest_data_date=data_date,
    )
    if feedback_eval.get("evaluated"):
        qs = feedback_eval.get("quality_summary", {})
        print(f"   → {qs.get('total', 0)} previous items evaluated, "
              f"{qs.get('incorrect', 0)} incorrect, "
              f"{len(feedback_eval.get('self_corrections', []))} self-corrections")
    else:
        print(f"   → {feedback_eval.get('reason', 'No previous analysis')}")

    # --- Step 5: Rule Aggregator ---
    print("\n📋 Step 5/6: Action Aggregator...")
    aggregator = ActionAggregator(workspace_root)
    agg_result = aggregator.aggregate(model_diagnoses, combo_diagnoses, execution_risk_output)
    print(f"   → {agg_result['raw_count']} raw → {agg_result['deduped_count']} deduped "
          f"({agg_result['in_scope_count']} in-scope, {agg_result['out_of_scope_count']} out-of-scope)")
    if agg_result["conflicts"]:
        print(f"   → {len(agg_result['conflicts'])} conflicts detected for Synthesizer")

    # --- Step 6: Synthesizer ---
    print("\n📋 Step 6/6: Synthesizer...")
    synthesizer_output = critic_llm.generate_synthesizer_output(
        model_diagnoses=model_diagnoses,
        combo_diagnoses=combo_diagnoses,
        execution_risk_output=execution_risk_output,
        rule_aggregator_result=agg_result,
        feedback_eval=feedback_eval,
        triage_result=triage_result,
        workspace_root=workspace_root,
    )

    if synthesizer_output is None:
        print("   ❌ Synthesizer failed. Falling back to aggregated items.")
        action_items = [
            ActionItem.from_dict(item) for item in agg_result.get("deduped_items", [])
        ]
    else:
        gd = synthesizer_output.get("global_diagnosis", {})
        llm_health = gd.get("health_status", "unknown")
        print(f"   → Health: {llm_health}, "
              f"Trend: {gd.get('trend', '?')}")
        print(f"   → {len(synthesizer_output.get('action_items', []))} final ActionItems, "
              f"{len(synthesizer_output.get('conflict_resolutions', []))} conflicts resolved")

        # Override the rule-based health status with the LLM Synthesizer's
        # more nuanced diagnosis
        synthesis_result["health_status"] = llm_health
        synthesis_result["_llm_diagnosis"] = gd

        # Inject layered Critic Pipeline output so the Executive Summary
        # LLM can see and respect the Synthesizer's conclusions.
        synthesis_result["_critic_pipeline_output"] = {
            "action_items": [
                {"target": ai.get("target", ""),
                 "action_type": ai.get("action_type", ""),
                 "reason": (ai.get("reason", "") or "")[:200]}
                for ai in synthesizer_output.get("action_items", [])
            ],
            "conflict_resolutions": synthesizer_output.get("conflict_resolutions", []),
            "scope_recommendations": synthesizer_output.get("scope_recommendations", []),
            "cross_validation_notes": synthesizer_output.get("cross_validation_notes", []),
        }

        action_items = [
            ActionItem.from_dict(item)
            for item in synthesizer_output.get("action_items", [])
        ]

        # Filter out synthetic targets leaked from internal pipeline stages
        # (e.g. "_execution_risk" — not a real model)
        synthetic = [ai for ai in action_items if ai.target.startswith("_")]
        if synthetic:
            print(f"   ⚠️  Filtered {len(synthetic)} synthetic target(s): "
                  f"{[ai.target for ai in synthetic]}")
            action_items = [ai for ai in action_items if not ai.target.startswith("_")]

        # Backfill source_signals from the structured Signal list when the
        # Synthesizer LLM didn't populate them (common).  This restores the
        # traceability link between ActionItems and upstream diagnostic Signals.
        signals_by_target: dict = {}
        for sig in signals:
            signals_by_target.setdefault(sig.target, []).append(sig.signal_type)
        for ai in action_items:
            if not ai.source_signals and ai.target in signals_by_target:
                ai.source_signals = list(dict.fromkeys(
                    signals_by_target[ai.target]
                ))  # deduplicated, order-preserving

    # --- Validate ---
    print("\n🔍 Validating action items...")
    validator = ActionItemValidator(
        feedback_scope_path=os.path.join(workspace_root, 'config', 'feedback_scope.json'),
        hyperparam_bounds_path=os.path.join(workspace_root, 'config', 'hyperparam_bounds.json'),
        workspace_root=workspace_root,
    )
    action_items = validator.validate(action_items)
    n_in = sum(1 for a in action_items if a.scope_status == 'in_scope')
    n_out = sum(1 for a in action_items if a.scope_status == 'out_of_scope')
    n_rej = sum(1 for a in action_items if a.scope_status == 'rejected')
    print(f"   → {n_in} in-scope, {n_out} out-of-scope, {n_rej} rejected")

    # --- Persist ---
    if not dry_run:
        print("\n💾 Persisting action items...")
        snap_path = persist_action_items(action_items, workspace_root, run_date=data_date, run_label=run_label)
        print(f"   → Saved to {snap_path}")

        if synthesizer_output:
            _persist_feedback_report(synthesizer_output, feedback_eval, workspace_root, data_date, run_label=run_label)
    else:
        print("\n🏜️  Dry-run mode — action items not persisted.")
        for ai in action_items:
            status_icon = {'in_scope': '✅', 'out_of_scope': '⚠️', 'rejected': '❌'}.get(ai.scope_status, '❓')
            print(f"   {status_icon} [{ai.scope_status}] {ai.action_type}: {ai.target} — {ai.reason[:80]}")

    return action_items


# ------------------------------------------------------------------
# Helpers: model/combo profile builders for layered pipeline
# ------------------------------------------------------------------

def _build_model_profiles(
    model_names: list,
    triage_input: dict,
    signals: list,
    all_findings: list,
    workspace_root: str,
) -> dict:
    """
    Build a per-model profile dict for each model in model_names.

    Each profile contains: training_history, ranking_table, family_stats,
    correlation_excerpt, combo_role, signals, current_params, hyperparam_bounds.
    """
    import yaml

    profiles = {}

    # Index signals by target
    signals_by_target: dict = {}
    for s in signals:
        signals_by_target.setdefault(s.target, []).append(s)

    # Load combo membership
    combo_membership = _load_combo_membership(workspace_root)

    # Load training history
    training_by_model = _load_training_history_by_model(workspace_root)

    # Load hyperparam bounds
    bounds = _load_hyperparam_bounds(workspace_root)

    # Load current params from model_registry + YAML files
    current_params = _load_current_params(workspace_root, model_names)

    # Load correlation matrix excerpt (if available)
    corr_excerpts = _load_correlation_excerpts(workspace_root, model_names)

    # Load diversity signals (orthogonal/diversifier info from combo groups + corr matrix)
    diversity_signals = _load_diversity_signals(workspace_root, model_names)

    # Load per-model tuning knowledge (experiment history + architecture priors)
    model_knowledge = _load_model_knowledge(workspace_root)

    for model_name in model_names:
        # Combo role
        combos = combo_membership.get(model_name, [])
        combo_role = {
            "in_combos": combos,
            "is_active": len(combos) > 0,
        }

        profiles[model_name] = {
            "training_history": training_by_model.get(model_name, []),
            "ranking_table": triage_input.get("model_ranking", []),
            "family_stats": triage_input.get("family_stats", {}),
            "correlation_excerpt": corr_excerpts.get(model_name, {}),
            "combo_role": combo_role,
            "signals": signals_by_target.get(model_name, []),
            "current_params": current_params.get(model_name, {}),
            "hyperparam_bounds": bounds,
            "diversity_signals": diversity_signals.get(model_name, {}),
            "tuning_knowledge": model_knowledge.get(model_name, {}),
        }

    return profiles


def _build_combo_profile(
    combo_name: str,
    model_diagnoses: dict,
    triage_input: dict,
    all_findings: list,
    workspace_root: str = "",
) -> dict:
    """Build a per-combo profile with member diagnoses and history."""
    combo_summary = triage_input.get("combo_summary", {})
    combo_info = combo_summary.get(combo_name, {})

    # Build {combo → [models]} reverse index to filter diagnoses
    combo_members: set = set()
    if workspace_root:
        membership = _load_combo_membership(workspace_root)
        for model, combos in membership.items():
            if combo_name in combos:
                combo_members.add(model)

    # Gather member diagnoses — only for models that belong to this combo
    member_diagnoses = {}
    for diag_name, diag in model_diagnoses.items():
        if combo_members and diag_name not in combo_members:
            continue
        member_diagnoses[diag_name] = {
            "diagnosis": diag.get("diagnosis", "?"),
            "detail": diag.get("diagnosis_detail", "")[:300],
        }

    return {
        "member_diagnoses": member_diagnoses,
        "combo_history": [
            {"date": combo_info.get("latest_date", ""),
             "excess": combo_info.get("latest_excess"),
             "calmar": combo_info.get("latest_calmar"),
             "sharpe": combo_info.get("latest_sharpe")}
        ],
        "loo_analysis": {},  # populated from ensemble_eval findings below
        "pairwise_correlation": {},
        "oos_trend": combo_summary.get("_oos_trend", {}),
        "market_context": triage_input.get("market_context", {}),
    }


def _build_execution_profile(all_findings: list, triage_input: dict) -> dict:
    """Build an execution/risk profile from agent findings for the dedicated LLM call.

    ``all_findings`` is a list of ``AgentFindings`` dataclass instances (from the
    Coordinator).  Each has an ``agent_name`` field (e.g. "execution_quality") and
    a ``findings`` list of ``Finding`` dataclasses (with ``severity``, ``title``,
    ``detail``).
    """
    exec_issues = []
    trade_issues = []

    for af in all_findings:
        # AgentFindings is a dataclass — use attribute access
        agent_name = getattr(af, 'agent_name', '')
        findings_list = getattr(af, 'findings', [])

        for f in findings_list:
            # Finding is a dataclass with: severity, category, title, detail, data
            sev = getattr(f, 'severity', '')
            title = getattr(f, 'title', '')
            detail = getattr(f, 'detail', '')
            entry = {
                "severity": sev,
                "title": title[:200] if title else "",
                "detail": detail[:500] if detail else "",
            }
            if 'execution' in agent_name.lower():
                exec_issues.append(entry)
            elif 'trade' in agent_name.lower():
                trade_issues.append(entry)

    return {
        "execution_issues": exec_issues,
        "trade_pattern_issues": trade_issues,
        "_execution_context": triage_input.get("market_context", {}),
    }


def _build_execution_risk_summary(all_findings: list, triage_input: dict) -> dict:
    """Extract execution/risk highlights from agent findings (no LLM call).

    Aggregates key metrics from execution_quality and trade_pattern agents
    into a lightweight summary the Synthesizer can reference.

    ``all_findings`` is a list of ``AgentFindings`` dataclass instances.
    """
    summary = {
        "diagnosis": "rule_based",
        "diagnosis_detail": "",
        "execution_issues": [],
        "trade_pattern_issues": [],
    }

    for af in all_findings:
        agent_name = getattr(af, 'agent_name', '')
        findings_list = getattr(af, 'findings', [])

        for f in findings_list:
            sev = getattr(f, 'severity', '')
            title = getattr(f, 'title', '')
            if sev in ('high', 'critical'):
                issue_text = title[:200] if title else ""
                if 'execution' in agent_name.lower() and issue_text:
                    summary["execution_issues"].append(issue_text)
                elif 'trade' in agent_name.lower() and issue_text:
                    summary["trade_pattern_issues"].append(issue_text)

    summary["diagnosis_detail"] = (
        f"Execution issues: {len(summary['execution_issues'])}, "
        f"Trade pattern issues: {len(summary['trade_pattern_issues'])}"
    )

    mc = triage_input.get("market_context", {})
    if mc and isinstance(mc, dict):
        summary["market_context"] = {
            "n_regime_changes": mc.get("n_regime_changes", 0),
        }

    return summary


# ------------------------------------------------------------------
# Helpers: data loaders for layered pipeline
# ------------------------------------------------------------------

def _load_model_knowledge(workspace_root: str) -> dict:
    """Load per-model tuning knowledge from config/model_knowledge.yaml.

    Returns a dict mapping model_name → knowledge dict with keys:
    architecture_family, regularization_direction, tuning_notes,
    known_effective_params, known_ineffective_params, preferred_param_ranges.

    This knowledge is injected into the Per-Model Critic LLM prompt so it
    can make architecture-aware and history-informed tuning suggestions.
    """
    import yaml as _yaml

    path = os.path.join(workspace_root, 'config', 'model_knowledge.yaml')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            doc = _yaml.safe_load(f)
        return doc.get('models', {}) if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _load_combo_membership(workspace_root: str) -> dict:
    """Load {model_name: [combo_names]} from ensemble_config.json."""
    path = os.path.join(workspace_root, 'config', 'ensemble_config.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            config = json.load(f)
    except Exception:
        return {}
    membership = {}
    for combo_name, combo_info in config.get('combos', {}).items():
        if not isinstance(combo_info, dict):
            continue
        for m in combo_info.get('models', []):
            if isinstance(m, str):
                membership.setdefault(m, []).append(combo_name)
    return membership


def _load_training_history_by_model(workspace_root: str) -> dict:
    """Load training_history.jsonl grouped by model_name."""
    path = os.path.join(workspace_root, 'data', 'training_history.jsonl')
    if not os.path.exists(path):
        return {}
    result = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                model = rec.get('model_name', '')
                # Keep only relevant fields to control prompt size
                clean = {
                    k: rec.get(k) for k in [
                        'trained_at', 'anchor_date', 'duration_seconds',
                        'early_stopped', 'actual_epochs', 'configured_epochs',
                        'best_epoch', 'final_val_score', 'score_type',
                        'n_epochs', 'early_stop', 'lr', 'learning_rate',
                        'batch_size', 'dropout', 'hidden_size', 'num_layers',
                    ] if k in rec
                }
                result.setdefault(model, []).append(clean)
    except Exception:
        pass
    return result


def _load_hyperparam_bounds(workspace_root: str) -> dict:
    """Load hyperparam bounds from config."""
    path = os.path.join(workspace_root, 'config', 'hyperparam_bounds.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('bounds', {})
    except Exception:
        return {}


def _load_current_params(workspace_root: str, model_names: list) -> dict:
    """Load current hyperparam values for the given models from YAML files."""
    import yaml

    registry_path = os.path.join(workspace_root, 'config', 'model_registry.yaml')
    registry = {}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                registry = yaml.safe_load(f)
        except Exception:
            pass
    registry = registry.get('models', {}) if isinstance(registry, dict) else {}

    current = {}
    for model_name in model_names:
        model_info = registry.get(model_name, {})
        yaml_file = model_info.get('yaml_file', '')
        if not yaml_file:
            continue
        yaml_path = os.path.join(workspace_root, yaml_file)
        if not os.path.exists(yaml_path):
            continue
        try:
            with open(yaml_path, 'r') as f:
                cfg = yaml.safe_load(f)
            kwargs = cfg.get('task', {}).get('model', {}).get('kwargs', {})
            clean = {
                k: v for k, v in kwargs.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            if clean:
                current[model_name] = clean
        except Exception:
            pass

    return current


def _find_best_correlation_csv(workspace_root: str) -> Optional[str]:
    """Find the best correlation matrix CSV under output/.

    Strategy: pick the file with the most columns (= most complete model set).
    On ties, pick the latest by filename sort order (date suffix).
    Returns None if no file found.
    """
    corr_files = []
    for root, dirs, files in os.walk(os.path.join(workspace_root, 'output')):
        for fname in files:
            if 'correlation' in fname.lower() and fname.endswith(('.csv', '.json')):
                corr_files.append(os.path.join(root, fname))
    if not corr_files:
        return None
    corr_files.sort()

    best_file = None
    max_cols = 0
    for cf in corr_files:
        try:
            with open(cf, 'r') as f:
                n = len(f.readline().split(','))
            if n > max_cols or (n == max_cols and (best_file is None or cf > best_file)):
                max_cols = n
                best_file = cf
        except Exception:
            pass
    return best_file or (corr_files[-1] if corr_files else None)


def _strip_static_suffix(name: str) -> str:
    """Strip @static / @rolling suffix from model name."""
    return str(name).rsplit('@', 1)[0] if '@' in str(name) else str(name)


def _load_correlation_df(workspace_root: str):
    """Load and normalize the best correlation matrix as a DataFrame.

    Returns (DataFrame, True) on success, (None, False) on failure.
    Normalizes index/column names by stripping @static suffixes.
    """
    import pandas as pd

    csv_path = _find_best_correlation_csv(workspace_root)
    if not csv_path:
        return None, False
    try:
        df = pd.read_csv(csv_path, index_col=0)
        df.index = [_strip_static_suffix(i) for i in df.index]
        df.columns = [_strip_static_suffix(c) for c in df.columns]
        return df, True
    except Exception:
        return None, False


def _load_correlation_excerpts(workspace_root: str, model_names: list) -> dict:
    """Load top/bottom correlation excerpts for given models."""
    import pandas as pd

    corr_df, ok = _load_correlation_df(workspace_root)
    if not ok or corr_df is None:
        return {}

    excerpts = {}
    for model in model_names:
        if model not in corr_df.index:
            continue
        row = corr_df.loc[model]
        # Handle duplicate rows (e.g. @static + @rolling after suffix strip)
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        row = row.drop(model, errors='ignore')
        if not isinstance(row, pd.Series) or row.empty:
            continue
        numeric_row = pd.to_numeric(row, errors='coerce').dropna()
        if numeric_row.empty:
            continue
        sorted_corr = numeric_row.sort_values(ascending=False)
        excerpts[model] = {
            "top_correlated": [
                {"model": m, "correlation": round(float(v), 3)}
                for m, v in sorted_corr.head(3).items()
            ],
            "bottom_correlated": [
                {"model": m, "correlation": round(float(v), 3)}
                for m, v in sorted_corr.tail(3).items()
            ],
        }

    return excerpts


def _load_diversity_signals(workspace_root: str, model_names: list) -> dict:
    """Load orthogonal/diversifier signals from combo_groups YAML and correlation matrix.

    Returns a dict mapping model_name → {avg_corr, group_label, is_diversifier}.

    Sources:
    - ``combo_groups_27.yaml``: group label per model (with @static suffix stripped).
    - Correlation CSV: average pairwise correlation across all other models.

    Models with avg_corr < 0.15 are flagged as diversifiers, meaning their
    standalone IC is a poor signal — they may add value through orthogonality.
    """
    import yaml as _yaml
    import pandas as pd

    diversity = {}

    # 1. Load correlation matrix to compute avg_corr per model
    avg_corrs: dict = {}
    corr_df, ok = _load_correlation_df(workspace_root)
    if ok and corr_df is not None:
        try:
            for model in model_names:
                if model in corr_df.index:
                    row = corr_df.loc[model]
                    # Handle duplicate rows (e.g. @static + @rolling)
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    row = row.drop(model, errors='ignore')
                    if isinstance(row, pd.Series):
                        numeric_row = pd.to_numeric(row, errors='coerce').dropna()
                        if len(numeric_row) > 0:
                            avg_corrs[model] = float(numeric_row.mean())
        except Exception:
            pass

    # 2. Load combo_groups_27.yaml for group labels
    group_labels: dict = {}
    groups_path = os.path.join(workspace_root, 'config', 'combo_groups_27.yaml')
    if os.path.exists(groups_path):
        try:
            with open(groups_path, 'r', encoding='utf-8') as f:
                groups_doc = _yaml.safe_load(f)
            for group_name, members in (groups_doc.get('groups', {}) or {}).items():
                for member_entry in members:
                    if isinstance(member_entry, str):
                        # Strip @static suffix
                        member_name = member_entry.rsplit('@', 1)[0] if '@' in member_entry else member_entry
                        group_labels[member_name] = group_name
        except Exception:
            pass

    # 3. Build diversity signals per model
    for model_name in model_names:
        avg_corr = avg_corrs.get(model_name, None)
        group_label = group_labels.get(model_name, None)
        is_diversifier = (avg_corr is not None and avg_corr < 0.15)

        diversity[model_name] = {
            "avg_corr": round(avg_corr, 4) if avg_corr is not None else None,
            "group_label": group_label,
            "is_diversifier": is_diversifier,
            "diversifier_note": (
                "This model has very low average correlation (avg_corr < 0.15) with the model pool. "
                "Its standalone IC may be low, but it CAN provide orthogonal diversification value "
                "in ensembles. Do NOT recommend disable_model based solely on low IC — LOO delta "
                "evidence is REQUIRED to prove this model actually harms the ensemble."
            ) if is_diversifier else "",
        }

    return diversity


def _persist_feedback_report(
    synthesizer_output: dict,
    feedback_eval: dict,
    workspace_root: str,
    data_date: str = "",
    run_label: str = "",
) -> None:
    """Persist synthesizer output as feedback_report_{date}[_{label}].json."""
    date_str = data_date or datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(workspace_root, 'output', 'deep_analysis')
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "date": date_str,
        "global_diagnosis": synthesizer_output.get("global_diagnosis", {}),
        "conflict_resolutions": synthesizer_output.get("conflict_resolutions", []),
        "cross_validation_notes": synthesizer_output.get("cross_validation_notes", []),
        "scope_recommendations": synthesizer_output.get("scope_recommendations", []),
        "feedback_summary": feedback_eval.get("quality_summary", {}),
    }

    _label_part = f"_{run_label}" if run_label else ""
    path = os.path.join(output_dir, f"feedback_report_{date_str}{_label_part}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"   → Feedback report saved to {path}")


def _resolve_data_date() -> str:
    """Query Qlib for the latest trading day — the canonical data clock.

    Falls back to today if Qlib is unavailable.
    """
    try:
        from quantpits.scripts.analysis.utils import init_qlib
        from qlib.data import D
        init_qlib()
        cal = D.calendar()
        if len(cal) > 0:
            return cal[-1].strftime("%Y-%m-%d")
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d")


def main():
    args = parse_args()
    workspace_root = env.ROOT_DIR
    data_date = _resolve_data_date()
    run_label = re.sub(r'[\\/]', '-', args.run_label.strip())

    # Ensure args are not MagicMocks from tests
    def clean_arg(val, default):
        if val is None:
            return default
        if hasattr(val, '_mock_return_value') or 'mock' in str(type(val)).lower():
            return default
        return val

    target_stage = clean_arg(args.stage, "all")
    resume_path = clean_arg(args.resume_from, None)
    resume_latest = clean_arg(getattr(args, 'resume_latest', False), False)
    manifest_arg = clean_arg(getattr(args, 'manifest', None), None)

    print("=" * 60)
    print("  🤖 MAS Deep Analysis System")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data date: {data_date}")
    print(f"  Workspace: {workspace_root}")
    print(f"  Stage: {target_stage}")
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
    # --- Initialize StageRunner ---
    from quantpits.scripts.deep_analysis.stage_runner import StageRunner
    stage_runner = StageRunner(workspace_root, data_date, run_label=run_label)

    if resume_latest:
        latest_ckpt = None
        latest_mtime = 0
        for stg in StageRunner.STAGES:
            ckpt = stage_runner.find_latest_checkpoint(stg)
            if ckpt:
                mtime = os.path.getmtime(ckpt)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_ckpt = ckpt
        if latest_ckpt:
            resume_path = latest_ckpt
            print(f"   🔄 Found latest checkpoint to resume: {resume_path}")
        else:
            print("   ⚠️  No latest checkpoint found. Running from start.")

    start_stage_idx = 0
    stage_data = {stg: None for stg in StageRunner.STAGES}

    if resume_path:
        loaded = stage_runner.load_checkpoint(resume_path)
        meta = loaded["meta"]
        ckpt_stage = meta["stage"]
        print(f"   🔄 Resuming from stage '{ckpt_stage}' checkpoint: {resume_path}")
        stage_data[ckpt_stage] = loaded["data"]
        start_stage_idx = StageRunner.STAGES.index(ckpt_stage) + 1

    # Check if a specific stage was requested
    requested_stage_name = target_stage
    specific_agents = []
    run_only_one_stage = False

    # Determine stage target and optional agent filter
    specific_agents = None
    stage_target = target_stage
    if target_stage != "all":
        if target_stage.startswith("agents:"):
            stage_target = "agents"
            specific_agents = [a.strip() for a in target_stage.split(":")[1].split(",")]
        if stage_target not in StageRunner.STAGES and stage_target != "all":
            print(f"❌ Unknown stage: {target_stage}. Available: {StageRunner.STAGES}")
            return 1

    # Resolve final target stage name for runner
    runner_target = "report" if target_stage == "all" else stage_target

    # Register built-in stages (import triggers @register_stage decorators)
    import quantpits.scripts.deep_analysis.stages  # noqa: F401

    # Initialize state
    from quantpits.scripts.deep_analysis.state import DeepAnalysisState
    state = DeepAnalysisState(
        workspace_root=workspace_root,
        run_date=data_date,
        run_label=run_label,
        external_notes=external_notes,
        freq_change_date=freq_change_date,
    )

    # Handle resume
    if resume_path:
        loaded = stage_runner.load_checkpoint(resume_path)
        state = DeepAnalysisState.from_dict(loaded["data"])
        print(f"   🔄 Resumed state from: {resume_path}")

    # Handle agents:NAME syntax — narrow target to 'agents'
    if target_stage.startswith("agents:"):
        runner_target = "agents"
        # Mark upstream stages as completed since they provide the window data
        # needed by agents stage
        pass  # let resolve_plan figure out from checkpoints

    try:
        state = stage_runner.run(
            target=runner_target,
            state=state,
            workspace_root=workspace_root,
            manifest_path=manifest_arg,
            # CLI args forwarded to stages
            snapshot_config=not clean_arg(getattr(args, 'no_snapshot', False), False),
            no_snapshot=clean_arg(getattr(args, 'no_snapshot', False), False),
            agent_filter=clean_arg(args.agents, 'all'),
            specific_agents=specific_agents,
            windows=windows,
            critic_enabled=clean_arg(getattr(args, 'critic', False), False)
                          or clean_arg(getattr(args, 'critic_dry_run', False), False),
            critic_dry_run=clean_arg(getattr(args, 'critic_dry_run', False), False),
            enable_llm_summary=clean_arg(getattr(args, 'llm', False), False),
            llm_model=clean_arg(getattr(args, 'llm_model', None), None),
            api_key=clean_arg(getattr(args, 'api_key', None), None),
            base_url=clean_arg(getattr(args, 'base_url', None), None),
            shareable=clean_arg(getattr(args, 'shareable', False), False),
            run_date=data_date,
            run_label=run_label,
            output=clean_arg(args.output, 'output/deep_analysis_report.md'),
        )
    finally:
        # Clean up workspace-local stages
        from quantpits.scripts.deep_analysis.stage_runner import (
            unregister_workspace_stages,
        )
        unregister_workspace_stages(workspace_root)

    if state.output_path:
        print(f"\n✅ Report saved to: {state.output_path}")
        if state.synthesis_result:
            print(f"   Health: {state.synthesis_result.get('health_status', 'Unknown')}")
    print(f"✅ Pipeline finished at stage: {state.last_completed_stage or runner_target}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
