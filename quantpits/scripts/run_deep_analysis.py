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
        snap_path = persist_action_items(action_items, workspace_root, run_date=data_date)
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
        snap_path = persist_action_items(action_items, workspace_root, run_date=data_date)
        print(f"   → Saved to {snap_path}")

        if synthesizer_output:
            _persist_feedback_report(synthesizer_output, feedback_eval, workspace_root, data_date)
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


def _load_correlation_excerpts(workspace_root: str, model_names: list) -> dict:
    """Load top/bottom correlation excerpts for given models."""
    # Search recursively under output/ for the latest correlation matrix
    corr_files = []
    for root, dirs, files in os.walk(os.path.join(workspace_root, 'output')):
        for fname in files:
            if 'correlation' in fname.lower() and fname.endswith(('.csv', '.json')):
                corr_files.append(os.path.join(root, fname))
    corr_files.sort()

    if not corr_files:
        return {}

    try:
        import pandas as pd
        df = pd.read_csv(corr_files[-1], index_col=0)
    except Exception:
        return {}

    excerpts = {}
    for model in model_names:
        if model not in df.index:
            continue
        row = df.loc[model].drop(model, errors='ignore')
        sorted_corr = row.sort_values(ascending=False)
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

    diversity = {}

    # 1. Load correlation matrix to compute avg_corr per model
    # Search recursively under output/ for the latest correlation matrix CSV
    corr_files = []
    for root, dirs, files in os.walk(os.path.join(workspace_root, 'output')):
        for fname in files:
            if 'correlation' in fname.lower() and fname.endswith(('.csv', '.json')):
                corr_files.append(os.path.join(root, fname))
    corr_files.sort()

    avg_corrs: dict = {}
    if corr_files:
        try:
            import pandas as pd
            # Use the file with the most columns (full model set, not combo-specific)
            best_file = None
            max_cols = 0
            for cf in corr_files:
                try:
                    with open(cf, 'r') as f:
                        n = len(f.readline().split(','))
                    if n > max_cols:
                        max_cols = n
                        best_file = cf
                except Exception:
                    pass
            if best_file is None:
                best_file = corr_files[-1]

            corr_df = pd.read_csv(best_file, index_col=0)
            # Strip @static suffix from column/index names
            def _base(name):
                return str(name).rsplit('@', 1)[0] if '@' in str(name) else str(name)

            corr_df.index = [_base(i) for i in corr_df.index]
            corr_df.columns = [_base(c) for c in corr_df.columns]

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
) -> None:
    """Persist synthesizer output as feedback_report_{date}.json."""
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

    path = os.path.join(output_dir, f"feedback_report_{date_str}.json")
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

    print("=" * 60)
    print("  🤖 MAS Deep Analysis System")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Data date: {data_date}")
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

    # --- 5.5. Signal Extraction ---
    print("\n📡 Extracting structured signals...")
    from quantpits.scripts.deep_analysis.signal_extractor import SignalExtractor

    signal_extractor = SignalExtractor(reference_date=data_date, workspace_root=workspace_root)
    signals = signal_extractor.extract(all_findings, synthesis_result)
    print(f"   → {len(signals)} signals extracted")

    # --- 5.6–5.8. Critic + Validation + Persist (if --critic) ---
    if args.critic or args.critic_dry_run:
        from quantpits.scripts.deep_analysis.llm_interface import LLMInterface as _LLMInterface

        critic_llm = _LLMInterface(
            api_key=args.api_key,
            model=args.llm_model,
            base_url=args.base_url,
        )

        # Detect layered skill files
        _skills_dir = os.path.join(workspace_root, 'config', 'skills')
        _layered_skills_available = all(
            os.path.exists(os.path.join(_skills_dir, f))
            for f in ['triage_system.md', 'model_critic_system.md',
                       'combo_critic_system.md', 'synthesizer_system.md']
        )

        if _layered_skills_available and critic_llm.is_available(workspace_root):
            action_items = _run_critic_layered(
                critic_llm=critic_llm,
                signals=signals,
                all_findings=all_findings,
                synthesis_result=synthesis_result,
                signal_extractor=signal_extractor,
                workspace_root=workspace_root,
                data_date=data_date,
                dry_run=args.critic_dry_run,
            )
        else:
            if not _layered_skills_available:
                print("   ℹ️  Layered skill files not found — using single-stage Critic.")
            action_items = _run_critic_single_stage(
                critic_llm=critic_llm,
                signals=signals,
                workspace_root=workspace_root,
                data_date=data_date,
                dry_run=args.critic_dry_run,
            )

    # --- 6. LLM Executive Summary ---
    print("\n📝 Generating executive summary...")
    from quantpits.scripts.deep_analysis.llm_interface import LLMInterface

    if args.llm:
        llm = LLMInterface(
            api_key=args.api_key,
            model=args.llm_model,
            base_url=args.base_url,
        )
    else:
        llm = LLMInterface()  # No API key → template mode

    executive_summary = llm.generate_executive_summary(synthesis_result, workspace_root)

    # Strip leading "Executive Summary" title markers from LLM output to
    # avoid double headers — ReportGenerator already emits the section title.
    def _dedup_executive_summary_title(text: str) -> str:
        if not text or not isinstance(text, str):
            return text
        lines = text.split("\n")
        import re
        while lines:
            stripped = lines[0].strip()
            if (stripped.lower() in ("**executive summary**", "executive summary")
                    or re.match(r'^#{1,3}\s+executive\s+summary', stripped, re.I)):
                lines.pop(0)
                if lines and not lines[0].strip():
                    lines.pop(0)
            else:
                break
        return "\n".join(lines)
    executive_summary = _dedup_executive_summary_title(executive_summary)

    # --- 7. Generate Report ---
    print("\n📊 Generating report...")
    from quantpits.scripts.deep_analysis.report_generator import ReportGenerator

    report_gen = ReportGenerator(all_findings, synthesis_result, executive_summary)
    report_md = report_gen.generate()

    # Write output — inject date into filename if not already present
    output_path = os.path.join(workspace_root, args.output)
    _base, _ext = os.path.splitext(output_path)
    if data_date not in _base:
        output_path = f"{_base}_{data_date}{_ext}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_md)

    print(f"\n✅ Report saved to: {output_path}")
    print(f"   Health: {synthesis_result.get('health_status', 'Unknown')}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
