"""
Stage 6: critic — LLM-powered action item generation and executive summary.

This stage wraps the Critic helpers currently defined in
``run_deep_analysis.py``.  The helpers will be moved here in a follow-up
to keep this migration incremental and testable.
"""

import os
import json

from ..stage_runner import register_stage


@register_stage(
    name='critic',
    depends_on=['agents', 'synthesis', 'window_analysis', 'signals'],
    provides=['action_items', 'executive_summary'],
    description='LLM Critic: generate actionable optimisation items from structured signals',
)
def run_critic(state, **kwargs):
    """Run LLM Critic (single-stage or layered) + generate executive summary.

    Populates ``state.action_items``, ``state.executive_summary``.
    """
    from quantpits.scripts.deep_analysis.signal_extractor import SignalExtractor
    from quantpits.scripts.deep_analysis.llm_interface import LLMInterface
    from quantpits.scripts.deep_analysis.llm_trace import LLMTraceLogger
    from quantpits.scripts.deep_analysis.langfuse_adapter import LangfuseAdapter

    workspace_root = kwargs.get('workspace_root', state.workspace_root)
    run_date = kwargs.get('run_date', state.run_date)
    run_label = kwargs.get('run_label', state.run_label)
    critic_enabled = kwargs.get('critic_enabled', False)
    critic_dry_run = kwargs.get('critic_dry_run', False)
    enable_llm_summary = kwargs.get('enable_llm_summary', False)
    api_key = kwargs.get('api_key', None)
    llm_model = kwargs.get('llm_model', None)
    base_url = kwargs.get('base_url', None)

    action_items = []

    # -- SignalExtractor for layered critic --
    signal_extractor = SignalExtractor(
        reference_date=run_date,
        workspace_root=workspace_root,
        window_analysis_findings=state.window_findings,
        window_analysis_context=state.window_analysis_context
        if state.window_analysis_context else None,
    )

    if critic_enabled or critic_dry_run:
        # — LLM setup —
        skills_dir = os.path.join(workspace_root, 'config', 'skills')
        layered_available = all(
            os.path.exists(os.path.join(skills_dir, f))
            for f in ['triage_system.md', 'model_critic_system.md',
                       'combo_critic_system.md', 'synthesizer_system.md']
        )

        ws_llm_cfg = {}
        llm_cfg_path = os.path.join(workspace_root, 'config', 'llm_config.json')
        if os.path.exists(llm_cfg_path):
            try:
                with open(llm_cfg_path) as f:
                    ws_llm_cfg = json.load(f)
            except Exception:
                pass

        langfuse = LangfuseAdapter.from_config(ws_llm_cfg)
        trace_logger = LLMTraceLogger.from_llm_config(
            llm_config=ws_llm_cfg,
            workspace_root=workspace_root,
            run_date=run_date,
            workspace_name=os.path.basename(workspace_root),
            pipeline_stage="layered" if layered_available else "single_stage",
            langfuse_adapter=langfuse,
            run_label=run_label,
        )

        critic_llm = LLMInterface(
            api_key=api_key,
            model=llm_model,
            base_url=base_url,
            trace_logger=trace_logger,
        )

        # — Window Critic (if benchmark data available) —
        benchmark_data = {}
        try:
            from quantpits.scripts.deep_analysis.benchmark_data_loader import (
                load_benchmark_data,
            )
            benchmark_data = load_benchmark_data(workspace_root)
        except Exception:
            pass

        if (state.window_analysis_context and benchmark_data
                and not benchmark_data.get("error")):
            try:
                from quantpits.scripts.deep_analysis.window_critic import WindowCritic
                wc = WindowCritic(critic_llm, workspace_root)
                wac = state.window_analysis_context

                diagnosis = wc.diagnose(
                    benchmark_data, state.window_findings,
                    model=llm_model, api_key=api_key, base_url=base_url,
                )
                print(f"   🩺 Window diagnosis: {diagnosis.get('root_cause', '?')} "
                      f"[urgency={diagnosis.get('urgency', '?')}]")
                wac["diagnosis"] = diagnosis

                rec = wc.recommend(
                    benchmark_data, diagnosis,
                    model=llm_model, api_key=api_key, base_url=base_url,
                )
                rcfg = rec.get("recommended_config", {})
                print(f"   💡 Window recommendation: train={rcfg.get('train')}, "
                      f"valid={rcfg.get('valid')}, test={rcfg.get('test')}")
                print(f"      Rationale: {rec.get('rationale', '')[:150]}")
                wac["recommendation"] = rec
            except Exception as e:
                print(f"   ⚠️  Window Critic failed: {e}")

        if state.window_analysis_context:
            signal_extractor.set_window_analysis_context(
                state.window_analysis_context)

        # — Run Critic —
        if layered_available and critic_llm.is_available(workspace_root):
            action_items = _run_critic_layered(
                critic_llm=critic_llm,
                signals=state.signals,
                all_findings=state.all_findings,
                synthesis_result=state.synthesis_result,
                signal_extractor=signal_extractor,
                workspace_root=workspace_root,
                data_date=run_date,
                dry_run=critic_dry_run,
                run_label=run_label,
            )
        else:
            if not layered_available:
                print("   ℹ️  Layered skill files not found — using single-stage Critic.")
            action_items = _run_critic_single_stage(
                critic_llm=critic_llm,
                signals=state.signals,
                workspace_root=workspace_root,
                data_date=run_date,
                dry_run=critic_dry_run,
                run_label=run_label,
            )

        trace_dir = trace_logger.finalize()
        if trace_dir:
            print(f"   📋 LLM traces: {trace_dir}")

    # — Executive summary (always runs, LLM or template) —
    print("\n📝 Generating executive summary...")
    if enable_llm_summary:
        ws_cfg = {}
        llm_cfg_p = os.path.join(workspace_root, 'config', 'llm_config.json')
        if os.path.exists(llm_cfg_p):
            try:
                with open(llm_cfg_p) as f:
                    ws_cfg = json.load(f)
            except Exception:
                pass
        s_langfuse = LangfuseAdapter.from_config(ws_cfg)
        s_trace = LLMTraceLogger.from_llm_config(
            llm_config=ws_cfg,
            workspace_root=workspace_root,
            run_date=run_date,
            workspace_name=os.path.basename(workspace_root),
            pipeline_stage="summary",
            langfuse_adapter=s_langfuse,
            run_label=run_label,
        )
        llm = LLMInterface(
            api_key=api_key, model=llm_model, base_url=base_url,
            trace_logger=s_trace,
        )
    else:
        s_trace = None
        llm = LLMInterface()

    executive_summary = llm.generate_executive_summary(
        state.synthesis_result, workspace_root)

    if s_trace is not None:
        s_dir = s_trace.finalize()
        if s_dir:
            print(f"   📋 Summary traces: {s_dir}")

    # Dedup title
    executive_summary = _dedup_executive_summary_title(executive_summary)

    # If Critic mutated synthesis_result, persist it back
    state.synthesis_result = state.synthesis_result or {}

    state.action_items = action_items
    state.executive_summary = executive_summary
    return state


# ------------------------------------------------------------------
# Helpers (imported from run_deep_analysis.py for now —
# will be moved here in a follow-up pass)
# ------------------------------------------------------------------

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


def _run_critic_single_stage(
    critic_llm, signals: list, workspace_root: str,
    data_date: str = "", dry_run: bool = False, run_label: str = "",
) -> list:
    from quantpits.scripts.run_deep_analysis import _run_critic_single_stage as _fn
    return _fn(critic_llm, signals, workspace_root, data_date, dry_run, run_label)


def _run_critic_layered(
    critic_llm, signals: list, all_findings: list,
    synthesis_result: dict, signal_extractor,
    workspace_root: str, data_date: str = "",
    dry_run: bool = False, run_label: str = "",
) -> list:
    from quantpits.scripts.run_deep_analysis import _run_critic_layered as _fn
    return _fn(critic_llm, signals, all_findings, synthesis_result,
               signal_extractor, workspace_root, data_date, dry_run, run_label)
