"""Single-combo ensemble pipeline orchestration."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SingleComboPipelineRequest:
    """Inputs for one ensemble combo execution."""

    combo_name: str | None
    selected_models: Sequence[str]
    method: str
    manual_weights_str: str | None
    norm_df: pd.DataFrame
    model_metrics: Mapping[str, float]
    loaded_models: Sequence[str]
    train_records: Mapping[str, Any]
    model_config: Mapping[str, Any]
    ensemble_config: Mapping[str, Any]
    anchor_date: str
    experiment_name: str
    args: Any
    is_default: bool = False


@dataclass(frozen=True)
class SingleComboPipelineResult:
    """Legacy-compatible result data for one ensemble combo."""

    name: str
    models: tuple[str, ...]
    method: str
    is_default: bool
    pred_file: str
    report_df: pd.DataFrame | None
    leaderboard_df: pd.DataFrame | None
    contributions: Mapping[str, Mapping[str, float]]
    recorder_id: str | None = None
    output_evidence: Mapping[str, Any] | None = None

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "models": list(self.models),
            "method": self.method,
            "is_default": self.is_default,
            "pred_file": self.pred_file,
            "report_df": self.report_df,
            "leaderboard_df": self.leaderboard_df,
            "contributions": dict(self.contributions),
            "recorder_id": self.recorder_id,
            "output_evidence": dict(self.output_evidence or {}),
        }


@dataclass(frozen=True)
class SingleComboPipelineHooks:
    """Injected stage hooks used by the single-combo pipeline."""

    correlation_analysis: Callable[..., Any]
    calculate_weights: Callable[..., tuple[Any, Mapping[str, float] | None, bool]]
    generate_ensemble_signal: Callable[..., Any]
    save_predictions: Callable[..., str]
    run_backtest: Callable[..., tuple[pd.DataFrame | None, Any]]
    run_detailed_backtest_analysis: Callable[..., Any]
    risk_analysis_and_leaderboard: Callable[..., tuple[dict, pd.DataFrame | None]]
    generate_charts: Callable[..., Any]
    calculate_loo_contribution: Callable[..., Mapping[str, Mapping[str, float]]]
    save_model_contribution_snapshot: Callable[..., Any]
    append_to_fusion_ledger: Callable[..., Any]
    get_workspace_root: Callable[[], str]


def _default_correlation_analysis(norm_df, output_dir, anchor_date, combo_name=None):
    from quantpits.ensemble.analytics import (
        CorrelationAnalysisRequest,
        run_correlation_analysis,
    )

    result = run_correlation_analysis(
        CorrelationAnalysisRequest(
            norm_df=norm_df,
            output_dir=output_dir,
            anchor_date=anchor_date,
            combo_name=combo_name,
        )
    )
    return result.matrix


def _default_calculate_weights(
    norm_df,
    model_metrics,
    method,
    model_config,
    ensemble_config,
    manual_weights_str=None,
):
    from quantpits.utils.fusion_engine import calculate_weights

    return calculate_weights(
        norm_df,
        model_metrics,
        method,
        model_config,
        ensemble_config,
        manual_weights_str,
    )


def _default_generate_ensemble_signal(norm_df, final_weights, static_weights, is_dynamic):
    from quantpits.utils.fusion_engine import generate_ensemble_signal

    return generate_ensemble_signal(norm_df, final_weights, static_weights, is_dynamic)


def _default_save_predictions(
    final_score,
    anchor_date,
    experiment_name,
    method,
    model_names,
    model_metrics,
    static_weights,
    is_dynamic,
    output_dir,
    combo_name=None,
    is_default=False,
    prediction_dir=None,
    save_csv=False,
    workspace_root=None,
    source_recorders=None,
    source_anchors=None,
    run_id=None,
    plan_fingerprint=None,
):
    from quantpits.ensemble.persistence import (
        PredictionSaveRequest,
        save_ensemble_predictions,
    )

    result = save_ensemble_predictions(
        PredictionSaveRequest(
            final_score=final_score,
            anchor_date=anchor_date,
            experiment_name=experiment_name,
            method=method,
            model_names=tuple(model_names),
            model_metrics=model_metrics,
            static_weights=static_weights,
            is_dynamic=is_dynamic,
            output_dir=output_dir,
            combo_name=combo_name,
            is_default=is_default,
            prediction_dir=prediction_dir,
            save_csv=save_csv,
            workspace_root=workspace_root,
            source_recorders=source_recorders,
            source_anchors=source_anchors,
            run_id=run_id,
            plan_fingerprint=plan_fingerprint,
        ),
        output_inspector=__import__(
            "quantpits.ensemble.persistence", fromlist=["inspect_saved_recorder"]
        ).inspect_saved_recorder,
    )
    return result.returned_ref


def _default_append_to_fusion_ledger(
    workspace_root: str,
    run_date: str,
    combo_name: str,
    models: list,
    method: str,
    is_default: bool,
    eval_window: dict,
    metrics: dict,
    sub_model_metrics: dict | None = None,
    loo_contributions: dict | None = None,
    cli_args: list | None = None,
    source_recorders: Mapping[str, str] | None = None,
    source_anchors: Mapping[str, str] | None = None,
    run_id: str | None = None,
    plan_fingerprint: str | None = None,
):
    from quantpits.ensemble.ledger import FusionLedgerEntry, append_fusion_ledger

    append_fusion_ledger(
        workspace_root,
        FusionLedgerEntry(
            run_date=run_date,
            combo_name=combo_name,
            models=tuple(models),
            method=method,
            is_default=is_default,
            eval_window=eval_window,
            metrics=metrics,
            sub_model_metrics=sub_model_metrics or {},
            loo_contributions=loo_contributions or {},
            cli_args=tuple(cli_args or ()),
            source_recorders=source_recorders or {},
            source_anchors=source_anchors or {},
            run_id=run_id,
            plan_fingerprint=plan_fingerprint,
        ),
    )


def _default_workspace_root() -> str:
    from quantpits.utils import env

    return env.get_workspace_context().root.as_posix()


def default_single_combo_pipeline_hooks() -> SingleComboPipelineHooks:
    """Return module-level hooks for non-script callers."""

    from quantpits.ensemble.analytics import (
        calculate_loo_contribution,
        save_model_contribution_snapshot,
    )
    from quantpits.ensemble.backtest import run_backtest, run_detailed_backtest_analysis
    from quantpits.ensemble.charts import generate_charts
    from quantpits.ensemble.risk_report import risk_analysis_and_leaderboard

    return SingleComboPipelineHooks(
        correlation_analysis=_default_correlation_analysis,
        calculate_weights=_default_calculate_weights,
        generate_ensemble_signal=_default_generate_ensemble_signal,
        save_predictions=_default_save_predictions,
        run_backtest=run_backtest,
        run_detailed_backtest_analysis=run_detailed_backtest_analysis,
        risk_analysis_and_leaderboard=risk_analysis_and_leaderboard,
        generate_charts=generate_charts,
        calculate_loo_contribution=calculate_loo_contribution,
        save_model_contribution_snapshot=save_model_contribution_snapshot,
        append_to_fusion_ledger=_default_append_to_fusion_ledger,
        get_workspace_root=_default_workspace_root,
    )


def _arg_value(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _valid_workspace_root(value: Any) -> str | os.PathLike | None:
    return value if isinstance(value, (str, os.PathLike)) else None


def _combo_models(selected_models: Sequence[str], norm_df: pd.DataFrame) -> list[str]:
    from quantpits.ensemble.input_integrity import assert_exact_members

    if not selected_models:
        return []
    if norm_df is None:
        assert_exact_members(tuple(selected_models), (), layer="single-combo dataframe")
    actual = tuple(model for model in norm_df.columns if model in set(selected_models))
    assert_exact_members(tuple(selected_models), actual, layer="single-combo dataframe")
    return list(selected_models)


def _ledger_metrics_from_leaderboard(leaderboard_df: pd.DataFrame) -> dict[str, Any]:
    ledger_metrics: dict[str, Any] = {}
    if "Ensemble" in leaderboard_df.index:
        ens_row = leaderboard_df.loc["Ensemble"]
        ledger_metrics = {
            key: round(float(value), 6) if pd.notna(value) else None
            for key, value in ens_row.items()
        }

        ann_ret = ledger_metrics.get("annualized_return")
        mdd = ledger_metrics.get("max_drawdown")
        if ann_ret is not None and mdd is not None and mdd < 0:
            ledger_metrics["calmar"] = round(ann_ret / abs(mdd), 4)
        else:
            ledger_metrics["calmar"] = None
    return ledger_metrics


def _sub_model_metrics_from_leaderboard(
    leaderboard_df: pd.DataFrame,
    combo_models: Sequence[str],
) -> dict[str, dict[str, Any]]:
    sub_metrics: dict[str, dict[str, Any]] = {}
    for model_name in combo_models:
        if model_name in leaderboard_df.index:
            row = leaderboard_df.loc[model_name]
            sub_metrics[model_name] = {
                key: round(float(value), 6) if pd.notna(value) else None
                for key, value in row.items()
            }
    return sub_metrics


def _eval_window_from_norm_df(combo_norm_df: pd.DataFrame, args: Any) -> dict[str, Any]:
    window_start = str(combo_norm_df.index.get_level_values("datetime").min().date())
    window_end = str(combo_norm_df.index.get_level_values("datetime").max().date())
    return {
        "start": window_start,
        "end": window_end,
        "only_last_years": _arg_value(args, "only_last_years", 0),
        "only_last_months": _arg_value(args, "only_last_months", 0),
    }


def _loo_summary(
    contributions: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        model_name: {"delta": round(values.get("delta", 0), 6)}
        for model_name, values in contributions.items()
    } if contributions else {}


def _cli_args(args: Any) -> list[str]:
    cli_args = _arg_value(args, "cli_args")
    if not isinstance(cli_args, (list, tuple)):
        cli_args = sys.argv[1:]
    return list(cli_args)


def _append_ledger_best_effort(
    *,
    hooks: SingleComboPipelineHooks,
    workspace_root: str | os.PathLike | None,
    request: SingleComboPipelineRequest,
    combo_models: Sequence[str],
    combo_norm_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    contributions: Mapping[str, Mapping[str, float]],
) -> None:
    try:
        hooks.append_to_fusion_ledger(
            workspace_root=workspace_root or hooks.get_workspace_root(),
            run_date=request.anchor_date,
            combo_name=request.combo_name,
            models=list(combo_models),
            method=request.method,
            is_default=request.is_default,
            eval_window=_eval_window_from_norm_df(combo_norm_df, request.args),
            metrics=_ledger_metrics_from_leaderboard(leaderboard_df),
            sub_model_metrics=_sub_model_metrics_from_leaderboard(
                leaderboard_df,
                combo_models,
            ),
            loo_contributions=_loo_summary(contributions),
            cli_args=_cli_args(request.args),
            source_recorders={
                model: request.train_records.get("models", {}).get(model, "")
                for model in combo_models
            },
            source_anchors={
                model: _arg_value(request.args, "source_anchors", {}).get(model, request.anchor_date)
                for model in combo_models
            },
            run_id=_arg_value(request.args, "run_id"),
            plan_fingerprint=_arg_value(request.args, "plan_fingerprint"),
        )
    except Exception as exc:
        print(f"[Ledger] 写入失败（非致命）: {exc}")


def run_single_combo_pipeline(
    request: SingleComboPipelineRequest,
    *,
    hooks: SingleComboPipelineHooks | None = None,
    verbose: bool = True,
) -> SingleComboPipelineResult | None:
    """Execute Stage 2-10 for one ensemble combo."""

    hooks = hooks or default_single_combo_pipeline_hooks()

    if verbose:
        print(f"\n{'@' * 60}")
        if request.combo_name:
            default_tag = " [DEFAULT]" if request.is_default else ""
            print(f"@ Combo: {request.combo_name}{default_tag}")
        print(f"@ 模型: {', '.join(request.selected_models)}")
        print(f"@ 权重: {request.method}")
        print(f"{'@' * 60}")

    top_k = request.model_config.get("TopK", 22)
    drop_n = request.model_config.get("DropN", 3)
    benchmark = request.model_config.get("benchmark", "SH000300")

    combo_models = _combo_models(request.selected_models, request.norm_df)
    if not combo_models:
        print(f"Warning: combo {request.combo_name} 没有有效模型，跳过")
        return None

    combo_norm_df = request.norm_df[combo_models].dropna(how="any")
    combo_metrics = {
        model_name: request.model_metrics.get(model_name, 0)
        for model_name in combo_models
    }

    combo_output_dir = request.args.output_dir
    hooks.correlation_analysis(
        combo_norm_df,
        combo_output_dir,
        request.anchor_date,
        combo_name=request.combo_name,
    )

    combo_ensemble_config = dict(request.ensemble_config)
    final_weights, static_weights, is_dynamic = hooks.calculate_weights(
        combo_norm_df,
        combo_metrics,
        request.method,
        request.model_config,
        combo_ensemble_config,
        request.manual_weights_str,
    )

    final_score = hooks.generate_ensemble_signal(
        combo_norm_df,
        final_weights,
        static_weights,
        is_dynamic,
    )

    prediction_dir = _arg_value(request.args, "prediction_dir")
    workspace_root = _valid_workspace_root(_arg_value(request.args, "workspace_root"))
    saved_prediction = hooks.save_predictions(
        final_score,
        request.anchor_date,
        request.experiment_name,
        request.method,
        combo_models,
        combo_metrics,
        static_weights,
        is_dynamic,
        combo_output_dir,
        combo_name=request.combo_name,
        is_default=request.is_default,
        prediction_dir=prediction_dir,
        save_csv=_arg_value(request.args, "save_csv", False),
        workspace_root=workspace_root,
        source_recorders={model: request.train_records.get("models", {}).get(model, "") for model in combo_models},
        source_anchors={
            model: _arg_value(request.args, "source_anchors", {}).get(model, request.anchor_date)
            for model in combo_models
        },
        run_id=_arg_value(request.args, "run_id"),
        plan_fingerprint=_arg_value(request.args, "plan_fingerprint"),
    )
    pred_file = getattr(saved_prediction, "returned_ref", saved_prediction)
    recorder_id = getattr(saved_prediction, "recorder_id", None)
    output_evidence = getattr(saved_prediction, "output_evidence", None)

    report_df = None
    executor_obj = None
    if not request.args.no_backtest:
        report_df, executor_obj = hooks.run_backtest(
            final_score,
            top_k,
            drop_n,
            benchmark,
            request.args.freq,
            verbose=request.args.verbose_backtest,
        )

        if request.args.detailed_analysis and executor_obj is not None:
            hooks.run_detailed_backtest_analysis(
                executor_obj,
                request.combo_name,
                request.anchor_date,
                combo_output_dir,
                request.args.freq,
                benchmark=benchmark,
            )

    all_reports = {}
    leaderboard_df = None
    if report_df is not None:
        all_reports, leaderboard_df = hooks.risk_analysis_and_leaderboard(
            report_df,
            combo_norm_df,
            request.train_records,
            combo_models,
            request.args.freq,
            combo_output_dir,
            request.anchor_date,
            combo_name=request.combo_name,
        )

    if not request.args.no_charts and all_reports:
        hooks.generate_charts(
            all_reports,
            report_df,
            final_weights,
            is_dynamic,
            request.args.freq,
            request.anchor_date,
            combo_output_dir,
            combo_name=request.combo_name,
        )

    print(f"\n{'=' * 60}")
    print("Stage 9: 模型贡献度分析 (LOO)")
    print(f"{'=' * 60}")
    contributions = hooks.calculate_loo_contribution(combo_norm_df, final_score)
    if contributions:
        contribution_result = hooks.save_model_contribution_snapshot(
            output_dir=combo_output_dir,
            anchor_date=request.anchor_date,
            combo_name=request.combo_name,
            contributions=contributions,
        )
        print(f"模型贡献度已保存: {contribution_result.path}")

    if report_df is not None and leaderboard_df is not None:
        _append_ledger_best_effort(
            hooks=hooks,
            workspace_root=workspace_root,
            request=request,
            combo_models=combo_models,
            combo_norm_df=combo_norm_df,
            leaderboard_df=leaderboard_df,
            contributions=contributions,
        )

    return SingleComboPipelineResult(
        name=request.combo_name or "default",
        models=tuple(combo_models),
        method=request.method,
        is_default=request.is_default,
        pred_file=pred_file,
        report_df=report_df,
        leaderboard_df=leaderboard_df,
        contributions=contributions,
        recorder_id=recorder_id,
        output_evidence=output_evidence,
    )


def run_single_combo(
    combo_name,
    selected_models,
    method,
    manual_weights_str,
    norm_df,
    model_metrics,
    loaded_models,
    train_records,
    model_config,
    ensemble_config,
    anchor_date,
    experiment_name,
    args,
    is_default=False,
    *,
    hooks: SingleComboPipelineHooks | None = None,
) -> dict[str, Any] | None:
    """Legacy-compatible wrapper around :func:`run_single_combo_pipeline`."""

    result = run_single_combo_pipeline(
        SingleComboPipelineRequest(
            combo_name=combo_name,
            selected_models=selected_models,
            method=method,
            manual_weights_str=manual_weights_str,
            norm_df=norm_df,
            model_metrics=model_metrics,
            loaded_models=loaded_models,
            train_records=train_records,
            model_config=model_config,
            ensemble_config=ensemble_config,
            anchor_date=anchor_date,
            experiment_name=experiment_name,
            args=args,
            is_default=is_default,
        ),
        hooks=hooks,
    )
    return None if result is None else result.to_legacy_dict()
