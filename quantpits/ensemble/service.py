"""Service layer for ensemble fusion planning, execution, and manifests."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from quantpits.config_contracts.workspace import validate_workspace
from quantpits.ensemble.config import load_ensemble_run_config
from quantpits.ensemble.types import (
    EnsembleExecutionHooks,
    EnsembleRunConfig,
    EnsembleRunOptions,
    EnsembleRunSummary,
    PreparedEnsembleRun,
    options_to_namespace,
)
from quantpits.runtime import (
    CommandResult,
    OutputRef,
    command_plan_to_public_dict,
    fingerprint_command_plan,
    manifest_from_result,
    manifest_path,
    render_command_plan,
    write_run_manifest,
)
from quantpits.utils.ensemble_plan import (
    build_ensemble_command_plan,
    resolve_ensemble_combos,
)
from quantpits.utils.workspace import WorkspaceContext


def workspace_relative(ctx: WorkspaceContext, path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ctx.path(candidate.as_posix())
    try:
        return candidate.resolve().relative_to(ctx.root).as_posix()
    except ValueError:
        return candidate.as_posix()


def resolve_workspace_path(ctx: WorkspaceContext, path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.as_posix()
    return ctx.path(candidate.as_posix()).as_posix()


def execution_options_for_workspace(
    ctx: WorkspaceContext,
    options: EnsembleRunOptions,
) -> EnsembleRunOptions:
    return replace(
        options,
        output_dir=resolve_workspace_path(ctx, options.output_dir) or options.output_dir,
        prediction_dir=resolve_workspace_path(ctx, options.prediction_dir),
    )


def _actual_output_refs(
    ctx: WorkspaceContext,
    combo_results: list[dict],
    *,
    anchor_date: str,
    options: EnsembleRunOptions,
    manifest_file: Path | None = None,
) -> tuple[OutputRef, ...]:
    outputs = []
    for result in combo_results:
        pred_file = result.get("pred_file")
        if pred_file:
            kind = "prediction" if str(pred_file).endswith(".csv") else "record"
            outputs.append(
                OutputRef(
                    str(pred_file),
                    kind=kind,
                    description=f"combo {result.get('name')} prediction",
                )
            )
    if len(combo_results) > 1:
        outputs.append(
            OutputRef(
                workspace_relative(ctx, Path(options.output_dir) / f"combo_comparison_{anchor_date}.csv"),
                kind="report",
                description="combo comparison",
                overwrite=True,
            )
        )
    if manifest_file is not None:
        outputs.append(
            OutputRef(
                workspace_relative(ctx, manifest_file),
                kind="manifest",
                description="run manifest",
                overwrite=True,
            )
        )
    return tuple(outputs)


def _write_manifest_safely(ctx: WorkspaceContext, result: CommandResult) -> Path:
    manifest = manifest_from_result(result)
    return write_run_manifest(ctx, manifest)


def prepare_ensemble_run(
    *,
    ctx: WorkspaceContext,
    options: EnsembleRunOptions,
    cli_args: tuple[str, ...],
    run_config: EnsembleRunConfig | None = None,
    validate: bool = True,
) -> PreparedEnsembleRun:
    validation_result = validate_workspace(ctx, include_optional=True, strict=False) if validate else None
    config = run_config or load_ensemble_run_config(ctx, record_file=options.record_file)

    resolved_options = options
    if not resolved_options.freq:
        resolved_options = replace(resolved_options, freq=config.model_config.get("freq", "week"))

    args = options_to_namespace(resolved_options)
    combos = resolve_ensemble_combos(
        args=args,
        train_records=config.train_records,
        ensemble_config=config.ensemble_config,
    )
    plan = build_ensemble_command_plan(
        ctx=ctx,
        args=args,
        train_records=config.train_records,
        model_config=config.model_config,
        ensemble_config=config.ensemble_config,
        combos=combos,
        validation_result=validation_result,
        run_id=resolved_options.run_id,
        cli_args=cli_args,
    )
    plan_fingerprint = fingerprint_command_plan(plan)
    return PreparedEnsembleRun(
        ctx=ctx,
        options=resolved_options,
        cli_args=cli_args,
        validation_result=validation_result,
        config=config,
        combos=combos,
        plan=plan,
        plan_fingerprint=plan_fingerprint,
    )


def prepared_plan_json(prepared: PreparedEnsembleRun) -> dict:
    return {
        "schema_version": 1,
        "plan_fingerprint": prepared.plan_fingerprint,
        "plan": command_plan_to_public_dict(prepared.plan),
    }


def render_prepared_plan(prepared: PreparedEnsembleRun) -> str:
    return f"{render_command_plan(prepared.plan)}\n\nPlan fingerprint: {prepared.plan_fingerprint}"


def _failed_result(
    prepared: PreparedEnsembleRun,
    *,
    started_at: str,
    finished_at: str,
    anchor_date: str,
    experiment_name: str,
    combo_results: list[dict],
    exc: BaseException,
) -> CommandResult:
    return CommandResult(
        plan=prepared.plan,
        status="failed",
        started_at=started_at,
        finished_at=finished_at,
        records={
            "anchor_date": anchor_date,
            "experiment_name": experiment_name,
            "n_combos": len(combo_results),
        },
        error={"type": type(exc).__name__, "message": str(exc)},
    )


def write_failed_manifest(
    *,
    prepared: PreparedEnsembleRun,
    started_at: str,
    anchor_date: str,
    experiment_name: str,
    combo_results: list[dict],
    exc: BaseException,
    oplog,
) -> str | None:
    if prepared.options.no_manifest:
        return None
    finished_at = datetime.now().isoformat()
    result = _failed_result(
        prepared,
        started_at=started_at,
        finished_at=finished_at,
        anchor_date=anchor_date,
        experiment_name=experiment_name,
        combo_results=combo_results,
        exc=exc,
    )
    manifest_file = _write_manifest_safely(prepared.ctx, result)
    manifest_rel = workspace_relative(prepared.ctx, manifest_file)
    oplog.set_run_manifest(run_id=prepared.plan.run_id, manifest_path=manifest_rel)
    return manifest_rel


class EnsembleFusionService:
    def __init__(self, hooks: EnsembleExecutionHooks):
        self.hooks = hooks

    def _write_failed_manifest_best_effort(
        self,
        *,
        prepared: PreparedEnsembleRun,
        started_at: str,
        anchor_date: str,
        experiment_name: str,
        combo_results: list[dict],
        exc: BaseException,
        oplog,
    ) -> None:
        try:
            write_failed_manifest(
                prepared=prepared,
                started_at=started_at,
                anchor_date=anchor_date,
                experiment_name=experiment_name,
                combo_results=combo_results,
                exc=exc,
                oplog=oplog,
            )
        except Exception as manifest_exc:
            print(f"Warning: failed to write run manifest: {manifest_exc}")

    def execute(self, prepared: PreparedEnsembleRun) -> EnsembleRunSummary:
        options = prepared.options
        execution_options = execution_options_for_workspace(prepared.ctx, options)
        args = options_to_namespace(execution_options)
        args.workspace_root = prepared.ctx.root.as_posix()
        args.cli_args = list(prepared.cli_args)
        train_records = prepared.config.train_records
        model_config = prepared.config.model_config
        ensemble_config = prepared.config.ensemble_config

        started_at = datetime.now().isoformat()
        combo_results: list[dict] = []
        manifest_file: Path | None = None

        from quantpits.utils.operator_log import OperatorLog

        with OperatorLog(
            "ensemble_fusion",
            args=list(prepared.cli_args),
            log_file=prepared.ctx.data_path("operator_log.jsonl").as_posix(),
            run_id=prepared.plan.run_id,
            plan_fingerprint=prepared.plan_fingerprint,
        ) as oplog:
            print(f"\n{'#'*60}")
            print("# Ensemble Fusion")
            print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'#'*60}")
            print(f"当前交易频率: {options.freq}")

            anchor_date = train_records.get("anchor_date", datetime.now().strftime("%Y-%m-%d"))
            experiment_name = train_records["experiment_name"]

            self.hooks.init_qlib()

            for combo in prepared.combos:
                for warning in combo.warnings:
                    print(f"Warning: {warning}")

            if options.from_config_all:
                print(f"多组合模式: 共 {len(prepared.combos)} 个 combo")
                for combo in prepared.combos:
                    tag = " [DEFAULT]" if combo.is_default else ""
                    print(f"  {combo.name}{tag}: {list(combo.models)} ({combo.method})")
            elif options.from_config:
                combo = prepared.combos[0]
                print(f"从 ensemble_config.json 加载 default combo: {combo.name}")
                print(f"模型: {list(combo.models)}")
                print(f"权重: {combo.method}")

            all_needed_models = set()
            for combo in prepared.combos:
                all_needed_models.update(combo.models)

            if not all_needed_models:
                print("Error: 没有有效的模型")
                sys.exit(1)

            print(f"\n所有 combo 涉及的模型并集 ({len(all_needed_models)}): {sorted(all_needed_models)}")

            try:
                norm_df, model_metrics, loaded_models = self.hooks.load_selected_predictions(
                    train_records,
                    sorted(all_needed_models),
                    norm_method=options.norm_method,
                )
                norm_df = self.hooks.filter_norm_df_by_args(norm_df, args)
                if norm_df.empty:
                    print("Error: 过滤后没有预测数据，请检查日期参数。")
                    sys.exit(1)
            except Exception as exc:
                if not options.no_manifest:
                    self._write_failed_manifest_best_effort(
                        prepared=prepared,
                        started_at=started_at,
                        anchor_date=anchor_date,
                        experiment_name=experiment_name,
                        combo_results=combo_results,
                        exc=exc,
                        oplog=oplog,
                    )
                raise

            try:
                for combo in prepared.combos:
                    combo_name = combo.name
                    models = list(combo.models)
                    valid_models = [model for model in models if model in loaded_models]
                    if not valid_models:
                        print(f"\nWarning: combo {combo_name} 没有有效模型，跳过")
                        continue

                    result = self.hooks.run_single_combo(
                        combo_name=combo_name,
                        selected_models=valid_models,
                        method=combo.method,
                        manual_weights_str=combo.manual_weights,
                        norm_df=norm_df,
                        model_metrics=model_metrics,
                        loaded_models=loaded_models,
                        train_records=train_records,
                        model_config=model_config,
                        ensemble_config=ensemble_config,
                        anchor_date=anchor_date,
                        experiment_name=experiment_name,
                        args=args,
                        is_default=combo.is_default,
                    )
                    if result:
                        combo_results.append(result)

                if len(combo_results) > 1:
                    self.hooks.compare_combos(
                        combo_results,
                        anchor_date,
                        execution_options.output_dir,
                        execution_options.freq,
                    )
            except Exception as exc:
                if not options.no_manifest:
                    self._write_failed_manifest_best_effort(
                        prepared=prepared,
                        started_at=started_at,
                        anchor_date=anchor_date,
                        experiment_name=experiment_name,
                        combo_results=combo_results,
                        exc=exc,
                        oplog=oplog,
                    )
                raise

            print(f"\n{'#'*60}")
            print("# 完成!")
            print(f"{'#'*60}")
            for result in combo_results:
                default_tag = " [DEFAULT]" if result["is_default"] else ""
                print(f"组合 {result['name']}{default_tag}: {', '.join(result['models'])}")
                print(f"  权重模式 : {result['method']}")
                print(f"  预测文件 : {result['pred_file']}")
                if result.get("report_df") is not None:
                    from quantpits.utils import strategy

                    bt_config = strategy.get_backtest_config()
                    initial_cash = bt_config["account"]
                    final_nav = result["report_df"].iloc[-1]["account"]
                    total_return = (final_nav - initial_cash) / initial_cash
                    print(f"  策略收益 : {total_return*100:.2f}%")
            print(f"输出目录   : {options.output_dir}")

            manifest_rel = None
            if not options.no_manifest:
                finished_at = datetime.now().isoformat()
                planned_manifest_file = manifest_path(prepared.ctx, prepared.plan.command, prepared.plan.run_id)
                manifest_outputs = _actual_output_refs(
                    prepared.ctx,
                    combo_results,
                    anchor_date=anchor_date,
                    options=options,
                    manifest_file=planned_manifest_file,
                )
                result = CommandResult(
                    plan=prepared.plan,
                    status="success",
                    started_at=started_at,
                    finished_at=finished_at,
                    outputs=manifest_outputs,
                    records={
                        "anchor_date": anchor_date,
                        "n_combos": len(combo_results),
                        "experiment_name": experiment_name,
                        "combos": [
                            {
                                "name": item.get("name"),
                                "models": item.get("models", []),
                                "method": item.get("method"),
                                "is_default": item.get("is_default", False),
                                "pred_file": item.get("pred_file"),
                            }
                            for item in combo_results
                        ],
                    },
                )
                try:
                    manifest_file = _write_manifest_safely(prepared.ctx, result)
                    manifest_rel = workspace_relative(prepared.ctx, manifest_file)
                    oplog.set_run_manifest(run_id=prepared.plan.run_id, manifest_path=manifest_rel)
                    print(f"Run manifest: {manifest_rel}")
                except Exception as manifest_exc:
                    print(f"Warning: failed to write run manifest: {manifest_exc}")

            oplog.set_result(
                {
                    "anchor_date": anchor_date,
                    "n_combos": len(combo_results),
                    "experiment_name": experiment_name,
                    "manifest_path": workspace_relative(prepared.ctx, manifest_file) if manifest_file else None,
                }
            )

            return EnsembleRunSummary(
                run_id=prepared.plan.run_id,
                anchor_date=anchor_date,
                experiment_name=experiment_name,
                combo_results=tuple(combo_results),
                manifest_path=manifest_rel,
            )
