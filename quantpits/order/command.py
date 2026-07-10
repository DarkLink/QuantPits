"""Workspace-safe command planning boundary for order generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

from quantpits.config_contracts.core import WorkspaceValidationResult
from quantpits.config_contracts.runtime_bridge import input_refs_from_validation
from quantpits.config_contracts.workspace import validate_workspace
from quantpits.runtime import (
    CommandPlan,
    CommandStep,
    InputRef,
    OutputRef,
    StateRef,
    fingerprint_command_plan,
    generate_run_id,
    render_command_plan,
)
from quantpits.runtime.render import command_plan_to_public_dict
from quantpits.utils.config_loader import load_workspace_config
from quantpits.utils.workspace import WorkspaceContext, fingerprint_file


class OrderPlanError(ValueError):
    """Raised when an order command plan cannot be prepared safely."""


@dataclass(frozen=True)
class OrderRunOptions:
    model: str | None = None
    output_dir: str = "output"
    record_file: str | None = None
    dry_run: bool = False
    verbose: bool = False
    combo: str | None = None
    explain_plan: bool = False
    json_plan: bool = False
    run_id: str | None = None


@dataclass(frozen=True)
class OrderRunConfig:
    merged_config: dict
    cashflow_config: dict
    strategy_config: dict
    ensemble_config: dict
    ensemble_records: dict
    train_records: dict


@dataclass(frozen=True)
class ResolvedOrderSource:
    mode: Literal["model", "ensemble"]
    requested_name: str | None
    resolved_name: str | None
    record_id: str | None
    experiment_name: str | None
    source_label: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedOrderRun:
    ctx: WorkspaceContext
    options: OrderRunOptions
    execution_options: OrderRunOptions
    cli_args: tuple[str, ...]
    validation_result: WorkspaceValidationResult | None
    config: OrderRunConfig
    source: ResolvedOrderSource
    plan: CommandPlan
    plan_fingerprint: str


@dataclass(frozen=True)
class OrderRunSummary:
    anchor_date: str
    trade_date: str
    source_label: str
    source_description: str
    holding_count: int
    sell_count: int
    buy_count: int
    sell_file: str | None
    buy_file: str | None
    dry_run: bool


@dataclass(frozen=True)
class OrderCommandDependencies:
    get_workspace_context: Callable[[], WorkspaceContext]
    load_run_config: Callable[[WorkspaceContext, OrderRunOptions], OrderRunConfig]
    safeguard: Callable[[str], None]
    execute: Callable[[PreparedOrderRun], OrderRunSummary]


@dataclass(frozen=True)
class OrderCommandRequest:
    args: argparse.Namespace
    cli_args: tuple[str, ...]


@dataclass(frozen=True)
class OrderCommandOutcome:
    mode: Literal["json-plan", "explain-plan", "execute"]
    prepared: PreparedOrderRun
    rendered_output: str | None = None
    summary: OrderRunSummary | None = None


def build_order_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Order Generation - 基于融合/单模型预测生成买卖订单",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用最新融合预测
  python quantpits/scripts/order_gen.py

  # 使用单模型预测（不融合）
  python quantpits/scripts/order_gen.py --model gru

  # 仅解释将读取和执行的内容（不初始化 Qlib）
  python quantpits/scripts/order_gen.py --explain-plan
""",
    )
    parser.add_argument("--model", type=str, help="使用单模型预测（从 Qlib 记录加载）")
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录 (默认 output)")
    parser.add_argument(
        "--record-file",
        type=str,
        default=None,
        help="训练记录文件，用于加载单模型 PKL 预测 (默认 latest_train_records.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="执行完整计算并打印订单计划，但不写入文件")
    parser.add_argument("--verbose", action="store_true", help="显示详细的排名和价格信息")
    parser.add_argument("--combo", type=str, default=None, help="指定要使用的融合组合名称")
    parser.add_argument("--explain-plan", action="store_true", help="仅打印执行计划，不初始化 Qlib，不写文件")
    parser.add_argument("--json-plan", action="store_true", help="以 JSON 输出执行计划；隐含 explain-plan")
    parser.add_argument("--run-id", type=str, default=None, help="显式指定运行 ID，用于计划身份")
    return parser


def options_from_namespace(args: argparse.Namespace) -> OrderRunOptions:
    defaults = OrderRunOptions()
    values = {item.name: getattr(args, item.name, getattr(defaults, item.name)) for item in fields(OrderRunOptions)}
    return OrderRunOptions(**values)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _workspace_path(ctx: WorkspaceContext, value: str | None, default: str) -> Path:
    candidate = Path(value or default).expanduser()
    return candidate.resolve() if candidate.is_absolute() else ctx.path(candidate.as_posix()).resolve()


def _display_path(ctx: WorkspaceContext, path: Path) -> str:
    try:
        return path.resolve().relative_to(ctx.root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_record_path(ctx: WorkspaceContext, record_file: str | None) -> Path:
    path = _workspace_path(ctx, record_file, "latest_train_records.json")
    if not path.exists() and record_file is None:
        fallback = ctx.config_path("latest_train_records.json")
        if fallback.exists():
            return fallback
    return path


def load_order_run_config(
    ctx: WorkspaceContext,
    options: OrderRunOptions,
    *,
    merged_config: dict | None = None,
    cashflow_config: dict | None = None,
    strategy_config: dict | None = None,
) -> OrderRunConfig:
    """Load config-only order inputs without initializing Qlib or writing files."""

    if merged_config is None:
        merged_config = load_workspace_config(ctx.root)
    if strategy_config is None:
        from quantpits.utils import strategy

        strategy_config = strategy.load_strategy_config(ctx.root)
    if cashflow_config is None:
        cashflow_config = _read_json(ctx.config_path("cashflow.json"))

    record_path = _resolve_record_path(ctx, options.record_file)

    return OrderRunConfig(
        merged_config=merged_config,
        cashflow_config=cashflow_config,
        strategy_config=strategy_config,
        ensemble_config=_read_json(ctx.config_path("ensemble_config.json")),
        ensemble_records=_read_json(ctx.config_path("ensemble_records.json")),
        train_records=_read_json(record_path),
    )


def _record_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get("record_id") or value.get("id")
        return str(candidate) if candidate else None
    return None


def resolve_order_source(options: OrderRunOptions, config: OrderRunConfig) -> ResolvedOrderSource:
    warnings: list[str] = []
    if options.model:
        if options.combo:
            warnings.append("--model takes precedence; --combo is ignored")
        models = config.train_records.get("models", {})
        resolved_name = options.model if options.model in models else None
        if resolved_name is None:
            from quantpits.utils.train_utils import resolve_model_key

            resolved_name = resolve_model_key(options.model, models)
        record_id = _record_id(models.get(resolved_name)) if resolved_name else None
        if not record_id:
            warnings.append(f"model '{options.model}' was not found in train records")
        experiment_name = config.train_records.get("experiment_name")
        if resolved_name:
            experiments = config.train_records.get("experiments", {})
            experiment_name = experiments.get(resolved_name, experiment_name)
        return ResolvedOrderSource(
            mode="model",
            requested_name=options.model,
            resolved_name=resolved_name,
            record_id=record_id,
            experiment_name=experiment_name,
            source_label=options.model,
            warnings=tuple(warnings),
        )

    records = config.ensemble_records
    combos = records.get("combos", {}) if isinstance(records.get("combos", {}), dict) else {}
    requested = options.combo
    resolved_name = requested or records.get("default_combo")
    if requested and requested not in combos:
        warnings.append(f"combo '{requested}' was not found in ensemble records")
        resolved_name = None
    if not resolved_name or resolved_name not in combos:
        if combos:
            resolved_name = next(reversed(combos))
            if not requested:
                warnings.append(f"no recorded default combo; using '{resolved_name}'")
        else:
            resolved_name = None
    record_id = _record_id(combos.get(resolved_name)) if resolved_name else None
    if not record_id:
        warnings.append("no ensemble prediction record is available")
    return ResolvedOrderSource(
        mode="ensemble",
        requested_name=requested,
        resolved_name=resolved_name,
        record_id=record_id,
        experiment_name="Ensemble_Fusion" if record_id else None,
        source_label="ensemble",
        warnings=tuple(warnings),
    )


def _validation_warnings(result: WorkspaceValidationResult | None) -> tuple[str, ...]:
    if result is None:
        return ()
    return tuple(
        f"{message.severity}: {message.path}: {message.message}"
        for message in result.messages
        if message.severity != "info"
    )


def _dedupe_inputs(refs: Sequence[InputRef]) -> tuple[InputRef, ...]:
    result: dict[str, InputRef] = {}
    for ref in refs:
        result.setdefault(ref.path, ref)
    return tuple(result.values())


def _file_ref(ctx: WorkspaceContext, path: Path, *, kind: str, description: str, required: bool = False) -> InputRef:
    return InputRef(
        path=_display_path(ctx, path),
        kind=kind,  # type: ignore[arg-type]
        fingerprint=fingerprint_file(path) if path.exists() else None,
        required=required and path.exists(),
        description=description,
    )


def _preview_outputs(options: OrderRunOptions) -> tuple[str, ...]:
    base = options.output_dir.rstrip("/") or "."
    source = options.model or "ensemble"
    return (
        f"{base}/model_opinions_<next_trade_date>.csv",
        f"{base}/model_opinions_<next_trade_date>.json",
        f"{base}/sell_suggestion_{source}_<next_trade_date>.csv",
        f"{base}/buy_suggestion_{source}_<next_trade_date>.csv",
    )


def _steps(dry_run: bool) -> tuple[CommandStep, ...]:
    names = (
        ("validate-configs", "validate workspace configuration contracts", False),
        ("resolve-prediction-source", "resolve the model or ensemble recorder", False),
        ("init-qlib", "initialize Qlib for the active workspace", True),
        ("resolve-trade-dates", "resolve anchor and next trade dates", False),
        ("load-predictions", "load selected predictions from a Qlib recorder", True),
        ("load-price-data", "load prices and price-limit estimates", True),
        ("analyze-positions", "rank predictions against current holdings", False),
        ("build-model-opinions", "compare available prediction sources", True),
        ("generate-sell-orders", "generate sell suggestions", False),
        ("generate-buy-orders", "generate buy suggestions", False),
    )
    steps = [CommandStep(name, desc, expensive=expensive) for name, desc, expensive in names]
    reason = "--dry-run" if dry_run else ""
    steps.extend(
        [
            CommandStep("write-model-opinions", "write model opinion reports", can_skip=dry_run, skip_reason=reason),
            CommandStep("write-order-files", "write buy and sell suggestion files", can_skip=dry_run, skip_reason=reason),
        ]
    )
    return tuple(steps)


def build_order_command_plan(
    *,
    ctx: WorkspaceContext,
    options: OrderRunOptions,
    cli_args: Sequence[str],
    config: OrderRunConfig,
    source: ResolvedOrderSource,
    validation_result: WorkspaceValidationResult | None,
) -> CommandPlan:
    run_id = options.run_id or generate_run_id("order_gen")
    refs = list(input_refs_from_validation(validation_result)) if validation_result else []
    record_path = _resolve_record_path(ctx, options.record_file)
    refs.append(_file_ref(ctx, record_path, kind="record", description="train records"))
    refs.extend(
        [
            _file_ref(ctx, ctx.config_path("cashflow.json"), kind="config", description="cashflow config"),
            _file_ref(ctx, ctx.config_path("ensemble_records.json"), kind="record", description="ensemble recorder mapping"),
        ]
    )

    preview = _preview_outputs(options)
    outputs = () if options.dry_run else tuple(OutputRef(path, kind="report", overwrite=True) for path in preview)
    params = config.strategy_config.get("strategy", {}).get("params", {})
    merged = config.merged_config
    cashflows = config.cashflow_config.get("cashflows", {})
    fingerprints = {
        artifact.name: artifact.fingerprint
        for artifact in (validation_result.artifacts if validation_result else ())
        if artifact.fingerprint
    }
    warnings = source.warnings + _validation_warnings(validation_result)
    metadata = {
        "source_mode": source.mode,
        "requested_model": options.model,
        "requested_combo": options.combo,
        "resolved_source_name": source.resolved_name,
        "source_record_available": bool(source.record_id),
        "record_file": _display_path(ctx, record_path),
        "output_dir": options.output_dir,
        "dry_run": options.dry_run,
        "verbose": options.verbose,
        "market": merged.get("market", "csi300"),
        "strategy_name": config.strategy_config.get("strategy", {}).get("name", "topk_dropout"),
        "topk": params.get("topk", 20),
        "n_drop": params.get("n_drop", 3),
        "buy_suggestion_factor": params.get("buy_suggestion_factor", 2),
        "current_state_date": merged.get("current_date"),
        "holding_count": len(merged.get("current_holding", [])),
        "pending_cashflow_date_count": len(cashflows) if isinstance(cashflows, dict) else 0,
        "preview_output_paths": list(preview),
    }
    return CommandPlan(
        command="order_gen",
        workspace=ctx.root.as_posix(),
        run_id=run_id,
        mode="model" if source.mode == "model" else ("ensemble-combo" if options.combo else "ensemble-default"),
        args=tuple(cli_args),
        inputs=_dedupe_inputs(refs),
        outputs=outputs,
        states=(
            StateRef("mlflow", action="read", description="selected prediction recorder"),
            StateRef("qlib-calendar", action="read", description="trading calendar"),
            StateRef("qlib-market-data", action="read", description="price and price-limit data"),
        ),
        steps=_steps(options.dry_run),
        config_fingerprints=fingerprints,
        warnings=warnings,
        metadata=metadata,
    )


def prepare_order_run(
    *,
    ctx: WorkspaceContext,
    options: OrderRunOptions,
    cli_args: Sequence[str],
    run_config: OrderRunConfig,
) -> PreparedOrderRun:
    validation_result = validate_workspace(ctx, include_optional=True, strict=False)
    source = resolve_order_source(options, run_config)
    output_path = _workspace_path(ctx, options.output_dir, "output")
    record_path = _workspace_path(ctx, options.record_file, "latest_train_records.json") if options.record_file else None
    execution_options = replace(
        options,
        output_dir=output_path.as_posix(),
        record_file=record_path.as_posix() if record_path else None,
    )
    plan = build_order_command_plan(
        ctx=ctx,
        options=options,
        cli_args=cli_args,
        config=run_config,
        source=source,
        validation_result=validation_result,
    )
    fingerprint = fingerprint_command_plan(plan)
    plan = replace(plan, metadata={**plan.metadata, "plan_fingerprint": fingerprint})
    return PreparedOrderRun(
        ctx=ctx,
        options=options,
        execution_options=execution_options,
        cli_args=tuple(cli_args),
        validation_result=validation_result,
        config=run_config,
        source=source,
        plan=plan,
        plan_fingerprint=fingerprint,
    )


def render_prepared_order_plan(prepared: PreparedOrderRun) -> str:
    return render_command_plan(prepared.plan)


def prepared_order_plan_json(prepared: PreparedOrderRun) -> dict[str, Any]:
    return {"schema_version": 1, **command_plan_to_public_dict(prepared.plan)}


def run_order_command(request: OrderCommandRequest, dependencies: OrderCommandDependencies) -> OrderCommandOutcome:
    options = options_from_namespace(request.args)
    if options.json_plan and not options.explain_plan:
        options = replace(options, explain_plan=True)
    ctx = dependencies.get_workspace_context()
    run_config = dependencies.load_run_config(ctx, options)
    prepared = prepare_order_run(ctx=ctx, options=options, cli_args=request.cli_args, run_config=run_config)

    if options.json_plan:
        return OrderCommandOutcome(
            mode="json-plan",
            prepared=prepared,
            rendered_output=json.dumps(prepared_order_plan_json(prepared), indent=2, sort_keys=True, ensure_ascii=False),
        )
    if options.explain_plan:
        return OrderCommandOutcome(mode="explain-plan", prepared=prepared, rendered_output=render_prepared_order_plan(prepared))

    dependencies.safeguard("Order Generation")
    return OrderCommandOutcome(mode="execute", prepared=prepared, summary=dependencies.execute(prepared))
