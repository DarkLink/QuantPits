"""Planning helpers for ensemble_fusion runtime dry-runs and manifests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from quantpits.config_contracts.core import WorkspaceValidationResult
from quantpits.config_contracts.runtime_bridge import input_refs_from_validation
from quantpits.runtime import CommandPlan, CommandStep, InputRef, OutputRef, StateRef, generate_run_id
from quantpits.utils.ensemble_utils import get_default_combo, parse_ensemble_config
from quantpits.utils.train_utils import resolve_model_key
from quantpits.utils.workspace import WorkspaceContext, fingerprint_file


class EnsemblePlanError(ValueError):
    """Raised when an ensemble fusion plan cannot be constructed."""


@dataclass(frozen=True)
class ResolvedCombo:
    name: str | None
    models: tuple[str, ...]
    method: str
    manual_weights: str | None = None
    is_default: bool = False
    source: str = ""
    warnings: tuple[str, ...] = ()


def _split_models(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _resolve_models(
    names: Iterable[str],
    *,
    train_records: dict,
    default_mode: str | None,
    warning_prefix: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    models_dict = train_records.get("models", {})
    resolved: list[str] = []
    warnings: list[str] = []
    for name in names:
        full_key = resolve_model_key(name, models_dict, default_mode)
        if full_key:
            resolved.append(full_key)
        else:
            warnings.append(f"{warning_prefix} model '{name}' was not found in train records")
    return tuple(resolved), tuple(warnings)


def _method_for_default_combo(args: argparse.Namespace, combo_cfg: dict) -> str:
    method = combo_cfg.get("method", "equal")
    if getattr(args, "method", "equal") != "equal":
        method = args.method
    return method


def resolve_ensemble_combos(
    *,
    args: argparse.Namespace,
    train_records: dict,
    ensemble_config: dict,
) -> tuple[ResolvedCombo, ...]:
    """Resolve CLI/config ensemble combo selection without touching Qlib."""

    cli_training_mode = getattr(args, "training_mode", None)

    if getattr(args, "models", None):
        resolved, warnings = _resolve_models(
            _split_models(args.models),
            train_records=train_records,
            default_mode=cli_training_mode,
            warning_prefix="CLI",
        )
        if not resolved:
            raise EnsemblePlanError("No valid models were resolved from --models")
        return (
            ResolvedCombo(
                name=None,
                models=resolved,
                method=getattr(args, "method", "equal"),
                manual_weights=getattr(args, "weights", None),
                is_default=True,
                source="models",
                warnings=warnings,
            ),
        )

    combos, _global_config = parse_ensemble_config(ensemble_config)

    if getattr(args, "combo", None):
        combo_name = args.combo
        if combo_name not in combos:
            available = ", ".join(combos.keys()) or "<none>"
            raise EnsemblePlanError(
                f"combo '{combo_name}' was not found in ensemble_config.json "
                f"(available: {available})"
            )
        cfg = combos[combo_name]
        combo_mode = cfg.get("training_mode", cli_training_mode)
        resolved, warnings = _resolve_models(
            cfg.get("models", []),
            train_records=train_records,
            default_mode=combo_mode,
            warning_prefix=f"combo '{combo_name}'",
        )
        if not resolved:
            raise EnsemblePlanError(f"No valid models were resolved from combo '{combo_name}'")
        return (
            ResolvedCombo(
                name=combo_name,
                models=resolved,
                method=cfg.get("method", "equal"),
                manual_weights=None,
                is_default=bool(cfg.get("default", False)),
                source="combo",
                warnings=warnings,
            ),
        )

    if getattr(args, "from_config_all", False):
        if not combos:
            raise EnsemblePlanError("ensemble_config.json does not define any combos")
        resolved_combos = []
        for name, cfg in combos.items():
            combo_mode = cfg.get("training_mode", cli_training_mode)
            resolved, warnings = _resolve_models(
                cfg.get("models", []),
                train_records=train_records,
                default_mode=combo_mode,
                warning_prefix=f"combo '{name}'",
            )
            resolved_combos.append(
                ResolvedCombo(
                    name=name,
                    models=resolved,
                    method=cfg.get("method", "equal"),
                    manual_weights=None,
                    is_default=bool(cfg.get("default", False)),
                    source="from-config-all",
                    warnings=warnings,
                )
            )
        if not any(combo.models for combo in resolved_combos):
            raise EnsemblePlanError("No valid models were resolved from ensemble_config.json")
        return tuple(resolved_combos)

    if getattr(args, "from_config", False):
        default_name, default_cfg = get_default_combo(combos)
        if not default_cfg:
            raise EnsemblePlanError("ensemble_config.json does not define a default combo")
        combo_mode = default_cfg.get("training_mode", cli_training_mode)
        resolved, warnings = _resolve_models(
            default_cfg.get("models", []),
            train_records=train_records,
            default_mode=combo_mode,
            warning_prefix="default combo",
        )
        if not resolved:
            raise EnsemblePlanError("No valid models were resolved from the default combo")
        return (
            ResolvedCombo(
                name=default_name,
                models=resolved,
                method=_method_for_default_combo(args, default_cfg),
                manual_weights=getattr(args, "weights", None),
                is_default=True,
                source="from-config",
                warnings=warnings,
            ),
        )

    raise EnsemblePlanError("Specify --models, --from-config, --from-config-all, or --combo")


def _display_path(ctx: WorkspaceContext, path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ctx.path(candidate.as_posix())
    try:
        return candidate.resolve().relative_to(ctx.root).as_posix()
    except ValueError:
        return candidate.as_posix()


def _record_input_ref(ctx: WorkspaceContext, record_file: str) -> InputRef:
    display = _display_path(ctx, record_file)
    path = ctx.path(record_file) if not Path(record_file).is_absolute() else Path(record_file)
    fingerprint = fingerprint_file(path) if path.exists() else None
    return InputRef(
        path=display,
        kind="record",
        fingerprint=fingerprint,
        required=path.exists(),
        description="train records",
    )


def _dedupe_inputs(refs: Iterable[InputRef]) -> tuple[InputRef, ...]:
    deduped: dict[str, InputRef] = {}
    for ref in refs:
        if ref.path not in deduped or (deduped[ref.path].kind == "config" and ref.kind != "config"):
            deduped[ref.path] = ref
    return tuple(deduped.values())


def _mode_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "models", None):
        return "models"
    if getattr(args, "combo", None):
        return "combo"
    if getattr(args, "from_config_all", False):
        return "from-config-all"
    if getattr(args, "from_config", False):
        return "from-config"
    return ""


def _plan_steps(args: argparse.Namespace) -> tuple[CommandStep, ...]:
    steps = [
        CommandStep("validate-configs", "validate workspace configuration contracts"),
        CommandStep("resolve-combos", "resolve ensemble combos and model record keys"),
        CommandStep("load-predictions", "load selected model predictions from Qlib recorders", expensive=True),
        CommandStep("filter-window", "filter prediction dates by CLI window options"),
        CommandStep("correlation-analysis", "write model correlation matrix"),
        CommandStep("calculate-weights", "calculate ensemble weights"),
        CommandStep("fuse-signals", "compute ensemble scores"),
        CommandStep("save-predictions", "save ensemble predictions to recorder and optional CSV", expensive=True),
    ]
    if getattr(args, "no_backtest", False):
        steps.append(CommandStep("backtest", "run Qlib backtest", expensive=True, can_skip=True, skip_reason="--no-backtest"))
        steps.append(CommandStep("risk-analysis", "generate risk reports and leaderboard", can_skip=True, skip_reason="--no-backtest"))
    else:
        steps.append(CommandStep("backtest", "run Qlib backtest", expensive=True))
        steps.append(CommandStep("risk-analysis", "generate risk reports and leaderboard"))
    if getattr(args, "no_charts", False) or getattr(args, "no_backtest", False):
        reason = "--no-charts" if getattr(args, "no_charts", False) else "--no-backtest"
        steps.append(CommandStep("charts", "generate ensemble charts", can_skip=True, skip_reason=reason))
    else:
        steps.append(CommandStep("charts", "generate ensemble charts"))
    if getattr(args, "detailed_analysis", False):
        steps.append(CommandStep("detailed-analysis", "generate detailed backtest analysis", expensive=True))
    steps.append(CommandStep("loo-contribution", "calculate leave-one-out model contribution"))
    if getattr(args, "no_backtest", False):
        steps.append(CommandStep("fusion-ledger", "append fusion run ledger", can_skip=True, skip_reason="--no-backtest"))
    else:
        steps.append(CommandStep("fusion-ledger", "append fusion run ledger"))
    if getattr(args, "no_manifest", False):
        steps.append(CommandStep("write-manifest", "write run manifest", can_skip=True, skip_reason="--no-manifest"))
    else:
        steps.append(CommandStep("write-manifest", "write run manifest"))
    return tuple(steps)


def _plan_outputs(args: argparse.Namespace, run_id: str) -> tuple[OutputRef, ...]:
    output_dir = getattr(args, "output_dir", "output/ensemble")
    refs = [
        OutputRef(f"{output_dir}/correlation_matrix_<combo>_<anchor_date>.csv", kind="report", overwrite=True),
        OutputRef(f"{output_dir}/ensemble_fusion_config_<combo>_<anchor_date>.json", kind="report", overwrite=True),
        OutputRef(f"{output_dir}/leaderboard_<combo>_<anchor_date>.csv", kind="report", overwrite=True),
        OutputRef(f"{output_dir}/model_contribution_<combo>_<anchor_date>.json", kind="report", overwrite=True),
    ]
    if getattr(args, "save_csv", False):
        prediction_dir = getattr(args, "prediction_dir", None) or "output/predictions"
        refs.append(
            OutputRef(
                f"{prediction_dir}/ensemble_<combo>_<anchor_date>.csv",
                kind="prediction",
                overwrite=True,
            )
        )
    if not getattr(args, "no_manifest", False):
        refs.append(
            OutputRef(
                f"output/manifests/ensemble_fusion/{run_id}.json",
                kind="manifest",
                overwrite=True,
            )
        )
    return tuple(refs)


def _plan_states(args: argparse.Namespace) -> tuple[StateRef, ...]:
    states = [
        StateRef("config/ensemble_records.json", action="read_write", description="ensemble recorder mapping"),
        StateRef("mlflow", action="read_write", description="prediction recorder outputs"),
    ]
    if not getattr(args, "no_backtest", False):
        states.append(
            StateRef("data/fusion_run_ledger.jsonl", action="write", description="fusion run ledger")
        )
    return tuple(states)


def _validation_warnings(result: WorkspaceValidationResult | None) -> tuple[str, ...]:
    if result is None:
        return ()
    warnings = []
    for message in result.messages:
        if message.severity != "info":
            warnings.append(f"{message.severity}: {message.path}: {message.message}")
    return tuple(warnings)


def _combo_metadata(combos: Sequence[ResolvedCombo]) -> list[dict[str, Any]]:
    return [
        {
            "name": combo.name,
            "models": list(combo.models),
            "method": combo.method,
            "manual_weights": combo.manual_weights,
            "is_default": combo.is_default,
            "source": combo.source,
        }
        for combo in combos
    ]


def build_ensemble_command_plan(
    *,
    ctx: WorkspaceContext,
    args: argparse.Namespace,
    train_records: dict,
    model_config: dict,
    ensemble_config: dict,
    combos: tuple[ResolvedCombo, ...],
    validation_result: WorkspaceValidationResult | None = None,
    run_id: str | None = None,
    cli_args: Sequence[str] = (),
) -> CommandPlan:
    """Build a read-only command plan for ensemble_fusion."""

    plan_run_id = run_id or getattr(args, "run_id", None) or generate_run_id("ensemble_fusion")
    inputs = list(input_refs_from_validation(validation_result)) if validation_result else []
    record_file = getattr(args, "record_file", "latest_train_records.json")
    inputs.append(_record_input_ref(ctx, record_file))

    config_fingerprints = {}
    if validation_result is not None:
        config_fingerprints = {
            artifact.name: artifact.fingerprint
            for artifact in validation_result.artifacts
            if artifact.fingerprint
        }

    combo_warnings = [warning for combo in combos for warning in combo.warnings]
    warnings = tuple(combo_warnings) + _validation_warnings(validation_result)

    metadata = {
        "resolved_combos": _combo_metadata(combos),
        "record_file": record_file,
        "freq": getattr(args, "freq", None) or model_config.get("freq", "week"),
        "no_backtest": bool(getattr(args, "no_backtest", False)),
        "no_charts": bool(getattr(args, "no_charts", False)),
        "detailed_analysis": bool(getattr(args, "detailed_analysis", False)),
        "save_csv": bool(getattr(args, "save_csv", False)),
        "date_filter": {
            "start_date": getattr(args, "start_date", None),
            "end_date": getattr(args, "end_date", None),
            "only_last_years": getattr(args, "only_last_years", 0),
            "only_last_months": getattr(args, "only_last_months", 0),
        },
        "combo_count": len(combos),
        "model_count": len({model for combo in combos for model in combo.models}),
        "ensemble_config_present": bool(ensemble_config),
        "anchor_date": train_records.get("anchor_date"),
        "experiment_name": train_records.get("experiment_name"),
    }

    return CommandPlan(
        command="ensemble_fusion",
        workspace=ctx.root.as_posix(),
        run_id=plan_run_id,
        mode=_mode_from_args(args),
        args=tuple(cli_args),
        inputs=_dedupe_inputs(inputs),
        outputs=_plan_outputs(args, plan_run_id),
        states=_plan_states(args),
        steps=_plan_steps(args),
        config_fingerprints=config_fingerprints,
        warnings=warnings,
        metadata=metadata,
    )
