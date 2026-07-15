"""Filesystem-only preparation for the Rolling command boundary.

This module intentionally does not import ``quantpits.utils.env``, Qlib, or
MLflow.  Everything produced here is a snapshot of explicit workspace files;
calendar-dependent windows remain deferred until Phase 28C runtime resolution.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from quantpits.rolling.errors import (
    RollingActionConflictError,
    RollingConfigInvalidError,
    RollingConfigMissingError,
    RollingResumeStateMissingError,
    RollingStateCorruptError,
    RollingStateUnsupportedError,
    RollingTargetSelectionEmptyError,
    RollingTargetUnknownError,
    RollingWorkflowMissingError,
    RollingWorkflowOutsideWorkspaceError,
    RollingWorkspaceRequiredError,
)
from quantpits.runtime.command import (
    CommandPlan,
    CommandStep,
    InputRef,
    OutputRef,
    StateRef,
    fingerprint_command_plan,
)
from quantpits.runtime.render import command_plan_to_public_dict, render_command_plan
from quantpits.utils.workspace import (
    WorkspaceContext,
    fingerprint_file,
    fingerprint_value,
)


@dataclass(frozen=True)
class RollingRunOptions:
    action: str
    models: tuple = ()
    algorithm: str = None
    dataset: str = None
    market: str = None
    tag: str = None
    all_enabled: bool = False
    skip: tuple = ()
    training_method: str = None
    no_pretrain: bool = False
    cache_size_mb: int = None
    allow_stale_predict: bool = False
    backtest: bool = False
    show_folds: bool = False
    explain_plan: bool = False
    json_plan: bool = False
    run_id: str = None


@dataclass(frozen=True)
class RollingTarget:
    target_key: str
    model_name: str
    family: str
    workflow_path: str
    workflow_fingerprint: str
    selected_by: str
    legacy_info: dict


@dataclass(frozen=True)
class LegacyRollingStateInspection:
    status: str
    path: str
    fingerprint: str = None
    anchor: str = None
    training_method: str = None
    completed_windows: int = 0
    completed_units: int = 0
    warning: str = None


@dataclass(frozen=True)
class RollingAnchorPolicy:
    resolution: str
    configured_value: str = None
    runtime_source: str = "qlib_trading_calendar"


@dataclass(frozen=True)
class PreparedRollingRun:
    ctx: WorkspaceContext
    options: RollingRunOptions
    cli_args: tuple
    effective_config: dict
    targets: tuple
    state: LegacyRollingStateInspection
    anchor_policy: RollingAnchorPolicy
    plan: CommandPlan
    plan_fingerprint: str
    input_fingerprints: dict


def resolve_workspace_context(workspace=None):
    """Resolve a context without importing or activating the legacy env."""

    root = workspace if isinstance(workspace, (str, os.PathLike)) else None
    root = root or os.environ.get("QLIB_WORKSPACE_DIR")
    if not root:
        raise RollingWorkspaceRequiredError(
            "pass --workspace PATH or set QLIB_WORKSPACE_DIR"
        )
    return WorkspaceContext.from_root(root)


def _csv(value):
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def options_from_namespace(args):
    primary = (
        ("cold_start", bool(getattr(args, "cold_start", False)), "--cold-start"),
        ("merge", bool(getattr(args, "merge", False)), "--merge"),
        ("retrain_models", bool(_csv(getattr(args, "retrain_models", None))), "--retrain-models"),
        ("retrain_last", bool(getattr(args, "retrain_last", False)), "--retrain-last"),
        ("predict_only", bool(getattr(args, "predict_only", False)), "--predict-only"),
        ("resume", bool(getattr(args, "resume", False)), "--resume"),
        ("backtest_only", bool(getattr(args, "backtest_only", False)), "--backtest-only"),
        ("clear_state", bool(getattr(args, "clear_state", False)), "--clear-state"),
        ("show_state", bool(getattr(args, "show_state", False)), "--show-state"),
    )
    enabled = [(action, flag) for action, active, flag in primary if active]
    if len(enabled) > 1:
        raise RollingActionConflictError(
            "conflicting primary actions: %s" % ", ".join(flag for _, flag in enabled)
        )
    action = enabled[0][0] if enabled else "daily"
    models = (_csv(getattr(args, "retrain_models", None))
              if action == "retrain_models" else _csv(getattr(args, "models", None)))
    default_all = action in ("merge", "resume", "backtest_only", "retrain_last")
    return RollingRunOptions(
        action=action,
        models=models,
        algorithm=getattr(args, "algorithm", None),
        dataset=getattr(args, "dataset", None),
        market=getattr(args, "market", None),
        tag=getattr(args, "tag", None),
        all_enabled=bool(getattr(args, "all_enabled", False) or default_all),
        skip=_csv(getattr(args, "skip", None)),
        training_method=getattr(args, "training_method", None),
        no_pretrain=bool(getattr(args, "no_pretrain", False)),
        cache_size_mb=getattr(args, "cache_size", None),
        allow_stale_predict=bool(getattr(args, "allow_stale_predict", False)),
        backtest=bool(getattr(args, "backtest", False)),
        show_folds=bool(getattr(args, "show_folds", False)),
        explain_plan=bool(getattr(args, "explain_plan", False)
                          or getattr(args, "dry_run", False)),
        json_plan=bool(getattr(args, "json_plan", False)),
        run_id=getattr(args, "run_id", None),
    )


def _load_yaml(path, missing_error, invalid_label):
    if not path.is_file():
        raise missing_error("%s not found" % invalid_label)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RollingConfigInvalidError("cannot read %s: %s" % (invalid_label, exc))
    if not isinstance(payload, dict):
        raise RollingConfigInvalidError("%s must contain a mapping" % invalid_label)
    return payload


def _normalize_rolling_config(raw, override):
    try:
        step = str(raw.get("test_step", "3M")).strip().upper()
        if step.endswith("M"):
            step_months = int(step[:-1])
        elif step.endswith("Y"):
            step_months = int(step[:-1]) * 12
        else:
            raise ValueError("test_step must use nM or nY")
        if step_months <= 0:
            raise ValueError("test_step must be positive")
        method = str(raw.get("training_method", "slide")).lower()
        if override is not None:
            method = override
        if method not in ("slide", "cpcv"):
            raise ValueError("training_method must be slide or cpcv")
        config = {
            "rolling_start": str(raw["rolling_start"]),
            "train_years": int(raw["train_years"]),
            "valid_years": int(raw.get("valid_years", 1)),
            "test_step": step,
            "test_step_months": step_months,
            "training_method": method,
        }
        if config["train_years"] <= 0 or config["valid_years"] < 0:
            raise ValueError("year lengths must be non-negative and train_years positive")
        if method == "cpcv":
            cpcv = {
                "n_groups": int(raw.get("cpcv_n_groups", 10)),
                "n_val_groups": int(raw.get("cpcv_n_val_groups", 1)),
                "purge_steps": int(raw.get("cpcv_purge_steps", 3)),
                "embargo_steps": int(raw.get("cpcv_embargo_steps", 5)),
            }
            if (cpcv["n_groups"] < 2 or cpcv["n_val_groups"] < 1
                    or cpcv["n_val_groups"] >= cpcv["n_groups"]
                    or cpcv["purge_steps"] < 0 or cpcv["embargo_steps"] < 0):
                raise ValueError("invalid CPCV group/purge/embargo values")
            estimated_group = (config["train_years"] * 52.0) / cpcv["n_groups"]
            if cpcv["purge_steps"] + cpcv["embargo_steps"] >= estimated_group * 0.8:
                raise ValueError("CPCV purge+embargo destroys the estimated group size")
            config["cpcv"] = cpcv
        return config
    except (KeyError, TypeError, ValueError) as exc:
        raise RollingConfigInvalidError("invalid config/rolling_config.yaml: %s" % exc)


def _contained_workflow(ctx, value):
    if not isinstance(value, str) or not value.strip():
        raise RollingWorkflowMissingError("model registry entry has no yaml_file")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = ctx.root / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(ctx.root)
    except ValueError:
        raise RollingWorkflowOutsideWorkspaceError(
            "workflow is outside workspace: %s" % value
        )
    if not candidate.is_file():
        raise RollingWorkflowMissingError("workflow not found: %s" % relative.as_posix())
    return candidate, relative.as_posix()


def _select_targets(ctx, registry, options, family):
    models = registry.get("models", registry)
    if not isinstance(models, dict):
        raise RollingConfigInvalidError("config/model_registry.yaml models must be a mapping")
    selected_by = ""
    if options.models:
        unknown = [name for name in options.models if name not in models]
        if unknown:
            raise RollingTargetUnknownError("unknown model(s): %s" % ", ".join(unknown))
        requested = set(options.models)
        names = [name for name in models if name in requested]
        selected_by = "exact_models"
    elif options.all_enabled:
        names = [name for name, info in models.items()
                 if isinstance(info, dict) and info.get("enabled", False)]
        selected_by = "all_enabled"
    elif any((options.algorithm, options.dataset, options.market, options.tag)):
        names = []
        selected_by = "filters"
        for name, info in models.items():
            if not isinstance(info, dict):
                continue
            if options.algorithm and str(info.get("algorithm", "")).lower() != options.algorithm.lower():
                continue
            if options.dataset and str(info.get("dataset", "")).lower() != options.dataset.lower():
                continue
            if options.market and str(info.get("market", "")).lower() != options.market.lower():
                continue
            tags = [str(item).lower() for item in info.get("tags", [])]
            if options.tag and options.tag.lower() not in tags:
                continue
            names.append(name)
    else:
        raise RollingTargetSelectionEmptyError(
            "specify --models, --algorithm, --dataset, --tag, or --all-enabled"
        )
    skipped = set(options.skip)
    names = [name for name in names if name not in skipped]
    if not names:
        raise RollingTargetSelectionEmptyError("no matching models")
    targets = []
    suffix = family
    for name in names:
        info = models[name]
        if not isinstance(info, dict):
            raise RollingConfigInvalidError("registry model %s must be a mapping" % name)
        workflow, relative = _contained_workflow(ctx, info.get("yaml_file"))
        targets.append(RollingTarget(
            target_key="%s@%s" % (name, suffix), model_name=name, family=family,
            workflow_path=relative, workflow_fingerprint=fingerprint_file(workflow),
            selected_by=selected_by, legacy_info=dict(info),
        ))
    return tuple(targets)


def inspect_legacy_state(path, workspace_root):
    relative = path.relative_to(workspace_root).as_posix()
    try:
        if not path.exists() or path.stat().st_size == 0:
            return LegacyRollingStateInspection(status="missing", path=relative)
    except OSError as exc:
        return LegacyRollingStateInspection(
            status="corrupt", path=relative,
            warning="state cannot be inspected: %s" % exc.__class__.__name__,
        )
    try:
        fingerprint = fingerprint_file(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return LegacyRollingStateInspection(
            status="corrupt", path=relative, fingerprint=locals().get("fingerprint"),
            warning="state cannot be decoded: %s" % exc.__class__.__name__,
        )
    if not isinstance(payload, dict):
        return LegacyRollingStateInspection(
            status="unsupported", path=relative, fingerprint=fingerprint,
            warning="state root must be a mapping",
        )
    completed = payload.get("completed_windows", {})
    if not isinstance(completed, dict) or any(
            not isinstance(value, dict) for value in completed.values()):
        return LegacyRollingStateInspection(
            status="unsupported", path=relative, fingerprint=fingerprint,
            warning="completed_windows must map window keys to model mappings",
        )
    anchor = payload.get("anchor_date")
    method = payload.get("training_method", "slide")
    if anchor is not None and not isinstance(anchor, str):
        return LegacyRollingStateInspection(
            status="unsupported", path=relative, fingerprint=fingerprint,
            warning="anchor_date must be a string or null",
        )
    if method not in ("slide", "cpcv"):
        return LegacyRollingStateInspection(
            status="unsupported", path=relative, fingerprint=fingerprint,
            warning="training_method is unsupported",
        )
    return LegacyRollingStateInspection(
        status="valid_legacy", path=relative, fingerprint=fingerprint,
        anchor=anchor, training_method=method, completed_windows=len(completed),
        completed_units=sum(len(value) for value in completed.values()),
        warning="legacy state has partial target/run identity",
    )


def _input_ref(ctx, relative, kind="config", required=False, description=""):
    path = ctx.path(relative)
    fingerprint = fingerprint_file(path) if path.is_file() else None
    return InputRef(relative, kind=kind, fingerprint=fingerprint,
                    required=required, description=description), fingerprint


def _validate_mapping_file(path, label, loader):
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = loader(handle)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise RollingConfigInvalidError("cannot read %s: %s" % (label, exc))
    if not isinstance(payload, dict):
        raise RollingConfigInvalidError("%s must contain a mapping" % label)


def _outputs_for_action(action, state_path, backtest):
    if action in ("show_state",):
        return (), (StateRef(state_path, "read", "strict readonly legacy state inspection"),)
    if action == "clear_state":
        return (
            OutputRef("data/history/rolling_state_<timestamp>.json", kind="state",
                      description="legacy state backup when state exists"),
            OutputRef("data/operator_log.jsonl", kind="data",
                      description="legacy best-effort operation log"),
        ), (
            StateRef(state_path, "read_write", "backup and delete legacy state"),
            StateRef("data/locks/training_execution.lock", "read_write",
                     "shared training execution lease"),
        )
    if action == "backtest_only":
        return (
            OutputRef("mlruns/<existing-recorder>/", kind="other",
                      description="backtest metrics and artifacts", overwrite=True),
            OutputRef("data/operator_log.jsonl", kind="data",
                      description="legacy best-effort operation log"),
        ), (
            StateRef(state_path, "read", "prepared legacy state baseline"),
            StateRef("data/locks/training_execution.lock", "read_write",
                     "shared training execution lease"),
        )
    outputs = [
        OutputRef("latest_train_records.json", kind="record",
                  description="legacy current recorder merge", overwrite=True),
        OutputRef("mlruns/<rolling-experiments>/", kind="other",
                  description="per-window and combined legacy recorders"),
        OutputRef("data/operator_log.jsonl", kind="data",
                  description="legacy best-effort operation log"),
    ]
    if action == "predict_only":
        outputs.append(OutputRef(
            "data/rolling_prediction_history.jsonl", kind="data",
            description="rolling prediction history",
        ))
    elif action != "backtest_only":
        outputs.append(OutputRef(
            "data/rolling_training_history.jsonl", kind="data",
            description="per-window training history",
        ))
        outputs.append(OutputRef(
            "data/promote_history.jsonl", kind="data",
            description="activate matching pending promotion records",
            overwrite=True,
        ))
    if backtest:
        outputs.append(OutputRef("mlruns/<existing-recorder>/", kind="other",
                                 description="backtest metrics and artifacts", overwrite=True))
    state_action = "read" if action == "predict_only" else "read_write"
    state_description = ("legacy rolling state input" if state_action == "read"
                         else "legacy unversioned rolling progress")
    return tuple(outputs), (
        StateRef(state_path, state_action, state_description),
        StateRef("data/locks/training_execution.lock", "read_write",
                 "shared training execution lease"),
    )


def prepare_rolling_run(ctx, options, cli_args=()):
    rolling_path = ctx.config_path("rolling_config.yaml")
    raw_config = _load_yaml(
        rolling_path, RollingConfigMissingError, "config/rolling_config.yaml"
    )
    effective = _normalize_rolling_config(raw_config, options.training_method)
    training_method = effective["training_method"]
    family = "cpcv_rolling" if training_method == "cpcv" else "rolling"
    state_path = ctx.data_path(
        "rolling_state_cpcv.json" if training_method == "cpcv" else "rolling_state.json"
    )
    state = inspect_legacy_state(state_path, ctx.root)
    if state.status == "corrupt" and options.action != "show_state":
        raise RollingStateCorruptError("%s: %s" % (state.path, state.warning))
    if state.status == "unsupported" and options.action != "show_state":
        raise RollingStateUnsupportedError("%s: %s" % (state.path, state.warning))
    if options.action == "resume" and (state.status != "valid_legacy" or not state.anchor):
        raise RollingResumeStateMissingError("resume requires valid legacy state with anchor")

    targets = ()
    registry = None
    if options.action not in ("show_state", "clear_state"):
        registry_path = ctx.config_path("model_registry.yaml")
        registry = _load_yaml(
            registry_path, RollingConfigMissingError, "config/model_registry.yaml"
        )
        targets = _select_targets(ctx, registry, options, family)

    inputs = []
    fingerprints = {}
    declared = (
        ("config/rolling_config.yaml", "config", True, "rolling source configuration"),
        ("config/model_registry.yaml", "config", options.action not in ("show_state", "clear_state"), "ordered model registry"),
        ("config/model_config.json", "config", False, "workspace market configuration"),
        ("config/strategy_config.yaml", "config", False, "backtest strategy configuration"),
        ("config/prod_config.json", "config", False, "account and production configuration"),
        ("latest_train_records.json", "record", False, "current recorder baseline"),
        ("data/promote_history.jsonl", "data", False, "promotion status baseline"),
    )
    _validate_mapping_file(
        ctx.config_path("model_config.json"), "config/model_config.json", json.load,
    )
    _validate_mapping_file(
        ctx.config_path("strategy_config.yaml"), "config/strategy_config.yaml",
        yaml.safe_load,
    )
    _validate_mapping_file(
        ctx.config_path("prod_config.json"), "config/prod_config.json", json.load,
    )
    _validate_mapping_file(
        ctx.path("latest_train_records.json"), "latest_train_records.json", json.load,
    )
    for relative, kind, required, description in declared:
        ref, fingerprint = _input_ref(ctx, relative, kind, required, description)
        inputs.append(ref)
        if fingerprint:
            fingerprints[relative] = fingerprint
    for target in targets:
        inputs.append(InputRef(
            target.workflow_path, kind="config", fingerprint=target.workflow_fingerprint,
            required=True, description="workflow for %s" % target.target_key,
        ))
        fingerprints[target.workflow_path] = target.workflow_fingerprint
    inputs.append(InputRef(
        state.path, kind="state", fingerprint=state.fingerprint, required=False,
        description="legacy rolling state baseline (%s)" % state.status,
    ))
    if state.fingerprint:
        fingerprints[state.path] = state.fingerprint

    source_fp = fingerprint_file(rolling_path)
    normalized_fp = fingerprint_value(_normalize_rolling_config(raw_config, None))
    effective_fp = fingerprint_value(effective)
    fingerprints["rolling_source"] = source_fp
    fingerprints["rolling_normalized"] = normalized_fp
    fingerprints["rolling_effective"] = effective_fp
    outputs, states = _outputs_for_action(options.action, state.path, options.backtest)
    warnings = []
    if options.training_method and options.training_method != str(raw_config.get("training_method", "slide")).lower():
        warnings.append("--training-method overrides config in memory; source file is unchanged")
    if training_method == "cpcv":
        cpcv = effective["cpcv"]
        estimated_group = (effective["train_years"] * 52.0) / cpcv["n_groups"]
        gap = cpcv["purge_steps"] + cpcv["embargo_steps"]
        if gap >= estimated_group * 0.5:
            warnings.append("CPCV purge+embargo is at least 50% of estimated group size")
    if state.warning:
        warnings.append(state.warning)
    if options.action == "clear_state" and state.status == "missing":
        warnings.append("legacy state is already missing")
    steps = (
        CommandStep("confirm-safeguard", "confirm the explicit workspace before mutation"),
        CommandStep("acquire-shared-lease", "acquire the workspace training execution lease"),
        CommandStep("recheck-input-baselines", "fail if prepared filesystem inputs changed"),
        CommandStep("activate-workspace", "activate legacy workspace compatibility"),
        CommandStep("resolve-calendar-windows", "resolve exact anchor/windows/folds after Qlib init", expensive=True),
        CommandStep("execute-legacy-action", "execute the frozen Rolling action scope", expensive=True),
        CommandStep("write-operator-log", "append legacy best-effort operation metadata"),
    )
    if options.action == "show_state":
        steps = (CommandStep("inspect-state", "classify legacy state without a lock"),)
    elif options.action == "clear_state":
        steps = (
            CommandStep("confirm-safeguard", "confirm the explicit workspace before mutation"),
            CommandStep("acquire-shared-lease", "acquire the workspace training execution lease"),
            CommandStep("recheck-state-baseline", "fail if the prepared state input changed"),
            CommandStep("backup-and-clear-state", "back up and remove the selected family state"),
            CommandStep("write-operator-log", "append legacy best-effort operation metadata"),
        )
    run_id = options.run_id.strip() if isinstance(options.run_id, str) and options.run_id.strip() else "rolling-plan"
    semantic_args = _semantic_cli_args(cli_args)
    plan = CommandPlan(
        command="rolling_train", workspace=ctx.root.name, run_id=run_id,
        mode="%s:%s" % (family, options.action), args=semantic_args,
        inputs=tuple(inputs), outputs=outputs, states=states, steps=steps,
        config_fingerprints=dict(sorted(fingerprints.items())), warnings=tuple(warnings),
        metadata={
            "family": family,
            "training_method": training_method,
            "action": options.action,
            "target_keys": [target.target_key for target in targets],
            "targets": [{
                "target_key": target.target_key,
                "model_name": target.model_name,
                "workflow_path": target.workflow_path,
                "workflow_fingerprint": target.workflow_fingerprint,
                "selected_by": target.selected_by,
            } for target in targets],
            "effective_config": effective,
            "effective_config_fingerprint": effective_fp,
            "state_inspection": {
                "status": state.status, "path": state.path,
                "fingerprint": state.fingerprint, "anchor": state.anchor,
                "training_method": state.training_method,
                "completed_windows": state.completed_windows,
                "completed_units": state.completed_units,
            },
            "anchor_resolution": (
                "legacy_state_hint" if options.action == "resume" and state.anchor
                else "deferred_to_qlib_calendar"
            ),
            "anchor_hint": state.anchor,
            "window_resolution": "deferred_to_runtime",
            "window_count": None,
            "cpcv_fold_resolution": "deferred_to_qlib_calendar" if training_method == "cpcv" else "not_applicable",
            "runtime_validations": [
                "input baseline recheck", "workspace/backend activation",
                "exact trading anchor", "exact ordered windows/folds",
            ],
            "zero_write_plan_route": True,
            "resume_identity_strength": "legacy_partial" if state.status == "valid_legacy" else "none",
            "publication_protocol": "legacy_record_merge",
            "manifest_protocol": "not_yet_service_backed",
            "state_protocol": "legacy_unversioned",
        },
    )
    return PreparedRollingRun(
        ctx=ctx, options=options, cli_args=tuple(cli_args), effective_config=effective,
        targets=targets, state=state,
        anchor_policy=RollingAnchorPolicy(
            resolution=("legacy_state_hint"
                        if options.action == "resume" and state.anchor
                        else "deferred_to_qlib_calendar"),
            configured_value=state.anchor,
        ),
        plan=plan, plan_fingerprint=fingerprint_command_plan(plan),
        input_fingerprints=fingerprints,
    )


def prepared_plan_json(prepared):
    return {
        "schema_version": 1,
        "plan_fingerprint": prepared.plan_fingerprint,
        "plan": command_plan_to_public_dict(prepared.plan),
    }


def _semantic_cli_args(cli_args):
    """Exclude renderer/workspace/attempt flags from semantic plan identity."""

    result = []
    skip_value = False
    for item in cli_args:
        if skip_value:
            skip_value = False
            continue
        if item in ("--dry-run", "--explain-plan", "--json-plan"):
            continue
        if item in ("--workspace", "--run-id"):
            skip_value = True
            continue
        if item.startswith("--workspace=") or item.startswith("--run-id="):
            continue
        result.append(item)
    return tuple(result)


def render_prepared_plan(prepared):
    metadata = prepared.plan.metadata
    lines = [
        render_command_plan(prepared.plan),
        "",
        "Prepared Rolling facts:",
        "  - Action: %s" % metadata["action"],
        "  - Family: %s" % metadata["family"],
        "  - Training method: %s" % metadata["training_method"],
        "  - Legacy state: %s (%s)" % (
            metadata["state_inspection"]["status"],
            metadata["state_inspection"]["path"],
        ),
        "  - Anchor resolution: %s" % metadata["anchor_resolution"],
        "  - Window resolution: %s" % metadata["window_resolution"],
        "  - CPCV fold resolution: %s" % metadata["cpcv_fold_resolution"],
        "  - Publication protocol: %s" % metadata["publication_protocol"],
        "  - State protocol: %s" % metadata["state_protocol"],
        "  - Writes: none (plan route)",
        "",
        "Plan fingerprint: %s" % prepared.plan_fingerprint,
    ]
    return "\n".join(lines)


def resolve_legacy_targets(ctx, args):
    """Compatibility facade returning the legacy registry mapping."""

    options = options_from_namespace(args)
    raw = _load_yaml(
        ctx.config_path("rolling_config.yaml"), RollingConfigMissingError,
        "config/rolling_config.yaml",
    )
    method = _normalize_rolling_config(raw, options.training_method)["training_method"]
    family = "cpcv_rolling" if method == "cpcv" else "rolling"
    registry = _load_yaml(
        ctx.config_path("model_registry.yaml"), RollingConfigMissingError,
        "config/model_registry.yaml",
    )
    return {target.model_name: target.legacy_info
            for target in _select_targets(ctx, registry, options, family)}
