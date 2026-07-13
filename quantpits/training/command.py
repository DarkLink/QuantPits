"""Plan-first command boundary shared by static and CPCV training."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import yaml

from quantpits.runtime import (
    CommandPlan,
    CommandStep,
    InputRef,
    OutputRef,
    StateRef,
    command_plan_to_public_dict,
    fingerprint_command_plan,
    render_command_plan,
)
from quantpits.training.errors import TrainingPlanError
from quantpits.training.records import TrainingRecordSnapshot, snapshot_from_dict
from quantpits.utils.workspace import WorkspaceContext, fingerprint_file

FAMILIES = ("static", "cpcv")
ACTIONS = ("full", "incremental", "predict_only")


@dataclass(frozen=True)
class TrainingRunOptions:
    family: str
    action: str
    models: Tuple[str, ...] = ()
    algorithm: Optional[str] = None
    dataset: Optional[str] = None
    market: Optional[str] = None
    tag: Optional[str] = None
    all_enabled: bool = False
    skip: Tuple[str, ...] = ()
    resume: bool = False
    source_records: str = "latest_train_records.json"
    experiment_name: Optional[str] = None
    no_pretrain: bool = False
    cache_size_mb: Optional[int] = None
    explain_plan: bool = False
    json_plan: bool = False
    no_manifest: bool = False
    run_id: Optional[str] = None

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise TrainingPlanError("unsupported training family")
        if self.action not in ACTIONS:
            raise TrainingPlanError("unsupported training action")
        if self.cache_size_mb is not None and self.cache_size_mb < 0:
            raise TrainingPlanError("cache size must be non-negative")
        if self.resume and self.action != "incremental":
            raise TrainingPlanError("resume is supported only for incremental training")
        if self.action != "full" and not (
            self.models or self.algorithm or self.dataset or self.market or self.tag or self.all_enabled
        ):
            raise TrainingPlanError("select models with --models, --all-enabled, or a filter")


@dataclass(frozen=True)
class TrainingTarget:
    key: str
    model_name: str
    family: str
    workflow_path: str
    workflow_fingerprint: str
    selected_by: Tuple[str, ...]


@dataclass(frozen=True)
class PreparedTrainingRun:
    ctx: WorkspaceContext
    options: TrainingRunOptions
    cli_args: Tuple[str, ...]
    targets: Tuple[TrainingTarget, ...]
    plan: CommandPlan
    plan_fingerprint: str
    input_fingerprints: Mapping[str, str]
    source_snapshot: Optional[TrainingRecordSnapshot]
    anchor_resolution: str


def _contained(ctx: WorkspaceContext, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ctx.root / candidate
    candidate = candidate.expanduser().resolve()
    try:
        candidate.relative_to(ctx.root.resolve())
    except ValueError as exc:
        raise TrainingPlanError("training input path must stay inside workspace") from exc
    return candidate


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainingPlanError("required training JSON is missing or invalid: %s" % path.name) from exc
    if not isinstance(value, dict):
        raise TrainingPlanError("training JSON root must be an object: %s" % path.name)
    return value


def _registry(ctx: WorkspaceContext) -> Mapping[str, Mapping[str, Any]]:
    path = ctx.config_path("model_registry.yaml")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingPlanError("model registry is missing or invalid") from exc
    models = value.get("models", value)
    if not isinstance(models, dict):
        raise TrainingPlanError("model registry must contain a models mapping")
    return models


def _selected_models(options: TrainingRunOptions, registry) -> Tuple[Tuple[str, Mapping[str, Any], Tuple[str, ...]], ...]:
    requested = set(options.models)
    skipped = set(options.skip)
    selected = []
    for name, raw in registry.items():
        info = raw if isinstance(raw, dict) else {}
        reasons = []
        if options.action == "full" or options.all_enabled:
            if not info.get("enabled", False):
                continue
            reasons.append("enabled")
        elif requested:
            if name not in requested:
                continue
            reasons.append("models")
        else:
            checks = (
                ("algorithm", options.algorithm),
                ("dataset", options.dataset),
                ("market", options.market),
            )
            if any(
                value and str(info.get(field, "")).lower() != str(value).lower()
                for field, value in checks
            ):
                continue
            if options.tag and options.tag.lower() not in tuple(str(item).lower() for item in info.get("tags", ())):
                continue
            reasons.extend(field for field, value in checks if value)
            if options.tag:
                reasons.append("tag")
        if name in skipped:
            continue
        selected.append((name, info, tuple(reasons or ("selection",))))
    missing = requested.difference(name for name, _, _ in selected).difference(skipped)
    if missing:
        raise TrainingPlanError("unknown or unavailable model selection: %s" % ", ".join(sorted(missing)))
    if not selected:
        raise TrainingPlanError("training plan selected no models")
    return tuple(selected)


def _workflow_path(ctx: WorkspaceContext, value: str) -> Path:
    candidate = Path(value)
    candidates = [ctx.root / candidate, ctx.config_dir / candidate]
    for path in candidates:
        resolved = path.resolve()
        try:
            resolved.relative_to(ctx.root.resolve())
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    raise TrainingPlanError("workflow file is missing inside workspace: %s" % candidate.name)


def _run_id(options: TrainingRunOptions) -> str:
    return options.run_id or "training-%s" % uuid.uuid4().hex[:12]


def prepare_training_run(
    *, ctx: WorkspaceContext, options: TrainingRunOptions, cli_args: Tuple[str, ...] = ()
) -> PreparedTrainingRun:
    if options.json_plan and not options.explain_plan:
        options = replace(options, explain_plan=True)
    cli_args = tuple(str(value).replace(str(ctx.root), "<workspace>") for value in cli_args)
    registry_path = ctx.config_path("model_registry.yaml")
    model_config_path = ctx.config_path("model_config.json")
    model_config = _read_json(model_config_path)
    registry = _registry(ctx)
    selected = _selected_models(options, registry)
    suffix = "cpcv" if options.family == "cpcv" else "static"
    targets = []
    fingerprints = {
        "config/model_registry.yaml": fingerprint_file(registry_path),
        "config/model_config.json": fingerprint_file(model_config_path),
    }
    input_refs = [
        InputRef("config/model_registry.yaml", kind="config", fingerprint=fingerprints["config/model_registry.yaml"]),
        InputRef("config/model_config.json", kind="config", fingerprint=fingerprints["config/model_config.json"]),
    ]
    for name, info, reasons in selected:
        workflow = _workflow_path(ctx, str(info.get("yaml_file", "")))
        relative = workflow.relative_to(ctx.root).as_posix()
        digest = fingerprint_file(workflow)
        fingerprints[relative] = digest
        input_refs.append(InputRef(relative, kind="config", fingerprint=digest))
        targets.append(TrainingTarget(
            key="%s@%s" % (name, suffix), model_name=name, family=options.family,
            workflow_path=relative, workflow_fingerprint=digest, selected_by=reasons,
        ))

    source_snapshot = None
    warnings = []
    if options.action == "predict_only":
        source_path = _contained(ctx, options.source_records)
        source_data = _read_json(source_path)
        try:
            source_snapshot = snapshot_from_dict(source_data)
        except (TypeError, ValueError, KeyError) as exc:
            raise TrainingPlanError("source training record violates its declared schema") from exc
        source_rel = source_path.relative_to(ctx.root).as_posix()
        source_fp = fingerprint_file(source_path)
        fingerprints[source_rel] = source_fp
        input_refs.append(InputRef(source_rel, kind="record", fingerprint=source_fp))
        if "schema_version" not in source_data:
            warnings.append("source training record uses legacy V1 identity")
        available = source_snapshot.entry_map
        missing = [target.key for target in targets if target.key not in available]
        if missing:
            raise TrainingPlanError("source training record is missing selected models: %s" % ", ".join(missing))

    date_mode = str(model_config.get("train_date_mode", "last_trade_date"))
    anchor_resolution = "configured" if date_mode == "current_date" and model_config.get("anchor_date") else "deferred_to_qlib_calendar"
    run_id = _run_id(options)
    command = "%s_train" % ("cv" if options.family == "cpcv" else "static")
    manifest_rel = "output/manifests/%s/%s.json" % (command, run_id)
    outputs = [OutputRef("latest_train_records.json", kind="record", description="verified current recorder registry", overwrite=True)]
    if not options.no_manifest:
        outputs.append(OutputRef(manifest_rel, kind="manifest", overwrite=True))
    steps = (
        CommandStep("resolve-calendar-and-dates", "resolve exact train/test windows after Qlib initialization", expensive=True),
        CommandStep("initialize-handler-cache", "prepare reusable handler cache", can_skip=options.cache_size_mb == 0, skip_reason="disabled" if options.cache_size_mb == 0 else ""),
        CommandStep("resolve-source-recorders", "verify per-model source recorder identity", can_skip=options.action != "predict_only", skip_reason="not predict-only" if options.action != "predict_only" else ""),
        CommandStep("train-or-predict-targets", "execute %d deterministic target(s)" % len(targets), expensive=True),
        CommandStep("verify-output-recorders", "verify persisted prediction coverage and workspace lineage"),
        CommandStep("publish-training-records", "publish one atomic V2 registry update"),
        CommandStep("write-manifest", "write truthful run manifest", can_skip=options.no_manifest, skip_reason="--no-manifest" if options.no_manifest else ""),
        CommandStep("write-operator-log", "link operator log to plan and manifest"),
    )
    plan = CommandPlan(
        command=command, workspace=ctx.root.name, run_id=run_id,
        mode="%s:%s" % (options.family, options.action), args=cli_args,
        inputs=tuple(input_refs), outputs=tuple(outputs),
        states=(StateRef("data/run_state.json", "read_write", "resume and completion state"),),
        steps=steps, config_fingerprints=dict(sorted(fingerprints.items())),
        warnings=tuple(warnings), metadata={
            "family": options.family, "action": options.action,
            "target_keys": [item.key for item in targets],
            "target_count": len(targets), "anchor_resolution": anchor_resolution,
            "publication_policy": "overwrite_all" if options.action == "full" else "merge_successes",
        },
    )
    plan_fingerprint = fingerprint_command_plan(plan)
    return PreparedTrainingRun(
        ctx=ctx, options=options, cli_args=cli_args, targets=tuple(targets), plan=plan,
        plan_fingerprint=plan_fingerprint, input_fingerprints=fingerprints,
        source_snapshot=source_snapshot, anchor_resolution=anchor_resolution,
    )


def prepared_plan_json(prepared: PreparedTrainingRun) -> dict:
    return {"schema_version": 1, "plan_fingerprint": prepared.plan_fingerprint, "plan": command_plan_to_public_dict(prepared.plan)}


def render_prepared_plan(prepared: PreparedTrainingRun) -> str:
    return "%s\n\nPlan fingerprint: %s" % (render_command_plan(prepared.plan), prepared.plan_fingerprint)


def options_from_namespace(args: argparse.Namespace, family: str) -> TrainingRunOptions:
    action = "predict_only" if getattr(args, "predict_only", False) else (
        "full" if getattr(args, "full", False) else "incremental"
    )
    csv = lambda value: tuple(item.strip() for item in (value or "").split(",") if item.strip())
    return TrainingRunOptions(
        family=family, action=action, models=csv(getattr(args, "models", None)),
        algorithm=getattr(args, "algorithm", None), dataset=getattr(args, "dataset", None),
        market=getattr(args, "market", None), tag=getattr(args, "tag", None),
        all_enabled=bool(
            getattr(args, "all_enabled", False)
            or (getattr(args, "predict_only", False) and getattr(args, "full", False))
        ), skip=csv(getattr(args, "skip", None)),
        resume=bool(getattr(args, "resume", False)),
        source_records=getattr(args, "source_records", "latest_train_records.json"),
        experiment_name=getattr(args, "experiment_name", None),
        no_pretrain=bool(getattr(args, "no_pretrain", False)),
        cache_size_mb=getattr(args, "cache_size", None),
        explain_plan=bool(getattr(args, "explain_plan", False) or getattr(args, "dry_run", False)),
        json_plan=bool(getattr(args, "json_plan", False)),
        no_manifest=bool(getattr(args, "no_manifest", False)), run_id=getattr(args, "run_id", None),
    )
