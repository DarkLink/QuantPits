"""One-target adapters over the existing Qlib model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from quantpits.training.errors import TrainingRunnerContractError
from quantpits.training.records import ModelRecordEntry
from quantpits.training.resolved import ResolvedTrainingRun, ResolvedTrainingTarget


@dataclass(frozen=True)
class TrainingTargetRequest:
    resolved_run: ResolvedTrainingRun
    target: ResolvedTrainingTarget
    cache: Any = None


@dataclass(frozen=True)
class TrainingTargetResult:
    key: str
    operation: str
    outcome: str
    entry: Optional[ModelRecordEntry] = None
    performance: Optional[Mapping[str, Any]] = None
    error_code: Optional[str] = None
    error_type: Optional[str] = None
    history_event: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        if self.outcome not in ("success", "failed", "skipped", "preserved"):
            raise ValueError("unsupported training target outcome")
        if self.outcome == "success" and self.entry is None:
            raise ValueError("successful target requires record evidence")
        if self.entry is not None and (self.entry.key != self.key or self.entry.operation != self.operation):
            raise ValueError("target result identity differs from record evidence")


def _adapt_result(request: TrainingTargetRequest, raw: object) -> TrainingTargetResult:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("success"), bool):
        raise TrainingRunnerContractError("model helper returned an invalid result contract")
    target = request.target
    if not raw["success"]:
        if raw.get("record_entry") is not None:
            raise TrainingRunnerContractError("failed model helper returned publishable evidence")
        return TrainingTargetResult(
            target.key, target.operation, "failed",
            error_code="target_execution_failed",
            error_type=type(raw.get("error")).__name__ if raw.get("error") is not None else None,
        )
    value = raw.get("record_entry")
    if not isinstance(value, (ModelRecordEntry, Mapping)):
        raise TrainingRunnerContractError("successful model helper returned no record entry")
    entry = value if isinstance(value, ModelRecordEntry) else ModelRecordEntry.from_dict(value)
    if entry.key != target.key or entry.operation != target.operation:
        raise TrainingRunnerContractError("model helper returned record evidence for another target")
    if entry.requested_anchor != request.resolved_run.anchor_date:
        raise TrainingRunnerContractError("model helper returned evidence for another anchor")
    if entry.experiment_name != request.resolved_run.output_experiment_name:
        raise TrainingRunnerContractError("model helper returned evidence for another experiment")
    performance = raw.get("performance")
    if performance is not None and not isinstance(performance, Mapping):
        raise TrainingRunnerContractError("model helper performance must be a mapping")
    history_event = raw.get("history_entry")
    if history_event is not None and not isinstance(history_event, Mapping):
        raise TrainingRunnerContractError("model helper history event must be a mapping")
    return TrainingTargetResult(
        target.key, target.operation, "success", entry, performance,
        history_event=history_event,
    )


def run_static_target(request: TrainingTargetRequest) -> TrainingTargetResult:
    from quantpits.utils import train_utils

    run = request.resolved_run
    target = request.target
    workflow = str(run.prepared.ctx.path(target.target.workflow_path))
    history_file = False
    workspace_root = str(run.prepared.ctx.root)
    if target.operation == "train":
        raw = train_utils.train_single_model(
            target.target.model_name, workflow, dict(run.params), run.output_experiment_name,
            no_pretrain=run.prepared.options.no_pretrain, cache_mgr=request.cache, mode="static",
            history_file=history_file, workspace_root=workspace_root,
        )
    else:
        source = target.source_entry
        source_records = TrainingSourceView(source).to_dict()
        raw = train_utils.predict_single_model(
            target.target.model_name,
            {"yaml_file": workflow},
            dict(run.params), run.output_experiment_name, source_records,
            no_pretrain=run.prepared.options.no_pretrain, cache_mgr=request.cache,
            history_file=history_file, workspace_root=workspace_root,
        )
    return _adapt_result(request, raw)


def run_cpcv_target(request: TrainingTargetRequest) -> TrainingTargetResult:
    from quantpits.utils import train_utils

    run = request.resolved_run
    target = request.target
    workflow = str(run.prepared.ctx.path(target.target.workflow_path))
    history_file = False
    workspace_root = str(run.prepared.ctx.root)
    if target.operation == "train":
        raw = train_utils.train_cpcv_model(
            target.target.model_name, workflow, dict(run.params), run.output_experiment_name,
            no_pretrain=run.prepared.options.no_pretrain, cache_mgr=request.cache, mode="cpcv",
            history_file=history_file, workspace_root=workspace_root,
        )
    else:
        source = target.source_entry
        raw = train_utils.predict_cpcv_model(
            target.target.model_name,
            {"yaml_file": workflow, "record_id": source.recorder_id},
            dict(run.params), run.output_experiment_name,
            no_pretrain=run.prepared.options.no_pretrain, cache_mgr=request.cache,
            source_experiment_name=source.experiment_name,
            source_operation=source.operation,
            history_file=history_file, workspace_root=workspace_root,
        )
    return _adapt_result(request, raw)


class TrainingSourceView:
    """Minimal legacy view for one exact source entry."""

    def __init__(self, entry: ModelRecordEntry):
        self.entry = entry

    def to_dict(self):
        mode_field = "%s_experiment_name" % self.entry.training_mode
        return {
            "schema_version": 2,
            "models": {self.entry.key: self.entry.recorder_id},
            "model_records": {self.entry.key: self.entry.to_dict()},
            mode_field: self.entry.experiment_name,
            "experiment_name": self.entry.experiment_name,
        }
