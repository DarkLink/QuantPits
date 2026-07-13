"""Resolved, immutable execution contracts for static and CPCV training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from quantpits.training.command import PreparedTrainingRun, TrainingTarget
from quantpits.training.errors import TrainingExecutionError, TrainingSourceRecordError
from quantpits.training.records import ModelRecordEntry
from quantpits.utils.workspace import fingerprint_value


@dataclass(frozen=True)
class ResolvedTrainingTarget:
    target: TrainingTarget
    operation: str
    source_entry: Optional[ModelRecordEntry] = None

    @property
    def key(self) -> str:
        return self.target.key


@dataclass(frozen=True)
class ResolvedTrainingRun:
    prepared: PreparedTrainingRun
    params: Mapping[str, Any]
    anchor_date: str
    output_experiment_name: str
    targets: Tuple[ResolvedTrainingTarget, ...]
    execution_fingerprint: str
    resume_fingerprint: str


def operation_for(family: str, action: str) -> str:
    if action != "predict_only":
        return "train"
    return "cpcv_predict" if family == "cpcv" else "predict_only"


def default_experiment_name(family: str, action: str, params: Mapping[str, Any]) -> str:
    freq = str(params.get("freq", "week")).upper()
    if family == "cpcv":
        prefix = "Prod_Predict_CPCV" if action == "predict_only" else "Prod_Train_CPCV"
    else:
        prefix = "Prod_Predict" if action == "predict_only" else "Prod_Train"
    return "%s_%s" % (prefix, freq)


def resolve_training_run(prepared: PreparedTrainingRun, params: Mapping[str, Any]) -> ResolvedTrainingRun:
    anchor = str(params.get("anchor_date") or "")
    if not anchor:
        raise TrainingExecutionError("resolved training dates have no anchor_date")
    if prepared.date_policy.configured_anchor and anchor != prepared.date_policy.configured_anchor:
        raise TrainingExecutionError("resolved anchor differs from configured current_date")
    if prepared.options.family == "cpcv" and not params.get("cpcv_folds"):
        raise TrainingExecutionError("CPCV execution requires resolved folds")

    operation = operation_for(prepared.options.family, prepared.options.action)
    source_entries = prepared.source_snapshot.entry_map if prepared.source_snapshot else {}
    targets = []
    for target in prepared.targets:
        source = source_entries.get(target.key) if prepared.options.action == "predict_only" else None
        if prepared.options.action == "predict_only" and source is None:
            raise TrainingSourceRecordError("selected target has no source recorder: %s" % target.key)
        targets.append(ResolvedTrainingTarget(target, operation, source))
    if tuple(item.key for item in targets) != tuple(item.key for item in prepared.targets):
        raise TrainingExecutionError("resolved target set differs from prepared plan")

    experiment = prepared.options.experiment_name or default_experiment_name(
        prepared.options.family, prepared.options.action, params
    )
    resume_payload = {
        "params": dict(params),
        "target_keys": [item.key for item in targets],
        "operations": [item.operation for item in targets],
        "output_experiment_name": experiment,
        "publication_policy": prepared.plan.metadata.get("publication_policy"),
        "no_pretrain": prepared.options.no_pretrain,
        "cache_size_mb": prepared.options.cache_size_mb,
        "semantic_inputs": {
            key: value for key, value in prepared.input_fingerprints.items()
            if key not in ("latest_train_records.json", "data/run_state.json")
        },
    }
    resume_fingerprint = fingerprint_value(resume_payload)
    payload = {
        "plan_fingerprint": prepared.plan_fingerprint,
        "resume_fingerprint": resume_fingerprint,
        "current_record_baseline": prepared.current_record_baseline,
    }
    return ResolvedTrainingRun(
        prepared=prepared,
        params=dict(params),
        anchor_date=anchor,
        output_experiment_name=experiment,
        targets=tuple(targets),
        execution_fingerprint=fingerprint_value(payload),
        resume_fingerprint=resume_fingerprint,
    )
