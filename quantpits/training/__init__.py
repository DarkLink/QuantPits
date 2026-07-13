"""Canonical training-record contracts and persistence."""

from .records import (
    ModelRecordEntry,
    ModelRecordOutcome,
    ResolvedModelRecord,
    TrainingRecordSnapshot,
    build_model_record_entry,
    resolve_model_record,
)
from .record_repository import TrainingRecordRepository
from .command import PreparedTrainingRun, TrainingRunOptions, TrainingTarget, prepare_training_run

__all__ = [
    "ModelRecordEntry", "ModelRecordOutcome", "ResolvedModelRecord",
    "TrainingRecordSnapshot", "TrainingRecordRepository", "resolve_model_record",
    "build_model_record_entry",
    "PreparedTrainingRun", "TrainingRunOptions", "TrainingTarget", "prepare_training_run",
]
