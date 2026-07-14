"""Canonical training-record contracts and persistence."""

from .records import (
    ModelRecordEntry,
    ModelRecordOutcome,
    ResolvedModelRecord,
    TrainingRecordSnapshot,
    build_model_record_entry,
    resolve_model_record,
)
from .record_repository import TrainingRecordBaseline, TrainingRecordRepository
from .command import (
    PreparedTrainingRun, RequestedDatePolicy, TrainingRunOptions,
    TrainingTarget, prepare_training_run,
)
from .resolved import ResolvedTrainingRun, ResolvedTrainingTarget, resolve_training_run
from .runners import TrainingTargetRequest, TrainingTargetResult
from .lease import TrainingExecutionLease
from .history import TrainingHistoryRepository
from .persistence import FileBaseline
from .publication import (
    TrainingPublicationCoordinator, TrainingPublicationIntent, TrainingPublicationReceipt,
)
from .state import TrainingRunState, TrainingStateRepository

__all__ = [
    "ModelRecordEntry", "ModelRecordOutcome", "ResolvedModelRecord",
    "TrainingRecordSnapshot", "TrainingRecordBaseline", "TrainingRecordRepository", "resolve_model_record",
    "build_model_record_entry",
    "PreparedTrainingRun", "RequestedDatePolicy", "TrainingRunOptions", "TrainingTarget", "prepare_training_run",
    "ResolvedTrainingRun", "ResolvedTrainingTarget", "resolve_training_run",
    "TrainingTargetRequest", "TrainingTargetResult",
    "TrainingExecutionLease", "TrainingHistoryRepository", "FileBaseline", "TrainingPublicationCoordinator",
    "TrainingPublicationIntent", "TrainingPublicationReceipt", "TrainingRunState",
    "TrainingStateRepository",
]
