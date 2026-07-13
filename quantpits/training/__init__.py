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

__all__ = [
    "ModelRecordEntry", "ModelRecordOutcome", "ResolvedModelRecord",
    "TrainingRecordSnapshot", "TrainingRecordRepository", "resolve_model_record",
    "build_model_record_entry",
]
