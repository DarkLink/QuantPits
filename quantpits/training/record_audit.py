"""Read-only structural audit and V1-to-V2 migration preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .records import (
    TrainingRecordSchemaError,
    UnsupportedTrainingRecordSchemaError,
    snapshot_from_dict,
)


@dataclass(frozen=True)
class RecordAuditIssue:
    code: str
    severity: str
    model_key: str = ""

    def to_dict(self):
        return {"code": self.code, "severity": self.severity, "model_key": self.model_key}


@dataclass(frozen=True)
class TrainingRecordAuditReport:
    schema_version: int
    model_count: int
    issues: Tuple[RecordAuditIssue, ...]
    proposed_v2: Dict[str, Any]

    @property
    def ok(self):
        return not any(item.severity == "error" for item in self.issues)

    def to_public_dict(self, include_preview=False):
        result = {
            "schema_version": self.schema_version,
            "model_count": self.model_count,
            "ok": self.ok,
            "issues": [item.to_dict() for item in self.issues],
        }
        if include_preview:
            result["proposed_v2"] = self.proposed_v2
        return result


def audit_training_records(records: Mapping[str, Any]) -> TrainingRecordAuditReport:
    issues = []
    raw_schema = records.get("schema_version", 1)
    schema = raw_schema if isinstance(raw_schema, int) and not isinstance(raw_schema, bool) else 0
    if schema != 2:
        issues.append(RecordAuditIssue("legacy_record_schema", "warning"))
    try:
        snapshot = snapshot_from_dict(records)
    except UnsupportedTrainingRecordSchemaError:
        issues.append(RecordAuditIssue("unsupported_record_schema", "error"))
        return TrainingRecordAuditReport(schema, 0, tuple(issues), {})
    except TrainingRecordSchemaError as exc:
        code = "missing_model_records_v2" if schema == 2 and "model_records" in str(exc) else "invalid_training_record"
        issues.append(RecordAuditIssue(code, "error"))
        return TrainingRecordAuditReport(schema, 0, tuple(issues), {})
    except (TypeError, ValueError, KeyError):
        issues.append(RecordAuditIssue("invalid_training_record", "error"))
        return TrainingRecordAuditReport(schema, 0, tuple(issues), {})
    experiments_by_mode = {}
    for entry in snapshot.entries:
        experiments_by_mode.setdefault(entry.training_mode, set()).add(entry.experiment_name)
        if not entry.experiment_name:
            issues.append(RecordAuditIssue("missing_experiment_identity", "error", entry.key))
        if entry.status == "legacy_unverified":
            issues.append(RecordAuditIssue("legacy_unverified_identity", "warning", entry.key))
        if (
            entry.operation == "cpcv_predict"
            and entry.source_experiment_name
            and entry.source_experiment_name == entry.experiment_name
        ):
            issues.append(RecordAuditIssue(
                "cpcv_output_source_experiment_confusion", "error", entry.key
            ))
        mode_value = records.get("%s_experiment_name" % entry.training_mode)
        if schema == 1 and mode_value and mode_value != entry.experiment_name:
            issues.append(RecordAuditIssue("mode_experiment_mismatch", "error", entry.key))
    for mode, experiments in sorted(experiments_by_mode.items()):
        if len(experiments) > 1:
            issues.append(RecordAuditIssue("mixed_mode_experiments", "warning", mode))
    return TrainingRecordAuditReport(schema, len(snapshot.entries), tuple(issues), snapshot.to_dict())
