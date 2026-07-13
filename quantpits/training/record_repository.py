"""Locked, atomic persistence for Training Record V2."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from .records import ModelRecordOutcome, TrainingRecordSnapshot, snapshot_from_dict

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows compatibility
    fcntl = None


class TrainingRecordConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingRecordBaseline:
    """Exact current-record state observed during command preparation."""

    file_fingerprint: Optional[str]
    entries: Mapping[str, Optional[str]]


def _fingerprint(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainingRecordRepository:
    def __init__(self, path, *, clock: Optional[Callable[[], datetime]] = None):
        self.path = Path(path).resolve()
        self._clock = clock or datetime.now

    @classmethod
    def for_workspace(cls, ctx, relative_path="latest_train_records.json", **kwargs):
        path = (ctx.root / relative_path).resolve()
        try:
            path.relative_to(ctx.root.resolve())
        except ValueError as exc:
            raise ValueError("training-record path must stay inside workspace") from exc
        return cls(path, **kwargs)

    def load(self) -> TrainingRecordSnapshot:
        with self._lock(shared=True, create=False):
            return self._read_unlocked()

    def baseline(self) -> TrainingRecordBaseline:
        return self.inspect()[1]

    def inspect(self):
        """Return snapshot and exact baseline from one shared-read boundary."""
        with self._lock(shared=True, create=False):
            snapshot = self._read_unlocked()
            baseline = TrainingRecordBaseline(
                file_fingerprint=_fingerprint(self.path),
                entries={key: entry.recorder_id for key, entry in snapshot.entry_map.items()},
            )
            return snapshot, baseline

    def preview_upgrade(self, *, preview_time: Optional[str] = None):
        """Return a deterministic structural audit without locks or writes."""
        from .record_audit import audit_training_records

        if not self.path.exists():
            records = {"models": {}}
        else:
            with self.path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        report = audit_training_records(records)
        if preview_time is None or not report.proposed_v2:
            return report
        proposed = dict(report.proposed_v2)
        proposed["updated_at"] = preview_time
        proposed["timestamp"] = preview_time.replace("T", " ")
        return type(report)(report.schema_version, report.model_count, report.issues, proposed)

    def _read_unlocked(self) -> TrainingRecordSnapshot:
        if not self.path.exists():
            return TrainingRecordSnapshot(())
        with self.path.open("r", encoding="utf-8") as handle:
            return snapshot_from_dict(json.load(handle))

    @contextmanager
    def _lock(self, *, shared=False, create=True):
        if not create and not self.path.parent.exists():
            yield
            return
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        if not create and not lock_path.exists():
            yield
            return
        with lock_path.open("a+") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _validated_outcomes(outcomes: Iterable[ModelRecordOutcome]):
        values = tuple(outcomes)
        keys = [item.key for item in values]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate training-record outcome key")
        return values

    def merge(
        self,
        outcomes: Iterable[ModelRecordOutcome],
        *,
        expected_entries: Optional[Mapping[str, Optional[str]]] = None,
        baseline: Optional[TrainingRecordBaseline] = None,
    ) -> TrainingRecordSnapshot:
        outcomes = self._validated_outcomes(outcomes)
        with self._lock():
            observed = _fingerprint(self.path)
            if baseline is not None and observed != baseline.file_fingerprint:
                raise TrainingRecordConflictError("training record baseline differs from prepared run")
            entries = self._read_unlocked().entry_map
            if baseline is not None:
                expected_entries = baseline.entries
            if expected_entries:
                for key, recorder_id in expected_entries.items():
                    current = entries.get(key)
                    actual = current.recorder_id if current else None
                    if actual != recorder_id:
                        raise TrainingRecordConflictError("selected training record changed during run")
            for outcome in outcomes:
                if outcome.outcome == "success" and outcome.entry is not None:
                    entries[outcome.key] = outcome.entry
            snapshot = self._timestamped(tuple(entries[key] for key in sorted(entries)))
            self._replace(snapshot, observed)
            return snapshot

    def overwrite(
        self,
        outcomes: Iterable[ModelRecordOutcome],
        *,
        expected_file_fingerprint: Optional[str] = None,
        baseline: Optional[TrainingRecordBaseline] = None,
    ) -> TrainingRecordSnapshot:
        outcomes = self._validated_outcomes(outcomes)
        if any(item.outcome != "success" for item in outcomes):
            raise ValueError("incomplete full run cannot overwrite current records")
        entries = [item.entry for item in outcomes if item.entry is not None]
        if not entries:
            raise ValueError("empty overwrite requires an explicit migration workflow")
        with self._lock():
            observed = _fingerprint(self.path)
            expected = baseline.file_fingerprint if baseline is not None else expected_file_fingerprint
            if (baseline is not None or expected_file_fingerprint is not None) and observed != expected:
                raise TrainingRecordConflictError("training record baseline differs from prepared run")
            snapshot = self._timestamped(tuple(sorted(entries, key=lambda item: item.key)))
            self._replace(snapshot, observed)
            return snapshot

    def _timestamped(self, entries):
        updated_at = self._clock().replace(microsecond=0).isoformat()
        return TrainingRecordSnapshot(tuple(entries), updated_at=updated_at)

    def _replace(self, snapshot: TrainingRecordSnapshot, baseline: Optional[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if _fingerprint(self.path) != baseline:
            raise TrainingRecordConflictError("training record changed during update")
        payload = json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        fd, name = tempfile.mkstemp(prefix=".%s." % self.path.name, dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            if _fingerprint(self.path) != baseline:
                raise TrainingRecordConflictError("training record changed before replace")
            os.replace(name, str(self.path))
            directory_fd = os.open(str(self.path.parent), os.O_RDONLY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
        finally:
            if os.path.exists(name): os.unlink(name)
