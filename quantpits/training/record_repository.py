"""Locked, atomic persistence for Training Record V2."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping, Optional

from .records import ModelRecordOutcome, TrainingRecordSnapshot, snapshot_from_dict

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows compatibility
    fcntl = None


class TrainingRecordConflictError(RuntimeError):
    pass


def _fingerprint(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TrainingRecordRepository:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def load(self) -> TrainingRecordSnapshot:
        if not self.path.exists():
            return TrainingRecordSnapshot(())
        with self.path.open("r", encoding="utf-8") as handle:
            return snapshot_from_dict(json.load(handle))

    @contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def merge(self, outcomes: Iterable[ModelRecordOutcome]) -> TrainingRecordSnapshot:
        with self._lock():
            baseline = _fingerprint(self.path)
            entries = self.load().entry_map
            for outcome in outcomes:
                if outcome.outcome == "success" and outcome.entry is not None:
                    entries[outcome.key] = outcome.entry
            snapshot = TrainingRecordSnapshot(tuple(entries[key] for key in sorted(entries)))
            self._replace(snapshot, baseline)
            return snapshot

    def overwrite(self, outcomes: Iterable[ModelRecordOutcome]) -> TrainingRecordSnapshot:
        with self._lock():
            baseline = _fingerprint(self.path)
            outcomes = tuple(outcomes)
            if any(item.outcome != "success" for item in outcomes):
                raise ValueError("incomplete full run cannot overwrite current records")
            entries = [item.entry for item in outcomes if item.entry is not None]
            if not entries:
                raise ValueError("empty overwrite requires an explicit migration workflow")
            snapshot = TrainingRecordSnapshot(tuple(sorted(entries, key=lambda item: item.key)))
            self._replace(snapshot, baseline)
            return snapshot

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
