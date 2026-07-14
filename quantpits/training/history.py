"""Explicit-workspace, idempotent post-publication history events."""

from __future__ import annotations

import json
import os
from pathlib import Path

from quantpits.utils.workspace import fingerprint_value

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class TrainingHistoryRepository:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    @classmethod
    def for_run(cls, run):
        filename = "prediction_history.jsonl" if run.prepared.options.action == "predict_only" else "training_history.jsonl"
        return cls(run.prepared.ctx.data_path(filename))

    def append(self, event):
        value = dict(event)
        value.setdefault("event_id", fingerprint_value({
            key: value.get(key) for key in (
                "operation", "model_name", "mode", "record_id", "source_record_id"
            )
        }))
        line = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if self.path.is_file():
                    for existing in self.path.read_text(encoding="utf-8").splitlines():
                        try:
                            if json.loads(existing).get("event_id") == value["event_id"]:
                                return False
                        except (ValueError, AttributeError):
                            continue
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                return True
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
