"""Workspace-scoped execution lease for real training commands."""

from __future__ import annotations

import json
import os
from pathlib import Path

from quantpits.training.errors import TrainingLeaseError

try:
    import fcntl
except ImportError:  # pragma: no cover - production is Linux
    fcntl = None


class TrainingExecutionLease:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self._handle = None

    @classmethod
    def for_workspace(cls, ctx):
        return cls(ctx.data_path("locks", "training_execution.lock"))

    def acquire(self, *, run_id: str, non_blocking: bool = True) -> None:
        if self._handle is not None:
            raise TrainingLeaseError("training execution lease is already held")
        if fcntl is None:
            raise TrainingLeaseError("cross-process training lease is unavailable on this platform")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if non_blocking else 0)
        try:
            fcntl.flock(handle.fileno(), flags)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            raise TrainingLeaseError("another training command owns the workspace lease") from exc
        handle.seek(0)
        handle.truncate()
        json.dump({"run_id": run_id, "pid": os.getpid()}, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self):
        if self._handle is None:
            raise RuntimeError("acquire the training lease before entering it")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False
