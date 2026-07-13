"""Recoverable local-filesystem transaction for post-trade account state."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import string
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Tuple

from quantpits.post_trade.contracts import (
    PostTradeConcurrentRunError, PostTradeRecoveryRequiredError,
    PostTradeTransactionConflictError, PostTradeTransactionCorruptError,
    PostTradeTransactionSchemaError,
)
from quantpits.utils.workspace import WorkspaceContext

STATUSES = {"prepared", "committing", "state_committed", "completed", "conflicted"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PostTradeTransactionSchemaError("Transaction paths must be workspace-relative")
    return path.as_posix()


@dataclass(frozen=True)
class PostTradeTransactionArtifact:
    path: str
    role: str
    order: int
    baseline_sha256: Optional[str]
    target_sha256: str
    staged_path: str
    required: bool = True

    def __post_init__(self):
        object.__setattr__(self, "path", _relative(self.path))
        object.__setattr__(self, "staged_path", _relative(self.staged_path))
        hashes = (self.target_sha256,) + ((self.baseline_sha256,) if self.baseline_sha256 is not None else ())
        if any(len(value) != 64 or any(ch not in string.hexdigits for ch in value) for value in hashes):
            raise PostTradeTransactionSchemaError("Invalid transaction fingerprint")


@dataclass(frozen=True)
class DependentStep:
    status: str = "pending"
    attempts: int = 0
    last_error: Optional[str] = None
    output_paths: Tuple[str, ...] = ()
    output_fingerprints: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.status not in {"pending", "running", "success", "failed", "skipped"}:
            raise PostTradeTransactionSchemaError("Invalid classification status")
        if self.attempts < 0:
            raise PostTradeTransactionSchemaError("Invalid classification attempt count")
        paths = tuple(_relative(path) for path in self.output_paths)
        fingerprints = tuple((_relative(path), value) for path, value in self.output_fingerprints)
        if any(path not in paths for path, _ in fingerprints):
            raise PostTradeTransactionSchemaError("Classification fingerprint has no matching output path")
        if any(len(value) != 64 or any(ch not in string.hexdigits for ch in value) for _, value in fingerprints):
            raise PostTradeTransactionSchemaError("Invalid classification output fingerprint")
        object.__setattr__(self, "output_paths", paths)
        object.__setattr__(self, "output_fingerprints", fingerprints)


@dataclass(frozen=True)
class PostTradeTransactionJournal:
    transaction_id: str
    run_id: str
    workspace: str
    scope: str
    status: str
    created_at: str
    updated_at: str
    light_plan_fingerprint: str
    resolved_execution_fingerprint: str
    cursor_before: str
    cursor_after: str
    processed_dates: Tuple[str, ...]
    consumed_cashflow_dates: Tuple[str, ...]
    artifacts: Tuple[PostTradeTransactionArtifact, ...]
    committed_artifacts: Tuple[str, ...] = ()
    classification: DependentStep = DependentStep()
    manifest_path: Optional[str] = None
    error: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or self.status not in STATUSES:
            raise PostTradeTransactionSchemaError("Unsupported transaction journal")
        paths = [item.path for item in self.artifacts]
        orders = [item.order for item in self.artifacts]
        if len(paths) != len(set(paths)) or len(orders) != len(set(orders)):
            raise PostTradeTransactionSchemaError("Duplicate transaction artifact")
        required = [item for item in self.artifacts if item.required]
        if required and required[-1].role != "prod_config_cursor":
            raise PostTradeTransactionSchemaError("prod_config_cursor must be the final required artifact")

    def to_dict(self):
        return {
            "schema_version": self.schema_version, "transaction_id": self.transaction_id,
            "run_id": self.run_id, "workspace": self.workspace, "scope": self.scope,
            "status": self.status, "created_at": self.created_at, "updated_at": self.updated_at,
            "light_plan_fingerprint": self.light_plan_fingerprint,
            "resolved_execution_fingerprint": self.resolved_execution_fingerprint,
            "cursor_before": self.cursor_before, "cursor_after": self.cursor_after,
            "processed_dates": list(self.processed_dates),
            "consumed_cashflow_dates": list(self.consumed_cashflow_dates),
            "artifacts": [item.__dict__ for item in self.artifacts],
            "committed_artifacts": list(self.committed_artifacts),
            "dependent_steps": {"trade_classification": {
                "status": self.classification.status, "attempts": self.classification.attempts,
                "last_error": self.classification.last_error,
                "output_paths": list(self.classification.output_paths),
                "output_fingerprints": [
                    {"path": path, "sha256": fingerprint}
                    for path, fingerprint in self.classification.output_fingerprints
                ],
            }},
            "manifest_path": self.manifest_path, "error": self.error,
        }

    @classmethod
    def from_dict(cls, value):
        try:
            step = value.get("dependent_steps", {}).get("trade_classification", {})
            return cls(
                transaction_id=value["transaction_id"], run_id=value["run_id"], workspace=value["workspace"],
                scope=value["scope"], status=value["status"], created_at=value["created_at"],
                updated_at=value["updated_at"], light_plan_fingerprint=value["light_plan_fingerprint"],
                resolved_execution_fingerprint=value["resolved_execution_fingerprint"],
                cursor_before=value["cursor_before"], cursor_after=value["cursor_after"],
                processed_dates=tuple(value.get("processed_dates", ())),
                consumed_cashflow_dates=tuple(value.get("consumed_cashflow_dates", ())),
                artifacts=tuple(PostTradeTransactionArtifact(**item) for item in value.get("artifacts", ())),
                committed_artifacts=tuple(value.get("committed_artifacts", ())),
                classification=DependentStep(
                    step.get("status", "pending"), int(step.get("attempts", 0)),
                    step.get("last_error"), tuple(step.get("output_paths", ())),
                    tuple((item["path"], item["sha256"]) for item in step.get("output_fingerprints", ())),
                ),
                manifest_path=value.get("manifest_path"), error=value.get("error"),
                schema_version=int(value.get("schema_version", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PostTradeTransactionSchemaError("Malformed transaction journal") from exc


def _atomic(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
        try:
            directory = os.open(path.parent, os.O_DIRECTORY); os.fsync(directory); os.close(directory)
        except (AttributeError, OSError):
            pass
    finally:
        if os.path.exists(name): os.unlink(name)


class PostTradeTransactionManager:
    def __init__(self, ctx: WorkspaceContext, event_hook=None):
        self.ctx = ctx
        self.root = ctx.data_path(".post_trade_transactions")
        self._event_hook = event_hook or (lambda event, artifact: None)

    def journal_path(self, transaction_id: str) -> Path:
        if not transaction_id or "/" in transaction_id or ".." in transaction_id:
            raise PostTradeTransactionSchemaError("Invalid transaction id")
        return self.root / transaction_id / "journal.json"

    def write_journal(self, journal: PostTradeTransactionJournal):
        payload = (json.dumps(journal.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        _atomic(self.journal_path(journal.transaction_id), payload)

    def load(self, transaction_id: str) -> PostTradeTransactionJournal:
        try:
            value = json.loads(self.journal_path(transaction_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PostTradeTransactionCorruptError("Cannot read transaction journal: %s" % transaction_id) from exc
        journal = PostTradeTransactionJournal.from_dict(value)
        prefix = "data/.post_trade_transactions/%s/staged/" % transaction_id
        if any(not item.staged_path.startswith(prefix) for item in journal.artifacts):
            raise PostTradeTransactionSchemaError("Staged payload is outside its transaction directory")
        return journal

    def active(self) -> Tuple[PostTradeTransactionJournal, ...]:
        if not self.root.exists(): return ()
        journals = []
        for path in sorted(self.root.glob("*/journal.json")):
            journal = self.load(path.parent.name)
            if journal.status in {"prepared", "committing"}:
                journals.append(journal)
            elif journal.status == "state_committed" and journal.classification.status in {"pending", "running"}:
                journals.append(journal)
        return tuple(journals)

    @contextmanager
    def lock(self):
        path = self.ctx.data_path(".post_trade.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
        try:
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc: raise PostTradeConcurrentRunError("Another post-trade run holds the workspace lock") from exc
            yield
        finally:
            try: fcntl.flock(handle, fcntl.LOCK_UN)
            finally: handle.close()

    def prepare(self, *, transaction_id: str, run_id: str, scope: str, light_fingerprint: str,
                resolved_fingerprint: str, cursor_before: str, cursor_after: str,
                processed_dates: Iterable[str], consumed_cashflow_dates: Iterable[str], payloads: Iterable[tuple]):
        existing_path = self.journal_path(transaction_id)
        if existing_path.exists():
            existing = self.load(transaction_id)
            if existing.resolved_execution_fingerprint != resolved_fingerprint:
                raise PostTradeTransactionConflictError("Run id is already bound to another execution")
            return existing
        if self.active(): raise PostTradeRecoveryRequiredError("An unfinished post-trade transaction requires recovery")
        now = datetime.now().isoformat()
        artifacts = []
        tx_root = self.journal_path(transaction_id).parent
        for order, role, path, payload in payloads:
            target = Path(path)
            try: relative = target.resolve().relative_to(self.ctx.root).as_posix()
            except ValueError as exc: raise PostTradeTransactionSchemaError("Target is outside workspace") from exc
            staged_rel = (tx_root / "staged" / ("%03d_%s.bin" % (order, role))).relative_to(self.ctx.root).as_posix()
            staged = self.ctx.root / staged_rel
            _atomic(staged, payload)
            target_hash = sha256_bytes(payload)
            if sha256_file(staged) != target_hash: raise PostTradeTransactionCorruptError("Staged payload verification failed")
            artifacts.append(PostTradeTransactionArtifact(relative, role, order, sha256_file(target), target_hash, staged_rel))
        artifacts.sort(key=lambda item: item.order)
        journal = PostTradeTransactionJournal(transaction_id, run_id, self.ctx.root.name, scope, "prepared", now, now,
            light_fingerprint, resolved_fingerprint, cursor_before, cursor_after, tuple(processed_dates),
            tuple(consumed_cashflow_dates), tuple(artifacts))
        self.write_journal(journal)
        return journal

    def commit(self, journal: PostTradeTransactionJournal):
        journal = replace(journal, status="committing", updated_at=datetime.now().isoformat())
        self.write_journal(journal)
        committed = list(journal.committed_artifacts)
        for artifact in sorted(journal.artifacts, key=lambda item: item.order):
            target, staged = self.ctx.root / artifact.path, self.ctx.root / artifact.staged_path
            current = sha256_file(target)
            if current == artifact.target_sha256:
                pass
            elif current == artifact.baseline_sha256:
                if sha256_file(staged) != artifact.target_sha256:
                    raise PostTradeTransactionCorruptError("Staged payload is corrupt: %s" % artifact.role)
                self._event_hook("before_target_write", artifact)
                _atomic(target, staged.read_bytes())
                self._event_hook("after_target_write_before_verification", artifact)
                if sha256_file(target) != artifact.target_sha256:
                    raise PostTradeTransactionCorruptError("Target verification failed: %s" % artifact.role)
                self._event_hook("after_verification_before_journal", artifact)
            else:
                conflicted = replace(journal, status="conflicted", error="third_version:%s" % artifact.role, updated_at=datetime.now().isoformat())
                self.write_journal(conflicted)
                raise PostTradeTransactionConflictError("Transaction target has a third version: %s" % artifact.role)
            if artifact.path not in committed:
                committed.append(artifact.path)
                journal = replace(journal, committed_artifacts=tuple(committed), updated_at=datetime.now().isoformat())
                self.write_journal(journal)
            self._event_hook("after_journal_before_next_target", artifact)
        journal = replace(journal, status="state_committed", updated_at=datetime.now().isoformat())
        self.write_journal(journal)
        for artifact in journal.artifacts:
            staged = self.ctx.root / artifact.staged_path
            if staged.exists(): staged.unlink()
        return journal

    def recover(self, transaction_id: str):
        journal = self.load(transaction_id)
        if journal.status == "conflicted": raise PostTradeTransactionConflictError("Transaction is conflicted")
        if journal.status in {"state_committed", "completed"}:
            self.verify_committed(journal)
            return journal
        recovered = self.commit(journal)
        self.verify_committed(recovered)
        return recovered

    def verify_committed(self, journal):
        if journal.status not in {"state_committed", "completed"}:
            raise PostTradeTransactionConflictError("Transaction state is not committed")
        for artifact in journal.artifacts:
            if sha256_file(self.ctx.root / artifact.path) != artifact.target_sha256:
                raise PostTradeTransactionConflictError("Committed transaction artifact changed: %s" % artifact.role)
        return True

    def verified_target_paths(self, journal):
        return tuple(
            artifact.path for artifact in journal.artifacts
            if sha256_file(self.ctx.root / artifact.path) == artifact.target_sha256
        )

    def set_classification(self, journal, *, status, error=None, output_paths=(), output_fingerprints=()):
        if status not in {"pending", "running", "success", "failed", "skipped"}:
            raise PostTradeTransactionSchemaError("Invalid classification status")
        attempts = journal.classification.attempts + (1 if status == "running" else 0)
        paths = tuple(_relative(path) for path in output_paths)
        fingerprints = tuple((_relative(path), fingerprint) for path, fingerprint in output_fingerprints)
        if any(path not in paths for path, _ in fingerprints):
            raise PostTradeTransactionSchemaError("Classification fingerprint has no matching output path")
        if any(len(value) != 64 or any(ch not in string.hexdigits for ch in value) for _, value in fingerprints):
            raise PostTradeTransactionSchemaError("Invalid classification output fingerprint")
        step = DependentStep(status, attempts, error, paths, fingerprints)
        updated = replace(journal, classification=step, updated_at=datetime.now().isoformat())
        self.write_journal(updated)
        return updated

    def complete(self, journal, *, manifest_path=None):
        updated = replace(journal, status="completed", manifest_path=manifest_path, updated_at=datetime.now().isoformat())
        self.write_journal(updated)
        return updated

    def status_summary(self):
        if not self.root.exists(): return ()
        values = []
        for path in sorted(self.root.glob("*/journal.json")):
            journal = self.load(path.parent.name)
            values.append({
                "transaction_id": journal.transaction_id, "status": journal.status,
                "scope": journal.scope, "processed_date_count": len(journal.processed_dates),
                "date_from": journal.processed_dates[0] if journal.processed_dates else None,
                "date_to": journal.processed_dates[-1] if journal.processed_dates else None,
                "artifact_count": len(journal.artifacts),
                "classification_status": journal.classification.status,
            })
        return tuple(values)
