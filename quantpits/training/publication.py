"""Crash-recoverable publication of current training outputs."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from quantpits.training.errors import TrainingPublicationError, TrainingPublicationRecoveryError
from quantpits.training.persistence import (
    FileBaseline, atomic_write_bytes, atomic_write_json_bytes, baseline_matches,
    fsync_directory, read_with_baseline, sha256_bytes,
)
from quantpits.training.record_repository import TrainingRecordRepository
from quantpits.training.records import ModelRecordOutcome

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


@dataclass(frozen=True)
class PublicationMember:
    relative_path: str
    kind: str
    preimage: FileBaseline
    postimage_fingerprint: str
    staged_relative_path: str
    preimage_relative_path: Optional[str]
    commit_order: int

    def to_dict(self):
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "preimage": self.preimage.to_dict(),
            "postimage_fingerprint": self.postimage_fingerprint,
            "staged_relative_path": self.staged_relative_path,
            "preimage_relative_path": self.preimage_relative_path,
            "commit_order": self.commit_order,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            relative_path=str(value["relative_path"]), kind=str(value["kind"]),
            preimage=FileBaseline.from_dict(value["preimage"]),
            postimage_fingerprint=str(value["postimage_fingerprint"]),
            staged_relative_path=str(value["staged_relative_path"]),
            preimage_relative_path=value.get("preimage_relative_path"),
            commit_order=int(value["commit_order"]),
        )


@dataclass(frozen=True)
class TrainingPublicationIntent:
    transaction_id: str
    run_id: str
    family: str
    action: str
    plan_fingerprint: str
    execution_fingerprint: str
    resume_fingerprint: str
    published_keys: Tuple[str, ...]
    members: Tuple[PublicationMember, ...]
    performance_omitted_reason: Optional[str] = None
    schema_version: int = 1

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "transaction_id": self.transaction_id,
            "run_id": self.run_id,
            "family": self.family,
            "action": self.action,
            "plan_fingerprint": self.plan_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "resume_fingerprint": self.resume_fingerprint,
            "published_keys": list(self.published_keys),
            "members": [item.to_dict() for item in self.members],
            "performance_omitted_reason": self.performance_omitted_reason,
        }

    @classmethod
    def from_dict(cls, value):
        if value.get("schema_version") != 1:
            raise TrainingPublicationRecoveryError("unsupported training publication intent")
        return cls(
            transaction_id=str(value["transaction_id"]), run_id=str(value["run_id"]),
            family=str(value["family"]), action=str(value["action"]),
            plan_fingerprint=str(value["plan_fingerprint"]),
            execution_fingerprint=str(value["execution_fingerprint"]),
            resume_fingerprint=str(value.get("resume_fingerprint") or value["execution_fingerprint"]),
            published_keys=tuple(value.get("published_keys", ())),
            members=tuple(PublicationMember.from_dict(item) for item in value.get("members", ())),
            performance_omitted_reason=value.get("performance_omitted_reason"),
        )


@dataclass(frozen=True)
class TrainingPublicationReceipt:
    transaction_id: str
    run_id: str
    status: str
    published_keys: Tuple[str, ...]
    committed_outputs: Tuple[Mapping[str, str], ...]
    recovery_action: Optional[str] = None
    schema_version: int = 1

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "transaction_id": self.transaction_id,
            "run_id": self.run_id,
            "status": self.status,
            "published_keys": list(self.published_keys),
            "committed_outputs": [dict(item) for item in self.committed_outputs],
            "recovery_action": self.recovery_action,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            transaction_id=str(value["transaction_id"]), run_id=str(value["run_id"]),
            status=str(value["status"]), published_keys=tuple(value.get("published_keys", ())),
            committed_outputs=tuple(value.get("committed_outputs", ())),
            recovery_action=value.get("recovery_action"),
        )


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TrainingPublicationError("performance output contains an unsupported value")


class TrainingPublicationCoordinator:
    def __init__(self, run, *, clock, fault_hook=None):
        self.run = run
        self.ctx = run.prepared.ctx
        self.clock = clock
        self.transaction_id = "%s-%s" % (
            run.prepared.plan.run_id, run.execution_fingerprint[:12]
        )
        self.directory = self.ctx.data_path("training_transactions", self.transaction_id)
        self.intent_path = self.directory / "intent.json"
        self.receipt_path = self.directory / "receipt.json"
        self.lock_path = self.ctx.data_path("locks", "training_publication.lock")
        self.fault_hook = fault_hook or (lambda _point, _member=None: None)
        self.recovery_resume_fingerprint = None

    def _set_transaction_id(self, transaction_id):
        self.transaction_id = transaction_id
        self.directory = self.ctx.data_path("training_transactions", transaction_id)
        self.intent_path = self.directory / "intent.json"
        self.receipt_path = self.directory / "receipt.json"

    @classmethod
    def fresh_for_run(cls, run, *, clock):
        candidate = cls(run, clock=clock)
        if not candidate.intent_path.exists() and not candidate.receipt_path.exists():
            return candidate
        base = candidate.transaction_id
        attempt = 2
        while True:
            transaction_id = "%s-a%02d" % (base, attempt)
            directory = run.prepared.ctx.data_path("training_transactions", transaction_id)
            if not directory.exists():
                candidate._set_transaction_id(transaction_id)
                return candidate
            attempt += 1

    @contextmanager
    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _contained(self, relative: str) -> Path:
        path = (self.ctx.root / relative).resolve()
        try:
            path.relative_to(self.ctx.root.resolve())
        except ValueError as exc:
            raise TrainingPublicationError("publication path leaves workspace") from exc
        return path

    def _performance_payload(self, results, published_keys):
        values = {
            item.key.rsplit("@", 1)[0]: _json_safe(dict(item.performance))
            for item in results
            if item.key in published_keys and item.performance is not None
        }
        if not values:
            return None, "no_published_performance"
        if self.run.prepared.options.action != "full" and self.run.performance_baseline.existed:
            raw, observed = read_with_baseline(
                self.run.performance_path,
                display_path=self.run.performance_relative_path,
            )
            if observed.fingerprint != self.run.performance_baseline.fingerprint:
                raise TrainingPublicationError("performance output changed during training")
            try:
                existing = json.loads(raw.decode("utf-8"))
            except (ValueError, AttributeError) as exc:
                raise TrainingPublicationError("existing performance output is invalid") from exc
            if not isinstance(existing, dict):
                raise TrainingPublicationError("existing performance output must be an object")
            for model_name, new_value in values.items():
                old_value = existing.get(model_name)
                if isinstance(old_value, Mapping) and "convergence" in old_value and "convergence" not in new_value:
                    new_value["convergence"] = old_value["convergence"]
                existing[model_name] = new_value
            values = existing
        return _json_bytes(values), None

    def prepare(self, results) -> Optional[TrainingPublicationIntent]:
        successes = tuple(item for item in results if item.outcome == "success")
        failures = tuple(item for item in results if item.outcome == "failed")
        if self.run.prepared.options.action == "full" and (failures or len(successes) != len(self.run.targets)):
            return None
        if not successes:
            return None
        published_keys = tuple(item.key for item in successes)
        outcomes = tuple(ModelRecordOutcome(
            key=item.key, requested_operation=item.operation, outcome=item.outcome,
            entry=item.entry if item.outcome == "success" else None,
            error_type=item.error_type, error_code=item.error_code,
        ) for item in results)
        repository = TrainingRecordRepository.for_workspace(self.ctx, clock=self.clock)
        timestamp = self.clock().replace(microsecond=0).isoformat()
        if self.run.prepared.options.action == "full":
            snapshot = repository.project_overwrite(self.run.prepared.current_snapshot, outcomes, timestamp=timestamp)
        else:
            snapshot = repository.project_merge(self.run.prepared.current_snapshot, outcomes, timestamp=timestamp)
        record_payload = repository.serialize(snapshot)
        performance_payload, omitted_reason = self._performance_payload(results, frozenset(published_keys))

        self.directory.mkdir(parents=True, exist_ok=True)
        members = []
        specs = []
        if performance_payload is not None:
            specs.append((self.run.performance_relative_path, "performance", performance_payload, self.run.performance_baseline, 10))
        record_path = self.ctx.root / "latest_train_records.json"
        record_raw, record_baseline = read_with_baseline(record_path, display_path="latest_train_records.json")
        expected = self.run.prepared.current_record_baseline.file_fingerprint
        if record_baseline.fingerprint != expected:
            raise TrainingPublicationError("training record changed during execution")
        specs.append(("latest_train_records.json", "record", record_payload, record_baseline, 20))
        for index, (relative, kind, payload, baseline, order) in enumerate(specs):
            staged = "member-%02d.postimage" % index
            atomic_write_bytes(self.directory / staged, payload)
            self.fault_hook("after_postimage_stage", relative)
            preimage_name = None
            target = self._contained(relative)
            raw, observed = read_with_baseline(target, display_path=relative)
            if observed.fingerprint != baseline.fingerprint or observed.existed != baseline.existed:
                raise TrainingPublicationError("publication baseline changed: %s" % relative)
            if raw is not None:
                preimage_name = "member-%02d.preimage" % index
                atomic_write_bytes(self.directory / preimage_name, raw)
            members.append(PublicationMember(
                relative, kind, observed, sha256_bytes(payload), staged, preimage_name, order,
            ))
        intent = TrainingPublicationIntent(
            transaction_id=self.transaction_id, run_id=self.run.prepared.plan.run_id,
            family=self.run.prepared.options.family, action=self.run.prepared.options.action,
            plan_fingerprint=self.run.prepared.plan_fingerprint,
            execution_fingerprint=self.run.execution_fingerprint,
            resume_fingerprint=self.run.resume_fingerprint,
            published_keys=published_keys, members=tuple(members),
            performance_omitted_reason=omitted_reason,
        )
        atomic_write_json_bytes(self.intent_path, _json_bytes(intent.to_dict()))
        self.fault_hook("after_intent_write")
        return intent

    def load_intent(self) -> Optional[TrainingPublicationIntent]:
        if not self.intent_path.is_file():
            return None
        return TrainingPublicationIntent.from_dict(json.loads(self.intent_path.read_text(encoding="utf-8")))

    def load_receipt(self) -> Optional[TrainingPublicationReceipt]:
        if not self.receipt_path.is_file():
            return None
        return TrainingPublicationReceipt.from_dict(json.loads(self.receipt_path.read_text(encoding="utf-8")))

    def _verify_identity(self, intent, *, recovery=False):
        if recovery:
            expected = (
                self.run.prepared.plan.run_id, self.run.prepared.options.family,
                self.run.prepared.options.action,
                self.recovery_resume_fingerprint or self.run.resume_fingerprint,
            )
            actual = (intent.run_id, intent.family, intent.action, intent.resume_fingerprint)
            if actual != expected:
                raise TrainingPublicationRecoveryError("publication intent belongs to another resolved run")
            return
        expected = (
            self.run.prepared.plan.run_id, self.run.prepared.plan_fingerprint,
            self.run.execution_fingerprint,
        )
        actual = (intent.run_id, intent.plan_fingerprint, intent.execution_fingerprint)
        if actual != expected:
            raise TrainingPublicationRecoveryError("publication intent belongs to another execution")

    def _member_state(self, member):
        target = self._contained(member.relative_path)
        _raw, observed = read_with_baseline(target, display_path=member.relative_path)
        if observed.fingerprint == member.postimage_fingerprint:
            return "postimage"
        if (
            observed.existed == member.preimage.existed
            and observed.fingerprint == member.preimage.fingerprint
        ):
            return "preimage"
        return "unknown"

    def commit(self, intent, *, recovery_action=None, recovery=False):
        self._verify_identity(intent, recovery=recovery)
        with self._lock():
            states = {member.relative_path: self._member_state(member) for member in intent.members}
            if "unknown" in states.values():
                raise TrainingPublicationRecoveryError("current output differs from publication preimage and postimage")
            for member in sorted(intent.members, key=lambda item: item.commit_order):
                if states[member.relative_path] == "postimage":
                    continue
                staged = self.directory / member.staged_relative_path
                payload = staged.read_bytes()
                if sha256_bytes(payload) != member.postimage_fingerprint:
                    raise TrainingPublicationRecoveryError("staged publication payload is corrupt")
                atomic_write_bytes(self._contained(member.relative_path), payload)
                self.fault_hook("after_member_replace", member.relative_path)
            for member in intent.members:
                if self._member_state(member) != "postimage":
                    raise TrainingPublicationRecoveryError("publication postimage verification failed")
            receipt = TrainingPublicationReceipt(
                transaction_id=intent.transaction_id, run_id=intent.run_id, status="committed",
                published_keys=intent.published_keys,
                committed_outputs=tuple({
                    "path": item.relative_path, "kind": item.kind,
                    "fingerprint": item.postimage_fingerprint,
                } for item in sorted(intent.members, key=lambda member: member.commit_order)),
                recovery_action=recovery_action,
            )
            self.fault_hook("before_receipt_write")
            atomic_write_json_bytes(self.receipt_path, _json_bytes(receipt.to_dict()))
            self.fault_hook("after_receipt_write")
            fsync_directory(self.directory)
            return receipt

    def recover(self):
        receipt = self.load_receipt()
        if receipt is not None:
            intent = self.load_intent()
            if intent is None:
                raise TrainingPublicationRecoveryError("publication receipt has no durable intent")
            self._verify_identity(intent, recovery=True)
            for item in receipt.committed_outputs:
                _raw, observed = read_with_baseline(self._contained(item["path"]), display_path=item["path"])
                if observed.fingerprint != item["fingerprint"]:
                    raise TrainingPublicationRecoveryError("committed publication output changed")
            return receipt
        intent = self.load_intent()
        if intent is None:
            return None
        states = [self._member_state(item) for item in intent.members]
        if "unknown" in states:
            raise TrainingPublicationRecoveryError("publication recovery found an unknown current output")
        action = "finalize_postimages" if all(item == "postimage" for item in states) else "roll_forward"
        self._verify_identity(intent, recovery=True)
        return self.commit(intent, recovery_action=action, recovery=True)

    @classmethod
    def find_for_run(
        cls, run, *, clock, expected_resume_fingerprint=None, transaction_id=None,
    ):
        parent = run.prepared.ctx.data_path("training_transactions")
        if not parent.is_dir():
            return None
        prefix = run.prepared.plan.run_id + "-"
        matches = sorted(
            path for path in parent.iterdir()
            if path.is_dir()
            and (path.name == transaction_id if transaction_id else path.name.startswith(prefix))
        )
        for path in reversed(matches):
            candidate = cls(run, clock=clock)
            candidate._set_transaction_id(path.name)
            candidate.recovery_resume_fingerprint = expected_resume_fingerprint
            try:
                intent = candidate.load_intent()
                if intent is not None:
                    candidate._verify_identity(intent, recovery=True)
                    candidate.transaction_id = intent.transaction_id
                    return candidate
            except TrainingPublicationRecoveryError:
                continue
        return None
