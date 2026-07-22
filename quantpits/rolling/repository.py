"""Canonical locked CAS repository for disposable Rolling State V2."""

import errno
import hashlib
import os
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from quantpits.rolling.errors import (
    RollingIdentityError,
    RollingStatePathError,
    RollingStateTransitionError,
)
from quantpits.rolling.identity import ROLLING_FAMILIES, workspace_fingerprint
from quantpits.rolling.state import (
    RollingStateExpectation,
    RollingStateInspection,
    RollingStateV2Snapshot,
    inspect_rolling_state_bytes,
    serialize_rolling_state_v2,
)
from quantpits.rolling.evidence import RollingEvidenceSetInspection, _rebuild_evidence_set
from quantpits.utils.workspace import WorkspaceContext

try:
    import fcntl
except ImportError:  # pragma: no cover - mutation fails closed on this platform
    fcntl = None


_STATE_NAMES = {
    "rolling": "rolling_state.json",
    "cpcv_rolling": "rolling_state_cpcv.json",
}
_DIGEST_CHARS = frozenset("0123456789abcdef")
_RECEIPT_STATUSES = (
    "committed", "unchanged", "deleted", "missing_noop", "conflict",
    "invalid_source", "invalid_transition", "lock_unavailable",
    "write_failed_before_replace", "durability_uncertain", "interrupted",
)
_STATE_PHASES = ("prepared", "executing", "units_complete", "failed", "completed")
_REASON_FOR_STATUS = {
    "committed": "rolling_state_committed",
    "unchanged": "rolling_state_unchanged",
    "deleted": "rolling_state_deleted",
    "missing_noop": "rolling_state_missing",
    "conflict": "rolling_state_cas_conflict",
    "invalid_source": "rolling_state_source_invalid",
    "invalid_transition": "rolling_state_transition_invalid",
    "lock_unavailable": "rolling_state_lock_conflict",
    "write_failed_before_replace": "rolling_state_write_failed",
    "durability_uncertain": "rolling_state_durability_uncertain",
    "interrupted": "rolling_state_interrupted",
}
_STATUSES_FOR_OPERATION = {
    "create": frozenset((
        "committed", "conflict", "invalid_source", "invalid_transition",
        "lock_unavailable", "write_failed_before_replace",
        "durability_uncertain", "interrupted",
    )),
    "transition": frozenset((
        "committed", "unchanged", "conflict", "invalid_source",
        "invalid_transition", "lock_unavailable",
        "write_failed_before_replace", "durability_uncertain", "interrupted",
    )),
    "delete": frozenset((
        "deleted", "missing_noop", "conflict", "invalid_source",
        "lock_unavailable", "write_failed_before_replace",
        "durability_uncertain", "interrupted",
    )),
}
_COMMITTED_PHASE_TRANSITIONS = frozenset((
    ("prepared", "executing"),
    ("prepared", "failed"),
    ("executing", "executing"),
    ("executing", "failed"),
    ("failed", "executing"),
    ("executing", "units_complete"),
))


def _is_digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _DIGEST_CHARS for char in value)
    )


@dataclass(frozen=True)
class RollingStateBaseline:
    relative_path: str
    path_kind: str
    existed: bool
    size_bytes: int = None
    fingerprint: str = None

    def __post_init__(self):
        if (not isinstance(self.relative_path, str)
                or self.relative_path not in (
                    "data/rolling_state.json",
                    "data/rolling_state_cpcv.json",
                )):
            raise RollingIdentityError("baseline path is not canonical")
        if type(self.existed) is not bool:
            raise RollingIdentityError("baseline existed must be a boolean")
        if self.existed:
            if self.path_kind != "file":
                raise RollingIdentityError("existing baseline must be a file")
            if (type(self.size_bytes) is not int or self.size_bytes < 0
                    or not _is_digest(self.fingerprint)):
                raise RollingIdentityError("existing baseline facts are invalid")
        elif (self.path_kind != "missing" or self.size_bytes is not None
              or self.fingerprint is not None):
            raise RollingIdentityError("missing baseline facts are contradictory")

    def to_public_dict(self):
        return {
            "relative_path": self.relative_path,
            "path_kind": self.path_kind,
            "existed": self.existed,
            "size_bytes": self.size_bytes,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class RollingStateRepositoryView:
    inspection: object
    baseline: RollingStateBaseline

    def __post_init__(self):
        if not isinstance(self.inspection, RollingStateInspection):
            raise RollingIdentityError("repository view requires a typed inspection")
        if not isinstance(self.baseline, RollingStateBaseline):
            raise RollingIdentityError("repository view requires a typed baseline")
        if self.inspection.path != self.baseline.relative_path:
            raise RollingIdentityError("repository view path facts disagree")
        if self.baseline.existed:
            if (self.inspection.path_kind != "file"
                    or self.inspection.fingerprint != self.baseline.fingerprint):
                raise RollingIdentityError("repository view byte facts disagree")
        elif self.inspection.classification != "missing":
            raise RollingIdentityError("missing baseline has a non-missing inspection")

    @property
    def mutation_classification(self):
        if self.inspection.classification in ("missing", "valid_versioned"):
            return self.inspection.classification
        return None


@dataclass(frozen=True)
class RollingStateMutationReceipt:
    operation: str
    status: str
    reason_code: str
    did_write: bool
    relative_path: str
    before_baseline: RollingStateBaseline = None
    after_baseline: RollingStateBaseline = None
    before_phase: str = None
    after_phase: str = None

    def __post_init__(self):
        if self.operation not in ("create", "transition", "delete"):
            raise RollingIdentityError("repository receipt operation is invalid")
        if self.status not in _RECEIPT_STATUSES:
            raise RollingIdentityError("repository receipt status is invalid")
        if self.status not in _STATUSES_FOR_OPERATION[self.operation]:
            raise RollingIdentityError("repository receipt operation contradicts status")
        if self.reason_code != _REASON_FOR_STATUS[self.status]:
            raise RollingIdentityError("repository receipt reason contradicts status")
        if type(self.did_write) is not bool:
            raise RollingIdentityError("repository receipt did_write must be boolean")
        if self.relative_path not in (
                "data/rolling_state.json", "data/rolling_state_cpcv.json"):
            raise RollingIdentityError("repository receipt path is not canonical")
        for baseline in (self.before_baseline, self.after_baseline):
            if baseline is not None and (
                    not isinstance(baseline, RollingStateBaseline)
                    or baseline.relative_path != self.relative_path):
                raise RollingIdentityError("repository receipt baseline is foreign")
        for phase in (self.before_phase, self.after_phase):
            if phase is not None and phase not in _STATE_PHASES:
                raise RollingIdentityError("repository receipt phase is invalid")
        if ((self.before_baseline is None or not self.before_baseline.existed)
                and self.before_phase is not None):
            raise RollingIdentityError("receipt preimage phase lacks existing bytes")
        if ((self.after_baseline is None or not self.after_baseline.existed)
                and self.after_phase is not None):
            raise RollingIdentityError("receipt postimage phase lacks existing bytes")
        if self.status in ("committed", "deleted"):
            if not self.did_write or self.after_baseline is None:
                raise RollingIdentityError("committed receipt lacks an observed postimage")
        if self.status in ("unchanged", "missing_noop"):
            if self.did_write or self.after_baseline is None:
                raise RollingIdentityError("no-op receipt facts are contradictory")
        if self.status in (
                "conflict", "invalid_source", "invalid_transition",
                "lock_unavailable", "write_failed_before_replace",
        ) and self.did_write:
            raise RollingIdentityError("non-commit receipt cannot claim a state write")
        if self.status == "committed" and not self.after_baseline.existed:
            raise RollingIdentityError("committed receipt postimage is missing")
        if self.status == "committed" and self.after_phase is None:
            raise RollingIdentityError("committed receipt lacks an observed phase")
        if self.status in ("deleted", "missing_noop") and self.after_baseline.existed:
            raise RollingIdentityError("delete receipt postimage still exists")
        if self.status == "committed":
            if self.before_baseline is None or self.before_baseline == self.after_baseline:
                raise RollingIdentityError("committed receipt preimage is contradictory")
            if self.operation == "create":
                if (self.before_baseline.existed or self.before_phase is not None
                        or self.after_phase != "prepared"):
                    raise RollingIdentityError("create receipt phase facts are contradictory")
            elif (not self.before_baseline.existed
                  or (self.before_phase, self.after_phase)
                  not in _COMMITTED_PHASE_TRANSITIONS):
                raise RollingIdentityError("transition receipt phase facts are contradictory")
        if self.status == "unchanged":
            if (self.before_baseline is None
                    or self.before_baseline != self.after_baseline
                    or not self.before_baseline.existed
                    or self.before_phase is None
                    or self.before_phase != self.after_phase):
                raise RollingIdentityError("unchanged receipt facts are contradictory")
        if self.status == "deleted":
            if (self.before_baseline is None or not self.before_baseline.existed
                    or self.before_phase != "failed" or self.after_phase is not None):
                raise RollingIdentityError("deleted receipt facts are contradictory")
        if self.status == "missing_noop":
            if (self.before_baseline is None
                    or self.before_baseline != self.after_baseline
                    or self.before_baseline.existed
                    or self.before_phase is not None or self.after_phase is not None):
                raise RollingIdentityError("missing no-op receipt facts are contradictory")
        if (self.status in (
                "conflict", "invalid_source", "invalid_transition",
                "lock_unavailable", "write_failed_before_replace",
        ) and self.after_baseline is not None):
            raise RollingIdentityError("non-write receipt exposes a postimage")
        if self.status == "durability_uncertain" and not self.did_write:
            raise RollingIdentityError("uncertain receipt must follow an authoritative mutation")
        if self.status == "durability_uncertain":
            if self.before_baseline is None:
                raise RollingIdentityError("uncertain receipt lacks an observed preimage")
            if self.operation == "create":
                if (self.before_baseline.existed or self.before_phase is not None
                        or self.after_phase not in (None, "prepared")):
                    raise RollingIdentityError("uncertain create facts are contradictory")
            elif self.operation == "transition":
                if (not self.before_baseline.existed
                        or self.before_phase not in ("prepared", "executing", "failed")
                        or (self.after_phase is not None
                            and (self.before_phase, self.after_phase)
                            not in _COMMITTED_PHASE_TRANSITIONS)):
                    raise RollingIdentityError("uncertain transition facts are contradictory")
            elif (not self.before_baseline.existed
                  or self.before_phase != "failed" or self.after_phase is not None):
                raise RollingIdentityError("uncertain delete facts are contradictory")
        if self.status == "interrupted":
            if not self.did_write and self.after_baseline is not None:
                raise RollingIdentityError("unwritten interrupt exposes a postimage")
            if self.did_write and self.before_baseline is None:
                raise RollingIdentityError("written interrupt lacks an observed preimage")
            if self.did_write:
                if self.operation == "create":
                    if self.before_baseline.existed:
                        raise RollingIdentityError(
                            "interrupted create preimage is contradictory"
                        )
                elif (not self.before_baseline.existed
                      or (self.operation == "transition"
                          and self.before_phase
                          not in ("prepared", "executing", "failed"))
                      or (self.operation == "delete"
                          and self.before_phase != "failed")):
                    raise RollingIdentityError(
                        "interrupted mutation preimage is contradictory"
                    )
            if self.after_phase is not None:
                if not self.did_write or self.before_baseline is None:
                    raise RollingIdentityError(
                        "interrupted postimage lacks mutation facts"
                    )
                if self.operation == "create":
                    if self.after_phase != "prepared":
                        raise RollingIdentityError(
                            "interrupted create postimage is contradictory"
                        )
                elif (self.operation == "transition"
                      and (self.before_phase, self.after_phase)
                      not in _COMMITTED_PHASE_TRANSITIONS):
                    raise RollingIdentityError(
                        "interrupted transition phases are contradictory"
                    )
                elif self.operation == "delete":
                    raise RollingIdentityError(
                        "interrupted delete postimage is contradictory"
                    )

    @property
    def cas_baseline(self):
        if self.status in ("committed", "unchanged", "deleted", "missing_noop"):
            return self.after_baseline
        return None

    def to_public_dict(self):
        return {
            "operation": self.operation,
            "status": self.status,
            "reason_code": self.reason_code,
            "did_write": self.did_write,
            "relative_path": self.relative_path,
            "before_baseline": (
                self.before_baseline.to_public_dict()
                if self.before_baseline is not None else None
            ),
            "after_baseline": (
                self.after_baseline.to_public_dict()
                if self.after_baseline is not None else None
            ),
            "before_phase": self.before_phase,
            "after_phase": self.after_phase,
        }


class RollingStateRepository:
    """One canonical filesystem truth owner for a Rolling family state."""

    def __init__(self, context, family, fault_hook=None):
        if not isinstance(context, WorkspaceContext):
            raise RollingStatePathError("repository requires WorkspaceContext")
        if family not in ROLLING_FAMILIES:
            raise RollingStatePathError("repository family is unsupported")
        if context.data_dir != context.root / "data":
            raise RollingStatePathError("workspace data path is not canonical")
        self._context = context
        self._family = family
        self._fault_hook = fault_hook

    @property
    def context(self):
        return self._context

    @property
    def family(self):
        return self._family

    @property
    def state_name(self):
        return _STATE_NAMES[self._family]

    @property
    def relative_path(self):
        return "data/%s" % self.state_name

    @property
    def state_path(self):
        return self._context.data_dir / self.state_name

    @property
    def lock_name(self):
        return self.state_name + ".lock"

    @property
    def lock_path(self):
        return self._context.data_dir / self.lock_name

    @classmethod
    def for_workspace(cls, context, family, fault_hook=None):
        return cls(context, family, fault_hook=fault_hook)

    def _fault(self, point):
        if self._fault_hook is not None:
            self._fault_hook(point)

    @staticmethod
    def _mutation_platform_supported():
        return (
            fcntl is not None
            and os.name == "posix"
            and hasattr(os, "O_NOFOLLOW")
            and hasattr(os, "O_DIRECTORY")
            and os.open in os.supports_dir_fd
            and os.unlink in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
        )

    def _missing_baseline(self):
        return RollingStateBaseline(
            relative_path=self.relative_path,
            path_kind="missing",
            existed=False,
        )

    def _baseline_for_bytes(self, data):
        return RollingStateBaseline(
            relative_path=self.relative_path,
            path_kind="file",
            existed=True,
            size_bytes=len(data),
            fingerprint=hashlib.sha256(data).hexdigest(),
        )

    @staticmethod
    def _same_baseline(left, right):
        return left == right

    def _expectation(self):
        return RollingStateExpectation(family=self.family)

    def _validate_data_parent(self, allow_missing=False):
        root = Path(self.context.root).expanduser().absolute()
        data = Path(self.context.data_dir).expanduser().absolute()
        if data != root / "data":
            raise RollingStatePathError("workspace data path is not canonical")
        if root.resolve() != root:
            raise RollingStatePathError("workspace root contains a symlink")
        try:
            root_meta = os.lstat(str(root))
        except FileNotFoundError:
            if allow_missing:
                return None
            raise RollingStatePathError("workspace root does not exist")
        except OSError as exc:
            raise RollingStatePathError("workspace root is unavailable: %s" % exc)
        if not stat.S_ISDIR(root_meta.st_mode) or stat.S_ISLNK(root_meta.st_mode):
            raise RollingStatePathError("workspace root is not a real directory")
        try:
            data_meta = os.lstat(str(data))
        except FileNotFoundError:
            if allow_missing:
                return None
            raise RollingStatePathError("workspace data directory does not exist")
        except OSError as exc:
            raise RollingStatePathError("workspace data directory is unavailable: %s" % exc)
        if not stat.S_ISDIR(data_meta.st_mode) or stat.S_ISLNK(data_meta.st_mode):
            raise RollingStatePathError("workspace data path is not a real directory")
        try:
            data.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (OSError, ValueError):
            raise RollingStatePathError("workspace data path escapes the workspace")
        flags = os.O_RDONLY
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(str(data), flags)
        except OSError as exc:
            raise RollingStatePathError("workspace data directory cannot be opened: %s" % exc)
        return descriptor

    def _verify_parent_identity(self, descriptor):
        root = os.stat(str(self.context.root), follow_symlinks=False)
        current = os.stat(str(self.context.data_dir), follow_symlinks=False)
        opened = os.fstat(descriptor)
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        parent_flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open("..", parent_flags, dir_fd=descriptor)
        try:
            opened_parent = os.fstat(parent_fd)
        finally:
            os.close(parent_fd)
        if (not stat.S_ISDIR(root.st_mode) or not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
            raise RollingStatePathError("workspace data directory identity changed")
        if (root.st_dev, root.st_ino) != (opened_parent.st_dev, opened_parent.st_ino):
            raise RollingStatePathError("workspace root identity changed")
        resolved = Path(self.context.data_dir).resolve(strict=True)
        resolved.relative_to(Path(self.context.root).resolve(strict=True))

    def _verify_lock_identity(self, directory_fd, lock_fd):
        current = os.stat(
            self.lock_name, dir_fd=directory_fd, follow_symlinks=False,
        )
        opened = os.fstat(lock_fd)
        if (not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
            raise RollingStatePathError("state lock identity changed")

    def _read_bytes(self, directory_fd):
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.state_name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RollingStatePathError("state target is not a readable regular file: %s" % exc)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise RollingStatePathError("state target is not a regular file")
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def _view_from_bytes(self, data):
        baseline = (
            self._missing_baseline() if data is None
            else self._baseline_for_bytes(data)
        )
        inspection = inspect_rolling_state_bytes(
            data,
            relative_path=self.relative_path,
            workspace_root=self.context.root,
            expectation=self._expectation(),
        )
        return RollingStateRepositoryView(inspection=inspection, baseline=baseline)

    def _read_view(self, directory_fd):
        return self._view_from_bytes(self._read_bytes(directory_fd))

    def inspect_readonly(self):
        directory_fd = self._validate_data_parent(allow_missing=True)
        if directory_fd is None:
            return self._view_from_bytes(None)
        try:
            return self._read_view(directory_fd)
        finally:
            os.close(directory_fd)

    @contextmanager
    def _locked_directory(self, blocking=True):
        if not self._mutation_platform_supported():
            yield None, None, False
            return
        directory_fd = self._validate_data_parent(allow_missing=False)
        lock_fd = None
        acquired = False
        try:
            self._verify_parent_identity(directory_fd)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            try:
                lock_fd = os.open(
                    self.lock_name, flags, 0o600, dir_fd=directory_fd,
                )
            except OSError as exc:
                raise RollingStatePathError(
                    "state lock is not a writable regular file: %s" % exc
                )
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise RollingStatePathError("state lock is not a regular file")
            lock_flags = fcntl.LOCK_EX
            if not blocking:
                lock_flags |= fcntl.LOCK_NB
            try:
                fcntl.flock(lock_fd, lock_flags)
                acquired = True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
            if acquired:
                self._verify_lock_identity(directory_fd, lock_fd)
            yield directory_fd, lock_fd, acquired
        finally:
            if acquired and lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            if lock_fd is not None:
                os.close(lock_fd)
            if directory_fd is not None:
                os.close(directory_fd)

    def _receipt(self, operation, status, before=None, after=None,
                 before_phase=None, after_phase=None, did_write=None):
        if did_write is None:
            did_write = status in ("committed", "deleted")
        return RollingStateMutationReceipt(
            operation=operation,
            status=status,
            reason_code=_REASON_FOR_STATUS[status],
            did_write=did_write,
            relative_path=self.relative_path,
            before_baseline=before,
            after_baseline=after,
            before_phase=before_phase,
            after_phase=after_phase,
        )

    @staticmethod
    def _unit_map(snapshot):
        return {
            (unit.target_key, unit.window_key): unit for unit in snapshot.units
        }

    @staticmethod
    def _validate_evidence_authorization(proposed, evidence_set):
        if not isinstance(evidence_set, RollingEvidenceSetInspection):
            raise RollingStateTransitionError("evidence authorization must be inspector-owned")
        try:
            evidence_set = _rebuild_evidence_set(evidence_set)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            raise RollingStateTransitionError("evidence authorization is not canonical")
        success_units = tuple(unit for unit in proposed.units if unit.status == "success")
        success_keys = tuple((unit.target_key, unit.window_key) for unit in success_units)
        if evidence_set.requested_unit_keys != success_keys:
            raise RollingStateTransitionError("evidence scope does not exactly match successful units")
        if evidence_set.status != "all_valid" or evidence_set.n_valid != len(success_units):
            raise RollingStateTransitionError("successful units require all-valid evidence")
        for unit, evidence in zip(success_units, evidence_set.unit_results):
            summary = dict(evidence.source_summary)
            if (
                evidence.classification != "valid"
                or evidence.evidence_fingerprint != unit.evidence_id
                or summary.get("recorder_id") != unit.record_id
                or evidence.unit_key != (unit.target_key, unit.window_key)
            ):
                raise RollingStateTransitionError("unit success facts disagree with immutable evidence")
        if proposed.phase == "units_complete" and len(success_units) != len(proposed.units):
            raise RollingStateTransitionError("units_complete contains a non-success unit")
        return evidence_set

    def _validate_transition(self, previous, proposed, evidence_set=None):
        if not isinstance(proposed, RollingStateV2Snapshot):
            raise RollingStateTransitionError("proposed state is not State V2")
        if proposed.family != self.family:
            raise RollingStateTransitionError("proposed state family is foreign")
        if proposed.workspace_fingerprint != workspace_fingerprint(self.context.root):
            raise RollingStateTransitionError("proposed state workspace is foreign")
        if proposed.phase == "completed":
            raise RollingStateTransitionError("publication completion is not writable")
        if proposed.phase == "units_complete" and evidence_set is None:
            raise RollingStateTransitionError("completion requires immutable evidence")
        if evidence_set is None and any(unit.status in ("success", "completed")
               or unit.evidence_id is not None or unit.record_id is not None
               for unit in proposed.units):
            raise RollingStateTransitionError("evidence/completion claims are not writable")
        if evidence_set is not None:
            self._validate_evidence_authorization(proposed, evidence_set)
        if previous is None:
            if (proposed.phase != "prepared" or proposed.attempt_id is not None
                    or proposed.units):
                raise RollingStateTransitionError("create must begin with empty prepared state")
            return
        identity_fields = (
            "workspace_fingerprint", "run_id", "family", "action",
            "plan_fingerprint", "execution_fingerprint", "config_fingerprint",
            "anchor_date", "target_keys", "window_keys",
        )
        if any(getattr(previous, field) != getattr(proposed, field)
               for field in identity_fields):
            raise RollingStateTransitionError("Rolling state logical identity changed")
        if previous.extensions != proposed.extensions:
            raise RollingStateTransitionError("Rolling state extensions changed")
        if (previous.phase in ("prepared", "failed")
                and proposed.phase == previous.phase
                and previous.to_public_dict() != proposed.to_public_dict()):
            raise RollingStateTransitionError(
                "%s state only supports an exact idempotent retry" % previous.phase
            )
        allowed = {
            "prepared": ("prepared", "executing", "failed"),
            "executing": ("executing", "failed", "units_complete"),
            "failed": ("failed", "executing"),
            "units_complete": ("units_complete",),
        }
        if previous.phase not in allowed or proposed.phase not in allowed[previous.phase]:
            raise RollingStateTransitionError("Rolling state phase transition is invalid")
        if previous.phase == "prepared":
            if proposed.phase == "executing" and not proposed.attempt_id:
                raise RollingStateTransitionError("execution requires an attempt")
            if proposed.phase != "executing" and proposed.attempt_id is not None:
                raise RollingStateTransitionError("prepared failure cannot invent an attempt")
        elif previous.phase == "executing":
            if proposed.attempt_id != previous.attempt_id:
                raise RollingStateTransitionError("active attempt identity changed")
        elif previous.phase == "failed" and proposed.phase == "executing":
            if not proposed.attempt_id or proposed.attempt_id == previous.attempt_id:
                raise RollingStateTransitionError("retry requires a new attempt")
        elif proposed.attempt_id != previous.attempt_id:
            raise RollingStateTransitionError("failed state attempt identity changed")
        previous_units = self._unit_map(previous)
        proposed_units = self._unit_map(proposed)
        proposed_order = [
            (unit.target_key, unit.window_key) for unit in proposed.units
        ]
        retained = [item for item in proposed_order if item in previous_units]
        if retained != [
                (unit.target_key, unit.window_key) for unit in previous.units]:
            raise RollingStateTransitionError("existing unit identity was removed or reordered")
        unit_transitions = {
            "pending": ("pending", "running", "failed", "skipped"),
            "running": ("running", "success", "failed", "skipped"),
            "failed": ("failed", "running"),
            "skipped": ("skipped",),
            "blocked": ("blocked",),
            "success": ("success",),
        }
        for identity, old in previous_units.items():
            new = proposed_units[identity]
            if new.status not in unit_transitions.get(old.status, ()):
                raise RollingStateTransitionError("unit progress regressed")
            if old.status == "success":
                if (old.record_id != new.record_id
                        or old.evidence_id != new.evidence_id
                        or old._extensions_json != new._extensions_json):
                    raise RollingStateTransitionError("successful unit facts are immutable")
            elif new.status != "success" and (
                    old.record_id != new.record_id or old.evidence_id != new.evidence_id):
                raise RollingStateTransitionError("unit evidence facts changed without success")
        for identity, unit in proposed_units.items():
            if (identity not in previous_units
                    and unit._extensions_json is not None):
                raise RollingStateTransitionError("new unit extensions are not authorized")

    def _write_temp(self, directory_fd, payload):
        temporary = ".%s.%s.tmp" % (self.state_name, secrets.token_hex(8))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        try:
            self._fault("after_temp_create")
            offset = 0
            while offset < len(payload):
                self._fault("during_temp_write")
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("short Rolling state temp write")
                offset += written
            self._fault("after_temp_write_before_fsync")
            os.fsync(descriptor)
            self._fault("after_temp_fsync")
        except BaseException:
            os.close(descriptor)
            self._cleanup_temp(directory_fd, temporary)
            raise
        else:
            os.close(descriptor)
        return temporary

    @staticmethod
    def _cleanup_temp(directory_fd, temporary):
        if temporary is None:
            return
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _commit(self, proposed, expected, blocking=True, evidence_set=None):
        payload = serialize_rolling_state_v2(proposed)
        operation = "create" if not expected.existed else "transition"
        temporary = None
        replaced = False
        before = after = None
        before_phase = None
        try:
            self._fault("before_lock")
            with self._locked_directory(blocking=blocking) as locked:
                directory_fd, lock_fd, acquired = locked
                if not acquired:
                    return self._receipt(operation, "lock_unavailable")
                self._fault("after_lock_before_source_read")
                self._verify_parent_identity(directory_fd)
                view = self._read_view(directory_fd)
                before = view.baseline
                before_phase = getattr(view.inspection.snapshot, "phase", None)
                if not self._same_baseline(before, expected):
                    return self._receipt(
                        operation, "conflict", before=before,
                        before_phase=before_phase,
                    )
                if view.inspection.classification == "missing":
                    previous = None
                elif view.inspection.classification == "valid_versioned":
                    previous = view.inspection.snapshot
                else:
                    return self._receipt(
                        operation, "invalid_source", before=before,
                        before_phase=before_phase,
                    )
                try:
                    self._validate_transition(previous, proposed, evidence_set=evidence_set)
                except RollingStateTransitionError:
                    return self._receipt(
                        operation, "invalid_transition", before=before,
                        before_phase=before_phase,
                    )
                if before.existed and before.fingerprint == hashlib.sha256(payload).hexdigest():
                    return self._receipt(
                        operation, "unchanged", before=before, after=before,
                        before_phase=before_phase, after_phase=before_phase,
                    )
                self._fault("after_source_read_before_temp_create")
                self._verify_parent_identity(directory_fd)
                self._verify_lock_identity(directory_fd, lock_fd)
                temporary = self._write_temp(directory_fd, payload)
                try:
                    current = self._read_view(directory_fd)
                    if not self._same_baseline(current.baseline, before):
                        return self._receipt(
                            operation, "conflict", before=current.baseline,
                            before_phase=getattr(
                                current.inspection.snapshot, "phase", None,
                            ),
                        )
                    self._fault("after_cas_recheck_before_replace")
                    self._verify_parent_identity(directory_fd)
                    self._verify_lock_identity(directory_fd, lock_fd)
                    os.replace(
                        temporary, self.state_name,
                        src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
                    )
                    temporary = None
                    replaced = True
                    self._fault("after_replace_before_directory_fsync")
                    os.fsync(directory_fd)
                    self._fault("after_directory_fsync_before_reread")
                    self._verify_parent_identity(directory_fd)
                    final_view = self._read_view(directory_fd)
                    after = final_view.baseline
                    if (not after.existed
                            or after.fingerprint != hashlib.sha256(payload).hexdigest()
                            or final_view.inspection.classification != "valid_versioned"):
                        return self._receipt(
                            operation, "durability_uncertain", before=before,
                            after=after, before_phase=before_phase,
                            after_phase=getattr(
                                final_view.inspection.snapshot, "phase", None,
                            ),
                            did_write=True,
                        )
                    return self._receipt(
                        operation, "committed", before=before, after=after,
                        before_phase=before_phase,
                        after_phase=getattr(
                            final_view.inspection.snapshot, "phase", None,
                        ),
                    )
                finally:
                    if temporary is not None:
                        self._cleanup_temp(directory_fd, temporary)
                        temporary = None
        except RollingStatePathError:
            return self._receipt(
                operation,
                "durability_uncertain" if replaced else "invalid_source",
                before=before, after=after, before_phase=before_phase,
                after_phase=None,
                did_write=replaced,
            )
        except OSError:
            return self._receipt(
                operation,
                "durability_uncertain" if replaced else "write_failed_before_replace",
                before=before, after=after, before_phase=before_phase,
                after_phase=None,
                did_write=replaced,
            )

    def commit(self, proposed, expected, blocking=True):
        """Commit pre-evidence state only; completion authority is rejected."""

        return self._commit(proposed, expected, blocking=blocking, evidence_set=None)

    def commit_evidence_authorized(self, proposed, expected, evidence_set, blocking=True):
        """Commit exact success facts after rebuilding inspector-owned evidence."""

        return self._commit(
            proposed, expected, blocking=blocking, evidence_set=evidence_set,
        )

    def delete(self, expected, blocking=True):
        temporary = None
        unlinked = False
        before = after = None
        before_phase = None
        try:
            self._fault("before_lock")
            with self._locked_directory(blocking=blocking) as locked:
                directory_fd, lock_fd, acquired = locked
                if not acquired:
                    return self._receipt("delete", "lock_unavailable")
                self._fault("after_lock_before_source_read")
                self._verify_parent_identity(directory_fd)
                view = self._read_view(directory_fd)
                before = view.baseline
                before_phase = getattr(view.inspection.snapshot, "phase", None)
                if not self._same_baseline(before, expected):
                    return self._receipt(
                        "delete", "conflict", before=before,
                        before_phase=before_phase,
                    )
                if view.inspection.classification == "missing":
                    return self._receipt(
                        "delete", "missing_noop", before=before, after=before,
                    )
                if (view.inspection.classification != "valid_versioned"
                        or before_phase != "failed"):
                    return self._receipt(
                        "delete", "invalid_source", before=before,
                        before_phase=before_phase,
                    )
                current = self._read_view(directory_fd)
                if not self._same_baseline(current.baseline, before):
                    return self._receipt(
                        "delete", "conflict", before=current.baseline,
                        before_phase=getattr(
                            current.inspection.snapshot, "phase", None,
                        ),
                    )
                self._fault("after_cas_recheck_before_replace")
                self._verify_parent_identity(directory_fd)
                self._verify_lock_identity(directory_fd, lock_fd)
                os.unlink(self.state_name, dir_fd=directory_fd)
                unlinked = True
                self._fault("after_replace_before_directory_fsync")
                os.fsync(directory_fd)
                self._fault("after_directory_fsync_before_reread")
                self._verify_parent_identity(directory_fd)
                final_view = self._read_view(directory_fd)
                after = final_view.baseline
                if after.existed:
                    return self._receipt(
                        "delete", "durability_uncertain", before=before,
                        after=after, before_phase=before_phase, did_write=True,
                    )
                return self._receipt(
                    "delete", "deleted", before=before, after=after,
                    before_phase=before_phase,
                )
        except RollingStatePathError:
            return self._receipt(
                "delete",
                "durability_uncertain" if unlinked else "invalid_source",
                before=before, after=after, before_phase=before_phase,
                did_write=unlinked,
            )
        except OSError:
            return self._receipt(
                "delete",
                "durability_uncertain" if unlinked else "write_failed_before_replace",
                before=before, after=after, before_phase=before_phase,
                did_write=unlinked,
            )
