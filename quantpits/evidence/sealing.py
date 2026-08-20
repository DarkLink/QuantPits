"""Inspector-owned construction and create-only publication of cycle bundles."""

from __future__ import annotations

import errno
import ctypes
import hashlib
import io
import os
import re
import secrets
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from quantpits.evidence.contracts import (
    CaptureRequest,
    CaptureResult,
    ContractError,
    DecisionEvent,
    TypedDigest,
    _RESULT_AUTHORITY,
    canonical_json_bytes,
    finite_number,
)
from quantpits.evidence.inspection import (
    FileSnapshot,
    PathBoundaryError,
    contained_path,
    git_control_specs,
    inspect_file,
    inspect_git,
    inspect_many,
    inspect_tree,
    parse_json,
    root_identity,
    SourceMutationObserver,
    strict_json_object,
)
from quantpits.evidence.ranking import RankingResult, canonical_full_ranking


SCHEMA_VERSION = 1
EMBED_LIMIT = 2 * 1024 * 1024
EXPECTED_COMMANDS = {
    "post_trade": {"post-trade", "prod_post_trade"},
    "prediction": {"static_train", "static-train"},
    "ensemble": {"ensemble_fusion", "ensemble-fusion"},
    "order": {"order_gen", "order-gen"},
}
MANIFEST_FIELDS = frozenset({
    "schema_version", "cycle_identity", "engine_identity", "workspace_identity",
    "data_identity", "run_evidence", "model_and_ensemble_lineage", "ranking",
    "portfolio_state", "decision_state", "referenced_evidence", "preservation",
    "problems", "capture_time", "status", "request_content_digest",
})
ENGINE_SURFACE_MEMBERS = (
    "quantpits/evidence/__init__.py",
    "quantpits/evidence/contracts.py",
    "quantpits/evidence/inspection.py",
    "quantpits/evidence/ranking.py",
    "quantpits/evidence/sealing.py",
    "quantpits/scripts/cycle_evidence.py",
    "quantpits/config_contracts/normalizers.py",
    "quantpits/utils/workspace.py",
)


def _result(
    cycle_id: str, status: str, did_write: bool, bundle_path: Optional[str],
    seal_digest: Optional[TypedDigest], problems: Tuple[dict, ...] = (),
    *, sealed_status: Optional[str] = None,
) -> CaptureResult:
    if sealed_status is None and status in {"sealed_complete", "sealed_partial"}:
        sealed_status = status
    return CaptureResult(
        cycle_id, status, did_write, bundle_path, seal_digest, problems,
        sealed_status, _authority=_RESULT_AUTHORITY,
    )


def _problem(code: str, evidence_class: str, detail: str, *, blocking: bool = False) -> dict:
    return {
        "code": code, "evidence_class": evidence_class,
        "detail": detail[:1000], "blocks_complete": bool(blocking),
    }


def _exception_detail(exc: BaseException) -> str:
    if isinstance(exc, (ContractError, PathBoundaryError)):
        return str(exc)[:1000]
    detail = type(exc).__name__
    error_number = getattr(exc, "errno", None)
    return "%s(errno=%s)" % (detail, error_number) if error_number is not None else detail


def _stable_file_bytes(root: Path, path: Path) -> bytes:
    """Read one public regular file through a no-follow directory chain."""
    canonical_root = root.absolute()
    root_info = os.lstat(str(canonical_root))
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise PathBoundaryError("sealed authority root is not canonical")
    try:
        relative = path.absolute().relative_to(canonical_root)
    except ValueError as exc:
        raise PathBoundaryError("sealed member escapes its authority root") from exc
    if not relative.parts:
        raise PathBoundaryError("sealed member path is empty")
    before = os.lstat(str(path))
    descriptors = []
    try:
        current_fd = os.open(
            str(canonical_root), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptors.append(current_fd)
        for part in relative.parts[:-1]:
            current_fd = os.open(
                part, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current_fd,
            )
            descriptors.append(current_fd)
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current_fd,
        )
        descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ContractError("sealed member identity changed while opening")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise PathBoundaryError(
                "sealed member parent identity became noncanonical"
            ) from exc
        raise
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    after = os.lstat(str(path))
    identities = [
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, opened, after)
    ]
    if (
        identities[0] != identities[1]
        or identities[0] != identities[2]
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise ContractError("sealed member identity changed while reading")
    return b"".join(chunks)


def _lock_continuous(
    lock: Path, lock_parent_fd: int, lock_fd: int,
    parent_identity: Tuple[int, int], lock_identity: Tuple[int, int, int, int],
) -> bool:
    """Join the inspector-owned lock descriptor to its canonical public name."""
    try:
        parent_opened = os.fstat(lock_parent_fd)
        parent_public = os.lstat(str(lock.parent))
        opened = os.fstat(lock_fd)
        public = os.stat(lock.name, dir_fd=lock_parent_fd, follow_symlinks=False)
        return (
            stat.S_ISDIR(parent_opened.st_mode)
            and stat.S_ISDIR(parent_public.st_mode)
            and not stat.S_ISLNK(parent_public.st_mode)
            and (parent_opened.st_dev, parent_opened.st_ino) == parent_identity
            and (parent_public.st_dev, parent_public.st_ino) == parent_identity
            and stat.S_ISREG(opened.st_mode)
            and stat.S_ISREG(public.st_mode)
            and opened.st_nlink == 1
            and public.st_nlink == 1
            and (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) == lock_identity
            and (public.st_dev, public.st_ino, public.st_size, public.st_mtime_ns) == lock_identity
        )
    except OSError:
        return False


def _release_owned_lock(
    lock: Path, lock_parent_fd: int, lock_fd: int,
    parent_identity: Tuple[int, int], lock_identity: Tuple[int, int, int, int],
) -> bool:
    if not _lock_continuous(
        lock, lock_parent_fd, lock_fd, parent_identity, lock_identity,
    ):
        return False
    try:
        os.unlink(lock.name, dir_fd=lock_parent_fd)
        os.fsync(lock_parent_fd)
        return True
    except OSError:
        return False


def _open_directory_no_follow(root: Path, root_fd: int, target: Path) -> int:
    """Open a contained directory through a held no-follow descriptor chain."""
    canonical_root = root.absolute()
    try:
        relative = target.absolute().relative_to(canonical_root)
    except ValueError as exc:
        raise PathBoundaryError("directory escapes its authority root") from exc
    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.dup(root_fd)
    try:
        for part in relative.parts:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise PathBoundaryError("opened authority member is not a directory")
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise PathBoundaryError(
                "directory chain contains a noncanonical alias"
            ) from exc
        raise
    except Exception:
        os.close(descriptor)
        raise


def _stage_bytes(stage_fd: int, logical_path: str, data: bytes) -> None:
    """Create a staged member through no-follow directory descriptors."""
    parts = Path(logical_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise PathBoundaryError("staged member path is not canonical")
    descriptors = []
    try:
        current_fd = os.dup(stage_fd)
        descriptors.append(current_fd)
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o700, dir_fd=current_fd)
                os.fsync(current_fd)
            except FileExistsError:
                pass
            next_fd = os.open(
                part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd,
            )
            info = os.fstat(next_fd)
            if not stat.S_ISDIR(info.st_mode):
                raise PathBoundaryError("staged member parent is not a directory")
            descriptors.append(next_fd)
            current_fd = next_fd
        flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(parts[-1], flags, 0o600, dir_fd=current_fd)
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError(errno.EIO, "short staged evidence write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(current_fd)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _directory_chain(root: Path, targets: Sequence[Path]) -> Tuple[Tuple[str, int, int], ...]:
    """Freeze canonical directory names from the workspace root to targets."""
    canonical_root = root.absolute()
    identities = []
    seen = set()
    for target in targets:
        try:
            relative = target.absolute().relative_to(canonical_root)
        except ValueError as exc:
            raise PathBoundaryError("directory identity escapes workspace") from exc
        current = canonical_root
        for part in relative.parts:
            current = current / part
            if current in seen:
                continue
            info = os.lstat(str(current))
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise PathBoundaryError("directory identity contains an alias")
            identities.append((current.relative_to(canonical_root).as_posix(), info.st_dev, info.st_ino))
            seen.add(current)
    root_info = os.lstat(str(canonical_root))
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise PathBoundaryError("workspace root public identity is invalid")
    return ((".", root_info.st_dev, root_info.st_ino),) + tuple(identities)


def _safe_mkdirs(root: Path, root_fd: int, target: Path) -> None:
    """Create a contained directory chain without following mutable aliases."""
    canonical_root = root.absolute()
    try:
        relative = target.absolute().relative_to(canonical_root)
    except ValueError as exc:
        raise PathBoundaryError("write parent is outside workspace") from exc
    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.dup(root_fd)
    try:
        for part in relative.parts:
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    # A concurrent creator gets no authority: the member is
                    # still opened and validated through this held parent.
                    pass
                os.fsync(descriptor)
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            info = os.fstat(next_descriptor)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_descriptor)
                raise PathBoundaryError("write parent contains a non-directory alias")
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise PathBoundaryError(
                "write parent contains a non-directory alias"
            ) from exc
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _rename_noreplace(
    source_parent_fd: int, source_name: str,
    target_parent_fd: int, target_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic create-only directory publish is unsupported")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent_fd, os.fsencode(source_name),
        target_parent_fd, os.fsencode(target_name), 1,
    )
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), target_name)


def _relative_existing(root: Path, value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        canonical_root = root.resolve(strict=True)
        try:
            relative = candidate.relative_to(canonical_root).as_posix()
        except ValueError as exc:
            raise PathBoundaryError("referenced evidence escapes workspace") from exc
        return contained_path(root, relative).relative_to(canonical_root).as_posix()
    return contained_path(root, value).relative_to(root.resolve()).as_posix()


def _embedded_manifest_digests(value: Any) -> Tuple[str, ...]:
    found = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            evidence_fields = {"path", "status", "digest", "preservation_status", "detail"}
            if evidence_fields.issubset(item) and item.get("preservation_status") == "embedded":
                digest = item.get("digest")
                if not isinstance(digest, dict):
                    raise ContractError("embedded evidence lacks a typed digest")
                found.add(TypedDigest(**digest).value)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return tuple(sorted(found))


def _named_manifest_digests(manifest: Mapping[str, Any]) -> Dict[str, dict]:
    result = {}
    ranking = manifest.get("ranking")
    if isinstance(ranking, Mapping) and ranking.get("ranking_digest") is not None:
        digest = TypedDigest(**ranking["ranking_digest"])
        if digest.domain != "raw_bytes":
            raise ContractError("ranking named evidence digest has the wrong domain")
        result["ranking.csv"] = digest.to_dict()
    portfolio = manifest.get("portfolio_state")
    if isinstance(portfolio, Mapping) and portfolio.get("canonical_digest") is not None:
        digest = TypedDigest(**portfolio["canonical_digest"])
        if digest.domain != "semantic_config":
            raise ContractError("portfolio named evidence digest has the wrong domain")
        result["portfolio_state.json"] = TypedDigest(
            digest.algorithm, "raw_bytes", digest.value, digest.size_bytes,
        ).to_dict()
    return result


def _extract_anchor(manifest: Mapping[str, Any]) -> Optional[str]:
    records = manifest.get("records", {})
    if not isinstance(records, Mapping):
        return None
    for name in ("expected_anchor", "anchor_date", "cycle_id"):
        value = records.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _manifest_refs(manifest: Mapping[str, Any]) -> Tuple[str, ...]:
    refs = []
    for collection in ("inputs", "outputs"):
        values = manifest.get(collection, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                path = item["path"]
                if "<" not in path and path not in refs:
                    refs.append(path)
    return tuple(refs)


def _universe_from_file(data: bytes, anchor: Optional[str]) -> Tuple[str, ...]:
    if not isinstance(anchor, str):
        raise ContractError("eligible universe requires an exact anchor")
    try:
        datetime.strptime(anchor, "%Y-%m-%d")
    except ValueError as exc:
        raise ContractError("eligible universe anchor must be YYYY-MM-DD") from exc
    text = data.decode("utf-8")
    members = []
    declared = set()
    for raw in text.splitlines():
        fields = raw.strip().replace(",", "\t").split()
        if not fields:
            continue
        if len(fields) != 3 or not fields[0] or fields[0] in declared:
            raise ContractError("eligible universe rows must have unique exact identities")
        try:
            start = datetime.strptime(fields[1], "%Y-%m-%d").date()
            end = datetime.strptime(fields[2], "%Y-%m-%d").date()
        except ValueError as exc:
            raise ContractError("eligible universe dates must be YYYY-MM-DD") from exc
        if start > end:
            raise ContractError("eligible universe interval is inverted")
        declared.add(fields[0])
        if fields[1] <= anchor <= fields[2]:
            members.append(fields[0])
    return tuple(members)


def _selected_combo(records: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    combos = records.get("combos")
    if not isinstance(combos, list) or not combos:
        return None
    defaults = [item for item in combos if isinstance(item, Mapping) and item.get("is_default") is True]
    selected = defaults if defaults else [item for item in combos if isinstance(item, Mapping)]
    return selected[0] if len(selected) == 1 else None


def _selected_model_evidence(
    records: Mapping[str, Any], resolved_members: Sequence[str],
) -> Tuple[Mapping[str, Any], ...]:
    if (
        not isinstance(resolved_members, (list, tuple))
        or any(not isinstance(item, str) or not item for item in resolved_members)
        or len(set(resolved_members)) != len(resolved_members)
    ):
        raise ContractError("M3 resolved model identity is invalid")
    raw = records.get("input_models", [])
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise ContractError("M3 source model inventory is invalid")
    by_key = {}
    for item in raw:
        key = item.get("resolved_key")
        if not isinstance(key, str) or not key:
            raise ContractError("M3 source model identity is invalid")
        if key in by_key:
            raise ContractError("M3 source model inventory has duplicate identity")
        by_key[key] = item
    if set(by_key).intersection(resolved_members) != set(resolved_members):
        raise ContractError("M3 source model inventory does not cover exact combo members")
    return tuple(by_key[key] for key in resolved_members)


@dataclass
class _BundleDraft:
    manifest: dict
    named_files: Dict[str, bytes]
    objects: Dict[str, bytes]
    problems: list
    blocked: bool = False
    source_observations: Optional[Dict[str, FileSnapshot]] = None
    continuity_observations: Optional[Dict[Tuple[Path, str], FileSnapshot]] = None
    tree_observations: Optional[Dict[Tuple[Path, str], FileSnapshot]] = None
    mutation_observers: Tuple[SourceMutationObserver, ...] = ()


class ProductionCycleEvidenceSealer:
    """The sole truth owner for Phase 37A evidence and seal capability."""

    def __init__(
        self, workspace_root: Path, *, engine_root: Optional[Path] = None,
        qlib_data_dir: Optional[Path] = None,
        clock: Optional[Callable[[], datetime]] = None,
        fault_hook: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.root = Path(workspace_root).resolve(strict=True)
        self._root_public_identity = root_identity(self.root)
        self.engine_root = (engine_root or Path(__file__).resolve().parents[2]).resolve(strict=True)
        configured_qlib = (
            qlib_data_dir
            if qlib_data_dir is not None
            else os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data/cn_data")
        )
        self.qlib_data_dir = Path(configured_qlib).expanduser().resolve()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.fault_hook = fault_hook or (lambda _point: None)

    def _track_snapshot(
        self, draft: _BundleDraft, snapshot: FileSnapshot, *, origin_root: Optional[Path] = None,
    ) -> None:
        observations = draft.continuity_observations
        if observations is None:
            observations = {}
            draft.continuity_observations = observations
        key = (origin_root or self.root, snapshot.logical_path)
        if (origin_root or self.root) == self.root and draft.mutation_observers:
            draft.mutation_observers[0].add_paths((snapshot.logical_path,))
        previous = observations.get(key)
        if previous is not None and not self._same_source_observation(previous, snapshot):
            draft.problems.append(_problem(
                "observation_continuity_lost", "capture",
                "%s changed between evidence observations" % snapshot.logical_path,
                blocking=True,
            ))
            return
        observations[key] = snapshot

    def _tree_snapshot(self, root: Path, logical_path: str) -> FileSnapshot:
        try:
            base = contained_path(root, logical_path)
            info = os.lstat(str(base))
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                return FileSnapshot(
                    logical_path, "incomparable", None, None, None,
                    "artifact tree root is not a canonical directory",
                )
            members = inspect_tree(root, logical_path)
            if not members or any(item.status != "observed" or item.digest is None for item in members):
                return FileSnapshot(
                    logical_path, "incomparable", None, None,
                    (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
                    "artifact tree is empty or incomparable",
                )
            inventory = [
                {"path": item.logical_path, "digest": item.digest.to_dict()}
                for item in members
            ]
            return FileSnapshot(
                logical_path, "observed", None,
                TypedDigest.canonical(inventory, "file_inventory"),
                (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except PathBoundaryError:
            raise
        except FileNotFoundError:
            return FileSnapshot(logical_path, "missing", None, None, None, "artifact tree is missing")
        except Exception as exc:
            return FileSnapshot(logical_path, "incomparable", None, None, None, _exception_detail(exc))

    def _track_tree(self, draft: _BundleDraft, logical_path: str, *, origin_root: Optional[Path] = None) -> None:
        root = origin_root or self.root
        if root == self.root and draft.mutation_observers:
            draft.mutation_observers[0].add_paths((logical_path,))
        observations = draft.tree_observations
        if observations is None:
            observations = {}
            draft.tree_observations = observations
        key = (root, logical_path)
        current = self._tree_snapshot(root, logical_path)
        previous = observations.get(key)
        if previous is not None and not self._same_source_observation(previous, current):
            draft.problems.append(_problem(
                "tree_observation_continuity_lost", "capture",
                "%s changed between tree observations" % logical_path,
                blocking=True,
            ))
            return
        observations[key] = current

    def _embed(
        self, draft: _BundleDraft, snapshot: FileSnapshot, *,
        origin_root: Optional[Path] = None, track: bool = True,
    ) -> dict:
        if track:
            self._track_snapshot(draft, snapshot, origin_root=origin_root)
        if snapshot.status != "observed" or snapshot.digest is None:
            return snapshot.to_public_dict("missing" if snapshot.status == "missing" else "incomparable")
        digest = snapshot.digest.value
        if snapshot.data is not None and len(snapshot.data) <= EMBED_LIMIT:
            draft.objects.setdefault(digest, snapshot.data)
            preservation = "embedded"
        else:
            preservation = "workspace_file"
        return snapshot.to_public_dict(preservation)

    def _engine_surface(self, draft: _BundleDraft) -> Tuple[dict, Dict[str, TypedDigest]]:
        members = ENGINE_SURFACE_MEMBERS
        snapshots = inspect_many(self.engine_root, tuple((path, path) for path in members))
        public = []
        digests = {}
        for path, snapshot in snapshots:
            public.append(self._embed(draft, snapshot, origin_root=self.engine_root))
            if snapshot.status != "observed" or snapshot.digest is None:
                draft.problems.append(_problem("engine_surface_incomparable", "engine", path, blocking=True))
            else:
                digests[path] = snapshot.digest
        return {
            "members": public,
            "surface_digest": TypedDigest.canonical([
                {"path": path, "digest": digest.to_dict()}
                for path, digest in sorted(digests.items())
            ], "file_inventory").to_dict(),
        }, digests

    def _observe_sources(self, request: CaptureRequest, draft: _BundleDraft) -> Tuple[dict, Dict[str, FileSnapshot]]:
        requested = [(name, path) for name, path, _required in request.source_paths() if path]
        observations = inspect_many(self.root, requested)
        by_name = {name: snapshot for name, snapshot in observations}
        manifests = {}
        public = []
        for name, path, _required in request.source_paths():
            if not path:
                if name == "deep_analysis":
                    draft.problems.append(_problem("deep_analysis_missing", name, "no Deep Analysis run was provided", blocking=True))
                continue
            snapshot = by_name[name]
            public.append({"class": name, **self._embed(draft, snapshot)})
            parsed = parse_json(snapshot) if name != "deep_analysis" else None
            if name == "deep_analysis":
                tree = ()
                if snapshot.status != "missing":
                    tree = inspect_tree(self.root, path)
                public[-1]["members"] = [self._embed(draft, item) for item in tree]
                if not tree or any(item.status != "observed" for item in tree):
                    draft.problems.append(_problem("deep_analysis_incomplete", name, "trace tree is absent or incomparable", blocking=True))
                else:
                    inventory = [
                        {"path": item.logical_path, "digest": item.digest.to_dict()}
                        for item in tree if item.digest is not None
                    ]
                    inventory_data = canonical_json_bytes(inventory)
                    aggregate = FileSnapshot(
                        path, "observed", inventory_data,
                        TypedDigest.canonical(inventory, "file_inventory"),
                        self._tree_snapshot(self.root, path).identity,
                    )
                    by_name[name] = aggregate
                    members = public[-1]["members"]
                    public[-1] = {
                        "class": name, **self._embed(draft, aggregate, track=False),
                        "members": members,
                    }
            elif name == "decision":
                # Decision is validated separately; malformed input remains visible.
                pass
            elif parsed is None:
                draft.problems.append(_problem("manifest_invalid", name, "manifest is missing or invalid JSON", blocking=True))
            else:
                manifests[name] = parsed
                if parsed.get("status") != "success":
                    draft.problems.append(_problem("run_not_success", name, "manifest status is not success", blocking=True))
                if parsed.get("command") not in EXPECTED_COMMANDS.get(name, set()):
                    draft.problems.append(_problem("manifest_command_mismatch", name, "manifest command does not match its evidence class", blocking=True))
                if not isinstance(parsed.get("run_id"), str) or not parsed.get("run_id"):
                    draft.problems.append(_problem("manifest_run_id_missing", name, "manifest has no exact run ID", blocking=True))
        draft.manifest["run_evidence"] = public
        draft.source_observations = dict(by_name)
        return manifests, by_name

    def _source_snapshot(self, name: str, path: str) -> FileSnapshot:
        if name != "deep_analysis":
            return inspect_file(self.root, path)
        try:
            target = contained_path(self.root, path)
        except FileNotFoundError:
            return inspect_file(self.root, path)
        if not target.is_dir():
            return inspect_file(self.root, path)
        return self._tree_snapshot(self.root, path)

    @staticmethod
    def _same_source_observation(before: FileSnapshot, after: FileSnapshot) -> bool:
        return (
            before.logical_path == after.logical_path
            and before.status == after.status
            and before.digest == after.digest
            and before.identity == after.identity
            and before.detail == after.detail
        )

    def _decision(self, request: CaptureRequest, snapshot: Optional[FileSnapshot], draft: _BundleDraft) -> dict:
        if not request.decision_event:
            return {"status": "not_recorded", "as_of_capture": True}
        raw = parse_json(snapshot) if snapshot else None
        try:
            event = DecisionEvent.from_mapping(raw)  # type: ignore[arg-type]
            if event.evidence_cycle_id != request.cycle_id:
                raise ContractError("decision event belongs to a foreign cycle")
            canonical = event.to_dict()
            return {
                "status": "recorded", "event": canonical,
                "raw_digest": snapshot.digest.to_dict() if snapshot and snapshot.digest else None,
                "canonical_digest": TypedDigest.canonical(canonical).to_dict(),
            }
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            detail = _exception_detail(exc)
            draft.problems.append(_problem("decision_invalid", "decision", detail, blocking=True))
            return {"status": "invalid", "detail": detail}

    def _portfolio(self, draft: _BundleDraft) -> dict:
        snapshot = inspect_file(self.root, "config/prod_config.json")
        public = self._embed(draft, snapshot)
        if snapshot.status != "observed" or snapshot.data is None:
            draft.problems.append(_problem("portfolio_missing", "portfolio", snapshot.detail, blocking=True))
            return public
        try:
            raw = strict_json_object(snapshot.data)
            if raw is None:
                raise ContractError("prod_config is not strict JSON")
            cash = raw.get("current_cash")
            holdings = raw.get("current_holding")
            if not isinstance(holdings, list) or not finite_number(cash):
                raise ContractError("prod_config lacks canonical cash/holding state")
            instruments = []
            for holding in holdings:
                if not isinstance(holding, dict):
                    raise ContractError("portfolio holding is not an object")
                instrument = holding.get("instrument")
                if not isinstance(instrument, str) or not instrument:
                    raise ContractError("portfolio holding has no exact instrument")
                for field in ("value", "amount"):
                    if field not in holding or not finite_number(holding[field]) or holding[field] < 0:
                        raise ContractError("portfolio holding amount/value is invalid")
                instruments.append(instrument)
            if len(set(instruments)) != len(instruments):
                raise ContractError("portfolio holding identities are duplicated")
            canonical = {"current_cash": cash, "current_holding": holdings}
            data = canonical_json_bytes(canonical)
            draft.named_files["portfolio_state.json"] = data
            public["canonical_digest"] = TypedDigest.canonical(canonical, "semantic_config").to_dict()
            public["holding_count"] = len(holdings)
            return public
        except Exception as exc:
            draft.problems.append(_problem("portfolio_invalid", "portfolio", _exception_detail(exc), blocking=True))
            return public

    def _frozen_market(self, ensemble: Optional[Mapping[str, Any]], draft: _BundleDraft) -> Optional[str]:
        if not ensemble:
            return None
        expected = None
        fingerprints = ensemble.get("config_fingerprints", {})
        if isinstance(fingerprints, Mapping):
            expected = fingerprints.get("config/model_config.json", fingerprints.get("model_config"))
        if expected is None:
            for item in ensemble.get("inputs", []) if isinstance(ensemble.get("inputs"), list) else []:
                if isinstance(item, Mapping) and item.get("path") == "config/model_config.json":
                    expected = item.get("fingerprint")
                    break
        if not isinstance(expected, str):
            draft.problems.append(_problem(
                "market_config_fingerprint_missing", "data",
                "M3 does not bind an exact model config fingerprint",
                blocking=True,
            ))
            return None
        snapshot = inspect_file(self.root, "config/model_config.json")
        self._track_snapshot(draft, snapshot)
        if snapshot.data is None:
            draft.problems.append(_problem("market_config_missing", "data", "frozen model config is unavailable", blocking=True))
            return None
        try:
            from quantpits.config_contracts.normalizers import normalize_model_config
            from quantpits.utils.workspace import fingerprint_value

            raw = strict_json_object(snapshot.data)
            if raw is None:
                raise ContractError("model config is not strict JSON")
            normalized = normalize_model_config(raw)
            if fingerprint_value(normalized) != expected:
                raise ContractError("model config differs from M3 frozen fingerprint")
            market = normalized.get("market")
            if not isinstance(market, str) or not market:
                raise ContractError("frozen model config has no market")
            return market.lower()
        except Exception as exc:
            draft.problems.append(_problem("market_config_incomparable", "data", _exception_detail(exc), blocking=True))
            return None

    def _data_identity(
        self, anchor: Optional[str], market: Optional[str], draft: _BundleDraft,
    ) -> Tuple[dict, Optional[FileSnapshot]]:
        qlib = self.qlib_data_dir
        result = {
            "source_dolt_identity": {"status": "missing"},
            "source_to_materialization_relation": "unverified",
        }
        if not qlib.is_dir():
            result["qlib_materialization_identity"] = {"status": "missing"}
            draft.problems.append(_problem("qlib_identity_missing", "data", "Qlib materialization was not configured", blocking=True))
            return result, None
        # Qlib is a separate read-only authority and is intentionally not forced
        # under the private workspace. Only logical component names are emitted.
        calendar_logical = "calendars/day.txt"
        try:
            calendar_snapshot = inspect_file(qlib, calendar_logical)
            self._track_snapshot(draft, calendar_snapshot, origin_root=qlib)
            if calendar_snapshot.status != "observed" or calendar_snapshot.data is None:
                raise ContractError("calendar is missing, too large, or incomparable")
            calendar_data = calendar_snapshot.data
            calendars = [line.strip() for line in calendar_data.decode().splitlines() if line.strip()]
            try:
                for item in calendars:
                    datetime.strptime(item, "%Y-%m-%d")
            except ValueError as exc:
                raise ContractError("Qlib calendar members must be YYYY-MM-DD") from exc
            if calendars != sorted(set(calendars)):
                raise ContractError("Qlib calendar must be ordered and unique")
            instruments_dir = contained_path(qlib, "instruments")
            if not instruments_dir.is_dir():
                raise ContractError("instrument inventory is unavailable")
            instrument_files = sorted(instruments_dir.glob("*.txt"))
            matching = [path for path in instrument_files if market and path.stem.lower() == market]
            universe_path = matching[0] if len(matching) == 1 else None
            universe_snapshot = None
            if universe_path:
                universe_logical = "instruments/%s" % universe_path.name
                observed_universe = inspect_file(qlib, universe_logical)
                self._track_snapshot(draft, observed_universe, origin_root=qlib)
                if observed_universe.status != "observed" or observed_universe.data is None:
                    raise ContractError("universe is missing, too large, or incomparable")
                universe_snapshot = FileSnapshot(
                    "qlib/%s" % universe_logical, observed_universe.status,
                    observed_universe.data, observed_universe.digest,
                    observed_universe.identity, observed_universe.detail,
                )
            result["qlib_materialization_identity"] = {
                "status": "observed",
                "calendar_cutoff": calendars[-1][:10] if calendars else None,
                "calendar_digest": calendar_snapshot.digest.to_dict(),
                "universe_digest": universe_snapshot.digest.to_dict() if universe_snapshot else None,
                "universe_name": universe_path.stem if universe_path else None,
            }
            if not calendars or anchor not in set(calendars):
                draft.problems.append(_problem("calendar_anchor_missing", "data", "cycle anchor is absent from Qlib calendar", blocking=True))
            if universe_snapshot is None:
                draft.problems.append(_problem("universe_ambiguous", "data", "exact eligible universe file is not unique", blocking=True))
            return result, universe_snapshot
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            detail = _exception_detail(exc)
            result["qlib_materialization_identity"] = {"status": "incomparable", "detail": detail}
            draft.problems.append(_problem("qlib_identity_incomparable", "data", detail, blocking=True))
            return result, None

    def _ranking(
        self, ensemble: Optional[Mapping[str, Any]], anchor: Optional[str],
        universe_snapshot: Optional[FileSnapshot], draft: _BundleDraft,
    ) -> dict:
        if not ensemble:
            draft.problems.append(_problem("ranking_source_missing", "ranking", "M3 manifest is invalid", blocking=True))
            return {"status": "unavailable"}
        records = ensemble.get("records", {})
        if not isinstance(records, Mapping):
            records = {}
        combo = _selected_combo(records)
        if combo is None:
            draft.problems.append(_problem("ranking_combo_ambiguous", "ranking", "M3 does not select one exact combo", blocking=True))
            return {"status": "unavailable"}
        resolved_members = combo.get("resolved_models", combo.get("models", []))
        if (
            not isinstance(resolved_members, list)
            or not resolved_members
            or any(not isinstance(item, str) or not item for item in resolved_members)
            or len(set(resolved_members)) != len(resolved_members)
            or not isinstance(combo.get("method"), str)
            or not combo.get("method")
        ):
            draft.problems.append(_problem("ranking_combo_invalid", "ranking", "M3 combo definition is not canonical", blocking=True))
            return {"status": "unavailable"}
        pred_ref = combo.get("recorder_id")
        scores = None
        prediction_digest = None
        try:
            source_models = _selected_model_evidence(records, resolved_members)
            if isinstance(pred_ref, str) and pred_ref:
                import pandas as pd

                output_evidence = combo.get("output_evidence", {})
                if not isinstance(output_evidence, Mapping):
                    raise ContractError("M3 output recorder evidence is absent")
                if (
                    output_evidence.get("contained") is not True
                    or output_evidence.get("recorder_id") != pred_ref
                ):
                    raise ContractError("M3 output recorder identity is inconsistent")
                artifact_path = output_evidence.get("artifact_path")
                if not isinstance(artifact_path, str):
                    raise ContractError("M3 output artifact path is absent")
                relative = _relative_existing(self.root, artifact_path)
                pred_relative = (Path(relative) / "pred.pkl").as_posix()
                snapshot = inspect_file(self.root, pred_relative)
                self._track_snapshot(draft, snapshot)
                if snapshot.status != "observed":
                    raise ContractError("M3 output pred.pkl is incomparable")
                prediction_data = _stable_file_bytes(
                    self.root, contained_path(self.root, pred_relative),
                )
                if TypedDigest.raw(prediction_data) != snapshot.digest:
                    raise ContractError("M3 output pred.pkl changed before parsing")
                prediction = pd.read_pickle(io.BytesIO(prediction_data))
                if getattr(prediction, "name", None) == "score":
                    prediction = prediction.to_frame("score")
                if not hasattr(prediction, "index") or "score" not in prediction:
                    raise ContractError("M3 output pred.pkl schema is invalid")
                if not anchor or "datetime" not in prediction.index.names or "instrument" not in prediction.index.names:
                    raise ContractError("M3 output pred.pkl lacks exact datetime/instrument identity")
                dates = prediction.index.get_level_values("datetime")
                selected = [str(value)[:10] == anchor[:10] for value in dates]
                prediction = prediction[selected]
                if prediction.empty:
                    raise ContractError("M3 output pred.pkl has no exact anchor rows")
                prediction = prediction.droplevel([name for name in prediction.index.names if name != "instrument"])
                if prediction.index.has_duplicates:
                    raise ContractError("M3 output pred.pkl has duplicate instrument rows")
                scores = {key: value for key, value in prediction["score"].items()}
                prediction_digest = snapshot.digest.to_dict() if snapshot.digest else None
            else:
                raise ContractError("M3 combo lacks an exact recorder reference")
            if universe_snapshot is None or universe_snapshot.data is None:
                raise ContractError("eligible universe is not provable")
            universe = _universe_from_file(universe_snapshot.data, anchor)
            result: RankingResult = canonical_full_ranking(universe, scores)
            ranking_bytes = result.to_csv_bytes()
            draft.named_files["ranking.csv"] = ranking_bytes
            ranking_digest = TypedDigest.raw(ranking_bytes)
            if not result.complete:
                draft.problems.append(_problem("ranking_coverage_partial", "ranking", "eligible members lack finite predictions", blocking=True))
            return {
                "status": "complete" if result.complete else "partial",
                "eligible_count": result.eligible_count,
                "scored_count": result.scored_count,
                "missing_count": result.missing_count,
                "prediction_digest": prediction_digest,
                "ranking_digest": ranking_digest.to_dict(),
                "combo": {
                    "name": combo.get("name"), "method": combo.get("method"),
                    "resolved_members": resolved_members,
                    "ensemble_recorder_id": pred_ref,
                    "output_evidence": combo.get("output_evidence", {}),
                },
                "source_models": list(source_models),
            }
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except PathBoundaryError as exc:
            draft.blocked = True
            detail = _exception_detail(exc)
            draft.problems.append(_problem(
                "ranking_path_boundary", "ranking", detail, blocking=True,
            ))
            return {"status": "unavailable", "detail": detail}
        except Exception as exc:
            detail = _exception_detail(exc)
            draft.problems.append(_problem("ranking_unavailable", "ranking", detail, blocking=True))
            return {"status": "unavailable", "detail": detail}

    def _referenced_evidence(self, manifests: Mapping[str, Mapping[str, Any]], draft: _BundleDraft) -> list:
        observed = []
        seen = set()
        for evidence_class, manifest in manifests.items():
            for ref in _manifest_refs(manifest):
                try:
                    relative = _relative_existing(self.root, ref)
                except FileNotFoundError:
                    relative = ref if not Path(ref).is_absolute() else "<unavailable>"
                    draft.problems.append(_problem("referenced_file_missing", evidence_class, "referenced output is missing", blocking=True))
                    continue
                except PathBoundaryError as exc:
                    draft.problems.append(_problem("referenced_path_escape", evidence_class, str(exc), blocking=True))
                    draft.blocked = True
                    continue
                if relative in seen:
                    continue
                seen.add(relative)
                if draft.mutation_observers:
                    draft.mutation_observers[0].add_paths((relative,))
                snapshot = inspect_file(self.root, relative)
                observed.append(self._embed(draft, snapshot))
                if snapshot.status != "observed":
                    draft.problems.append(_problem("referenced_file_incomparable", evidence_class, relative, blocking=True))
        return observed

    def _lineage_artifacts(
        self, ensemble: Optional[Mapping[str, Any]], anchor: Optional[str],
        draft: _BundleDraft,
    ) -> list:
        if not ensemble or not isinstance(ensemble.get("records"), Mapping):
            return []
        records = ensemble["records"]
        observed = []
        combo = _selected_combo(records)
        resolved = combo.get("resolved_models", combo.get("models", [])) if combo else []
        try:
            models = _selected_model_evidence(records, resolved)
        except ContractError as exc:
            draft.problems.append(_problem("model_lineage_missing", "model", _exception_detail(exc), blocking=True))
            return observed
        if not models:
            draft.problems.append(_problem("model_lineage_missing", "model", "M3 has no source model evidence", blocking=True))
            return observed
        for position, model in enumerate(models):
            if not isinstance(model, Mapping):
                draft.problems.append(_problem("model_lineage_invalid", "model", "source model evidence is not an object", blocking=True))
                continue
            artifact_path = model.get("artifact_path")
            recorder_id = model.get("recorder_id")
            identity_valid = (
                isinstance(recorder_id, str) and bool(recorder_id)
                and isinstance(artifact_path, str)
                and model.get("status") == "ready"
                and isinstance(model.get("experiment_name"), str)
                and bool(model.get("experiment_name"))
                and isinstance(model.get("experiment_id"), str)
                and bool(model.get("experiment_id"))
                and model.get("prediction_end") == anchor
            )
            if not identity_valid:
                draft.problems.append(_problem("model_lineage_invalid", "model", "source recorder/artifact identity is absent", blocking=True))
                continue
            try:
                relative = _relative_existing(self.root, artifact_path)
                self._track_tree(draft, relative)
                members = inspect_tree(self.root, relative)
                public_members = [self._embed(draft, item) for item in members]
                if not members or any(item.status != "observed" for item in members):
                    raise ContractError("model artifact tree is incomparable")
                observed.append({
                    "position": position, "recorder_id": recorder_id,
                    "source_recorder_id": model.get("source_recorder_id"),
                    "artifact_locator": relative, "members": public_members,
                    "artifact_tree_digest": TypedDigest.canonical([
                        {"path": item.logical_path, "digest": item.digest.to_dict()}
                        for item in members if item.digest is not None
                    ], "file_inventory").to_dict(),
                })
                source_recorder_id = model.get("source_recorder_id")
                if source_recorder_id:
                    if not isinstance(source_recorder_id, str) or not model.get("source_experiment_name"):
                        raise ContractError("source training recorder identity is invalid")
                    source_path = model.get("source_artifact_path")
                    if source_recorder_id == recorder_id and not isinstance(source_path, str):
                        source_path = artifact_path
                    if not isinstance(source_path, str):
                        candidates = []
                        for base_name in ("mlruns", "mlartifacts"):
                            base = self.root / base_name
                            if base.is_dir() and not base.is_symlink():
                                candidates.extend(base.glob("*/%s/artifacts" % source_recorder_id))
                        contained = [
                            path for path in candidates
                            if path.is_dir() and not path.is_symlink()
                        ]
                        if len(contained) != 1:
                            raise ContractError("source training artifact path is not uniquely observable")
                        source_path = contained[0].relative_to(self.root).as_posix()
                    source_relative = _relative_existing(self.root, source_path)
                    self._track_tree(draft, source_relative)
                    source_members = inspect_tree(self.root, source_relative)
                    if not source_members or any(item.status != "observed" for item in source_members):
                        raise ContractError("source training artifact tree is incomparable")
                    observed.append({
                        "position": position, "role": "source_training",
                        "recorder_id": source_recorder_id,
                        "experiment_name": model.get("source_experiment_name"),
                        "artifact_locator": source_relative,
                        "members": [self._embed(draft, item) for item in source_members],
                        "artifact_tree_digest": TypedDigest.canonical([
                            {"path": item.logical_path, "digest": item.digest.to_dict()}
                            for item in source_members if item.digest is not None
                        ], "file_inventory").to_dict(),
                    })
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except PathBoundaryError as exc:
                draft.blocked = True
                draft.problems.append(_problem(
                    "model_artifact_path_boundary", "model",
                    _exception_detail(exc), blocking=True,
                ))
            except Exception as exc:
                draft.problems.append(_problem("model_artifact_incomparable", "model", _exception_detail(exc), blocking=True))
        output = combo.get("output_evidence", {}) if combo else {}
        if isinstance(output, Mapping) and isinstance(output.get("artifact_path"), str):
            try:
                relative = _relative_existing(self.root, output["artifact_path"])
                self._track_tree(draft, relative)
                members = inspect_tree(self.root, relative)
                if not members or any(item.status != "observed" for item in members):
                    raise ContractError("ensemble artifact tree is incomparable")
                observed.append({
                    "position": "ensemble", "recorder_id": combo.get("recorder_id"),
                    "artifact_locator": relative,
                    "members": [self._embed(draft, item) for item in members],
                    "artifact_tree_digest": TypedDigest.canonical([
                        {"path": item.logical_path, "digest": item.digest.to_dict()}
                        for item in members if item.digest is not None
                    ], "file_inventory").to_dict(),
                })
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except PathBoundaryError as exc:
                draft.blocked = True
                draft.problems.append(_problem(
                    "ensemble_artifact_path_boundary", "ensemble",
                    _exception_detail(exc), blocking=True,
                ))
            except Exception as exc:
                draft.problems.append(_problem("ensemble_artifact_incomparable", "ensemble", _exception_detail(exc), blocking=True))
        return observed

    def _build(
        self, request: CaptureRequest, observer: SourceMutationObserver,
        data_observer: Optional[SourceMutationObserver],
        engine_observer: SourceMutationObserver,
        workspace_git_observers: Tuple[SourceMutationObserver, ...],
        engine_git_observers: Tuple[SourceMutationObserver, ...],
        final_observer: Optional[SourceMutationObserver],
    ) -> _BundleDraft:
        draft = _BundleDraft(
            {}, {}, {}, [], source_observations={}, continuity_observations={},
            tree_observations={}, mutation_observers=tuple(
                item for item in (
                    observer, engine_observer, data_observer,
                    final_observer,
                    *workspace_git_observers, *engine_git_observers,
                )
                if item is not None
            ),
        )
        original_root = root_identity(self.root)
        manifests, snapshots = self._observe_sources(request, draft)
        anchored_classes = ("prediction", "ensemble", "order")
        anchors = {
            name: _extract_anchor(manifests[name])
            for name in anchored_classes if name in manifests
        }
        for name in anchored_classes:
            if name in manifests and anchors.get(name) is None:
                draft.problems.append(_problem("manifest_anchor_missing", name, "manifest has no exact cycle anchor", blocking=True))
        ensemble_records = manifests.get("ensemble", {}).get("records", {})
        if isinstance(ensemble_records, Mapping):
            expected_anchor = ensemble_records.get("expected_anchor")
            actual_anchor = ensemble_records.get("anchor_date")
            if expected_anchor != actual_anchor:
                draft.problems.append(_problem("ensemble_anchor_mismatch", "ensemble", "expected and actual ensemble anchors differ", blocking=True))
        comparable = {value for value in anchors.values() if value}
        if len(comparable) > 1:
            draft.problems.append(_problem("anchor_mismatch", "cycle", "run manifests disagree on cycle anchor", blocking=True))
        anchor = _extract_anchor(manifests.get("ensemble", {})) or (next(iter(comparable)) if comparable else None)
        engine_git = inspect_git(self.engine_root)
        workspace_git = inspect_git(self.root)
        engine_surface, engine_surface_digests = self._engine_surface(draft)
        if engine_git.get("status") == "dirty_unresolved":
            draft.problems.append(_problem("engine_git_unresolved", "engine", "engine Git identity is unresolved", blocking=True))
        if workspace_git.get("status") == "dirty_unresolved":
            draft.problems.append(_problem("workspace_git_unresolved", "workspace", "workspace Git identity is unresolved", blocking=True))
        if any(not item.supported or item.mutated() for item in workspace_git_observers):
            draft.problems.append(_problem(
                "workspace_git_mutation_observed", "workspace",
                "workspace Git control identity changed during capture",
                blocking=True,
            ))
        if any(not item.supported or item.mutated() for item in engine_git_observers):
            draft.problems.append(_problem(
                "engine_git_control_mutation_observed", "engine",
                "engine Git control identity changed during capture",
                blocking=True,
            ))
        market = self._frozen_market(manifests.get("ensemble"), draft)
        data_identity, universe = self._data_identity(anchor, market, draft)
        ranking = self._ranking(manifests.get("ensemble"), anchor, universe, draft)
        lineage_artifacts = self._lineage_artifacts(manifests.get("ensemble"), anchor, draft)
        portfolio = self._portfolio(draft)
        decision = self._decision(request, snapshots.get("decision"), draft)
        referenced = self._referenced_evidence(manifests, draft)
        engine_git_after = inspect_git(self.engine_root)
        workspace_git_after = inspect_git(self.root)
        git_keys = ("commit", "tree", "status", "status_inventory_digest", "tracked_diff_digest")
        if any(engine_git.get(key) != engine_git_after.get(key) for key in git_keys):
            draft.problems.append(_problem("engine_git_mutated", "engine", "engine Git facts changed during capture", blocking=True))
        if any(workspace_git.get(key) != workspace_git_after.get(key) for key in git_keys):
            draft.problems.append(_problem("workspace_git_mutated", "workspace", "workspace Git facts changed during capture", blocking=True))
        for path, expected_digest in engine_surface_digests.items():
            current = inspect_file(self.engine_root, path)
            if current.status != "observed" or current.digest != expected_digest:
                draft.problems.append(_problem("engine_surface_mutated", "engine", path, blocking=True))
        if root_identity(self.root) != original_root:
            draft.blocked = True
            draft.problems.append(_problem("workspace_root_drift", "workspace", "workspace root identity changed", blocking=True))
        if not observer.supported:
            draft.problems.append(_problem("source_observer_unavailable", "capture", "transient source mutation observer is unavailable", blocking=True))
        if observer.mutated():
            draft.problems.append(_problem("source_mutation_observed", "capture", "source namespace changed during observation", blocking=True))
        if not engine_observer.supported or engine_observer.mutated():
            draft.problems.append(_problem(
                "engine_mutation_observed", "engine",
                "engine surface continuity is unavailable or changed",
                blocking=True,
            ))
        if final_observer is not None and (
            not final_observer.supported or final_observer.mutated()
        ):
            draft.blocked = True
            draft.problems.append(_problem(
                "existing_final_mutation_observed", "publication",
                "existing final namespace continuity is unavailable or changed",
                blocking=True,
            ))
        if data_observer is not None and (not data_observer.supported or data_observer.mutated()):
            draft.problems.append(_problem("data_mutation_observed", "data", "Qlib source continuity is unavailable or changed", blocking=True))
        # Re-observe every explicit source. Digest disagreement is fail-closed.
        for name, path, _required in request.source_paths():
            if not path:
                continue
            previous = snapshots.get(name)
            current = self._source_snapshot(name, path)
            if previous is None or not self._same_source_observation(previous, current):
                draft.problems.append(_problem("source_continuity_lost", name, "source changed during capture", blocking=True))
        core = {
            "schema_version": SCHEMA_VERSION,
            "cycle_identity": {
                "cycle_id": request.cycle_id,
                "research_epoch_id": request.research_epoch_id,
                "evidence_as_of": anchor,
            },
            "engine_identity": {**engine_git, "executable_surface": engine_surface},
            "workspace_identity": workspace_git,
            "data_identity": data_identity,
            "run_evidence": draft.manifest.get("run_evidence", []),
            "model_and_ensemble_lineage": {
                **ranking, "source_artifacts": lineage_artifacts,
            },
            "ranking": ranking,
            "portfolio_state": portfolio,
            "decision_state": decision,
            "referenced_evidence": referenced,
            "preservation": {
                "embedded_object_count": len(draft.objects),
                "named_file_count": len(draft.named_files),
            },
            "problems": draft.problems,
        }
        # Publication itself changes the workspace Git inventory.  Replay
        # identity therefore joins every observed cycle fact except that
        # self-referential repository inventory; its original observation is
        # still sealed in the manifest.
        replay_core = {key: value for key, value in core.items() if key != "workspace_identity"}
        content_digest = TypedDigest.canonical(replay_core)
        complete = not any(item["blocks_complete"] for item in draft.problems)
        captured_at = self.clock()
        if (
            not isinstance(captured_at, datetime)
            or captured_at.tzinfo is None
            or captured_at.utcoffset() is None
        ):
            raise ContractError("capture clock must return a timezone-aware datetime")
        core["capture_time"] = captured_at.isoformat()
        core["status"] = "sealed_complete" if complete else "sealed_partial"
        core["request_content_digest"] = content_digest.to_dict()
        draft.manifest = core
        return draft

    def _sources_continuous(self, request: CaptureRequest, draft: _BundleDraft) -> bool:
        initial = draft.source_observations or {}
        for name, path, _required in request.source_paths():
            if not path:
                continue
            previous = initial.get(name)
            current = self._source_snapshot(name, path)
            if previous is None or not self._same_source_observation(previous, current):
                return False
        return True

    def _observations_continuous(self, draft: _BundleDraft) -> bool:
        if any(item.mutated() for item in draft.mutation_observers):
            return False
        for (origin_root, logical_path), previous in (
            draft.continuity_observations or {}
        ).items():
            current = inspect_file(origin_root, logical_path)
            if not self._same_source_observation(previous, current):
                return False
        for (origin_root, logical_path), previous in (
            draft.tree_observations or {}
        ).items():
            current = self._tree_snapshot(origin_root, logical_path)
            if not self._same_source_observation(previous, current):
                return False
        return True

    def _adoption_sources_continuous(
        self, request: CaptureRequest, draft: _BundleDraft,
        observer: SourceMutationObserver,
        data_observer: Optional[SourceMutationObserver],
    ) -> bool:
        return (
            not observer.mutated()
            and (data_observer is None or not data_observer.mutated())
            and self._sources_continuous(request, draft)
            and self._observations_continuous(draft)
        )

    def _existing(self, final: Path, cycle_id: str, request_digest: TypedDigest) -> Optional[CaptureResult]:
        if not final.exists():
            return None
        try:
            relative_final = final.absolute().relative_to(self.root.absolute()).as_posix()
            final = contained_path(self.root, relative_final)
            final_before = os.lstat(str(final))
            if os.path.islink(str(final)) or not final.is_dir():
                raise ContractError("existing final public name is not a canonical directory")
            manifest_path = final / "manifest.json"
            seal_path = final / "seal.json"
            for public_file in (manifest_path, seal_path):
                info = os.lstat(str(public_file))
                if public_file.is_symlink() or not public_file.is_file() or info.st_nlink != 1:
                    raise ContractError("existing seal member is not a canonical regular file")
            manifest_data = _stable_file_bytes(self.root, manifest_path)
            seal_data = _stable_file_bytes(self.root, seal_path)
            manifest = strict_json_object(manifest_data)
            seal = strict_json_object(seal_data)
            expected_seal_fields = {
                "schema_version", "cycle_id", "status", "manifest_digest",
                "artifact_root_digest", "object_digests", "named_file_digests",
            }
            if (
                not isinstance(manifest, dict)
                or set(manifest) != MANIFEST_FIELDS
                or not isinstance(seal, dict)
                or set(seal) != expected_seal_fields
                or manifest_data != canonical_json_bytes(manifest)
                or seal_data != canonical_json_bytes(seal)
            ):
                raise ContractError("existing seal representation is invalid")
            if (
                seal.get("schema_version") != SCHEMA_VERSION
                or seal.get("cycle_id") != cycle_id
                or manifest.get("cycle_identity", {}).get("cycle_id") != cycle_id
                or seal.get("status") != manifest.get("status")
            ):
                raise ContractError("existing cycle/seal identity is inconsistent")
            declared_request_digest = TypedDigest(**manifest["request_content_digest"])
            manifest_replay_core = {
                key: value for key, value in manifest.items()
                if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
            }
            if (
                declared_request_digest != TypedDigest.canonical(manifest_replay_core)
                or declared_request_digest != request_digest
            ):
                return _result(cycle_id, "conflict", False, None, None)
            problems = manifest.get("problems")
            if (
                not isinstance(problems, list)
                or any(
                    not isinstance(item, dict)
                    or set(item) != {"code", "evidence_class", "detail", "blocks_complete"}
                    or not isinstance(item.get("blocks_complete"), bool)
                    for item in problems
                )
            ):
                raise ContractError("existing problem inventory is invalid")
            derived_status = (
                "sealed_partial" if any(item["blocks_complete"] for item in problems)
                else "sealed_complete"
            )
            if manifest.get("status") != derived_status:
                raise ContractError("existing status is not derived from sealed problems")
            manifest_digest = TypedDigest(**seal["manifest_digest"])
            if manifest_digest != TypedDigest.raw(manifest_data):
                raise ContractError("existing manifest digest is invalid")
            object_digests = seal["object_digests"]
            named_digests = seal["named_file_digests"]
            if (
                not isinstance(object_digests, list)
                or object_digests != sorted(set(object_digests))
                or any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in object_digests)
                or not isinstance(named_digests, dict)
                or any(name not in {"ranking.csv", "portfolio_state.json"} for name in named_digests)
            ):
                raise ContractError("existing artifact inventory is invalid")
            expected_objects = list(_embedded_manifest_digests(manifest))
            expected_named = _named_manifest_digests(manifest)
            preservation = manifest.get("preservation")
            if (
                not isinstance(preservation, dict)
                or set(preservation) != {"embedded_object_count", "named_file_count"}
                or isinstance(preservation.get("embedded_object_count"), bool)
                or not isinstance(preservation.get("embedded_object_count"), int)
                or isinstance(preservation.get("named_file_count"), bool)
                or not isinstance(preservation.get("named_file_count"), int)
                or preservation["embedded_object_count"] != len(expected_objects)
                or preservation["named_file_count"] != len(expected_named)
                or object_digests != expected_objects
                or named_digests != expected_named
            ):
                raise ContractError("existing artifact inventory is not joined to its manifest")
            objects_root = final / "objects"
            actual_objects = []
            actual_object_dirs = []
            if objects_root.exists():
                if objects_root.is_symlink() or not objects_root.is_dir():
                    raise ContractError("existing object root is not canonical")
                for path in objects_root.rglob("*"):
                    if path.is_symlink():
                        raise ContractError("existing object inventory contains a symlink")
                    relative = path.relative_to(objects_root)
                    if path.is_dir():
                        if len(relative.parts) != 1:
                            raise ContractError("existing object directory layout is invalid")
                        actual_object_dirs.append(relative.as_posix())
                    elif path.is_file():
                        if len(relative.parts) != 2 or path.stat().st_nlink != 1 or path.parent.name != path.name[:2]:
                            raise ContractError("existing object public name is invalid")
                        actual_objects.append(path.name)
                    else:
                        raise ContractError("existing object inventory contains a special node")
            if sorted(actual_objects) != object_digests:
                raise ContractError("existing object inventory cardinality differs")
            if sorted(actual_object_dirs) != sorted(set(digest[:2] for digest in object_digests)):
                raise ContractError("existing object directory inventory differs")
            for digest in object_digests:
                path = final / "objects" / digest[:2] / digest
                if hashlib.sha256(_stable_file_bytes(self.root, path)).hexdigest() != digest:
                    raise ContractError("existing embedded object is invalid")
            expected_top = {"manifest.json", "seal.json", *named_digests}
            if object_digests:
                expected_top.add("objects")
            actual_top = {path.name for path in final.iterdir()}
            if actual_top != expected_top or any(path.is_symlink() for path in final.iterdir()):
                raise ContractError("existing bundle has an unsealed top-level member")
            for name, digest in named_digests.items():
                typed = TypedDigest(**digest)
                path = final / name
                if not path.is_file() or path.stat().st_nlink != 1 or TypedDigest.raw(_stable_file_bytes(self.root, path)) != typed:
                    raise ContractError("existing named evidence is invalid")
            artifact_root = TypedDigest.canonical({
                "objects": object_digests, "named_files": named_digests,
            })
            if TypedDigest(**seal["artifact_root_digest"]) != artifact_root:
                raise ContractError("existing artifact root digest is invalid")
            final_after = os.lstat(str(final))
            if (final_before.st_dev, final_before.st_ino) != (final_after.st_dev, final_after.st_ino):
                raise ContractError("existing final identity changed during verification")
            status = manifest.get("status")
            if status not in {"sealed_complete", "sealed_partial"}:
                raise ContractError("existing bundle status is invalid")
            return _result(
                manifest["cycle_identity"]["cycle_id"], "adopted", False,
                final.relative_to(self.root).as_posix(), TypedDigest.raw(seal_data),
                tuple(manifest.get("problems", [])),
                sealed_status=status,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception:
            return _result(cycle_id, "conflict", False, None, None)

    def capture(self, request: CaptureRequest, *, dry_run: bool = False) -> CaptureResult:
        if not isinstance(request, CaptureRequest):
            raise ContractError("capture requires a typed request")
        if not isinstance(dry_run, bool):
            raise ContractError("dry_run must be boolean")
        paths = [path for _name, path, _required in request.source_paths() if path]
        paths.extend(("config/prod_config.json", "config/model_config.json"))
        final_logical = "data/evidence/v1/cycles/%s" % request.cycle_id
        try:
            configured = self.qlib_data_dir
            with ExitStack() as stack:
                observer = stack.enter_context(SourceMutationObserver(self.root, paths))
                engine_observer = stack.enter_context(SourceMutationObserver(
                    self.engine_root, ENGINE_SURFACE_MEMBERS,
                ))
                workspace_git_observers = tuple(
                    stack.enter_context(SourceMutationObserver(root, members))
                    for root, members in git_control_specs(self.root)
                )
                engine_git_observers = tuple(
                    stack.enter_context(SourceMutationObserver(root, members))
                    for root, members in git_control_specs(self.engine_root)
                )
                data_observer = None
                if configured.is_dir():
                    data_observer = stack.enter_context(SourceMutationObserver(
                        configured, ("calendars/day.txt", "instruments"),
                    ))
                final_observer = None
                if os.path.lexists(str(self.root / final_logical)):
                    final_observer = stack.enter_context(SourceMutationObserver(
                        self.root, (final_logical,),
                    ))
                return self._capture(
                    request, dry_run=dry_run, observer=observer,
                    data_observer=data_observer, engine_observer=engine_observer,
                    workspace_git_observers=workspace_git_observers,
                    engine_git_observers=engine_git_observers,
                    final_observer=final_observer,
                )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except PathBoundaryError as exc:
            return _result(request.cycle_id, "blocked", False, None, None, (_problem("path_boundary", "workspace", str(exc), blocking=True),))
        except Exception as exc:
            return _result(
                request.cycle_id, "blocked", False, None, None,
                (_problem(
                    "observer_failed", "capture", _exception_detail(exc),
                    blocking=True,
                ),),
            )

    def _capture(
        self, request: CaptureRequest, *, dry_run: bool,
        observer: SourceMutationObserver,
        data_observer: Optional[SourceMutationObserver],
        engine_observer: SourceMutationObserver,
        workspace_git_observers: Tuple[SourceMutationObserver, ...],
        engine_git_observers: Tuple[SourceMutationObserver, ...],
        final_observer: Optional[SourceMutationObserver],
    ) -> CaptureResult:
        root_before = root_identity(self.root)
        if root_before != self._root_public_identity:
            return _result(
                request.cycle_id, "blocked", False, None, None,
                (_problem(
                    "workspace_root_drift", "workspace",
                    "workspace root public identity changed before capture",
                    blocking=True,
                ),),
            )
        try:
            draft = self._build(
                request, observer, data_observer, engine_observer,
                workspace_git_observers, engine_git_observers, final_observer,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except PathBoundaryError as exc:
            return _result(request.cycle_id, "blocked", False, None, None, (_problem("path_boundary", "workspace", str(exc), blocking=True),))
        except Exception as exc:
            return _result(request.cycle_id, "blocked", False, None, None, (_problem("inspection_failed", "capture", _exception_detail(exc), blocking=True),))
        if draft.blocked:
            return _result(request.cycle_id, "blocked", False, None, None, tuple(draft.problems))
        request_digest = TypedDigest.canonical({
            key: value for key, value in draft.manifest.items()
            if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
        })
        # _build computed this before adding time/status; enforce the internal join.
        if draft.manifest["request_content_digest"] != request_digest.to_dict():
            raise ContractError("inspector content digest join failed")
        evidence_root = self.root / "data" / "evidence" / "v1"
        final = evidence_root / "cycles" / request.cycle_id
        if os.path.lexists(str(final)):
            contained_path(
                self.root, final.relative_to(self.root).as_posix(),
            )
        if final.exists() and not dry_run:
            final_before = os.lstat(str(final))
            final_identity = (final_before.st_dev, final_before.st_ino)
            adoption_directories = _directory_chain(self.root, (final.parent,))
            existing = self._existing(final, request.cycle_id, request_digest)
            if existing is not None and existing.status == "adopted":
                sources_continuous = self._adoption_sources_continuous(
                    request, draft, observer, data_observer,
                )
                confirmed = (
                    self._existing(final, request.cycle_id, request_digest)
                    if sources_continuous else None
                )
                try:
                    final_after = os.lstat(str(final))
                    final_continuous = (
                        stat.S_ISDIR(final_after.st_mode)
                        and not stat.S_ISLNK(final_after.st_mode)
                        and (final_after.st_dev, final_after.st_ino) == final_identity
                    )
                except OSError:
                    final_continuous = False
                if (
                    final_continuous
                    and final_observer is not None
                    and final_observer.supported
                    and not final_observer.mutated()
                    and root_identity(self.root) == root_before
                    and _directory_chain(self.root, (final.parent,)) == adoption_directories
                    and confirmed is not None
                    and confirmed.status == "adopted"
                    and confirmed.seal_digest == existing.seal_digest
                    and confirmed.sealed_status == existing.sealed_status
                    and self._adoption_sources_continuous(
                        request, draft, observer, data_observer,
                    )
                ):
                    return confirmed
                return _result(
                    request.cycle_id, "blocked", False, None, None,
                    tuple(draft.problems + [_problem(
                        "adoption_continuity_lost", "capture",
                        "source identity changed while verifying existing evidence",
                        blocking=True,
                    )]),
                )
            if existing is not None:
                return existing
            return _result(request.cycle_id, "conflict", False, None, None, tuple(draft.problems))
        object_digests = sorted(draft.objects)
        named_digests = {
            name: TypedDigest.raw(data).to_dict()
            for name, data in sorted(draft.named_files.items())
        }
        artifact_root = TypedDigest.canonical({
            "objects": object_digests, "named_files": named_digests,
        })
        manifest_data = canonical_json_bytes(draft.manifest)
        manifest_digest = TypedDigest.raw(manifest_data)
        seal = {
            "schema_version": SCHEMA_VERSION,
            "cycle_id": request.cycle_id,
            "status": draft.manifest["status"],
            "manifest_digest": manifest_digest.to_dict(),
            "artifact_root_digest": artifact_root.to_dict(),
            "object_digests": object_digests,
            "named_file_digests": named_digests,
        }
        seal_data = canonical_json_bytes(seal)
        if dry_run:
            return _result(
                request.cycle_id,
                "preview_complete" if draft.manifest["status"] == "sealed_complete" else "preview_partial",
                False, None, TypedDigest.raw(seal_data),
                tuple(draft.problems),
            )
        lock = evidence_root / ".locks" / (request.cycle_id + ".lock")
        stage = None
        stage_identity = None
        staging_parent_identity = None
        lock_fd = None
        lock_parent_fd = None
        final_parent_fd = None
        staging_parent_fd = None
        stage_fd = None
        authority_root_fd = None
        lock_parent_identity = None
        lock_identity = None
        lock_owned = False
        publication_directories = None
        wrote_staging = False
        published = False
        try:
            authority_root_fd = os.open(
                str(self.root),
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            authority_root_info = os.fstat(authority_root_fd)
            if (
                not stat.S_ISDIR(authority_root_info.st_mode)
                or (authority_root_info.st_dev, authority_root_info.st_ino) != root_before
            ):
                raise PathBoundaryError(
                    "workspace root identity changed before publication writes"
                )
            self.fault_hook("before_write_parent_creation")
            _safe_mkdirs(self.root, authority_root_fd, lock.parent)
            _safe_mkdirs(self.root, authority_root_fd, evidence_root / "cycles")
            _safe_mkdirs(self.root, authority_root_fd, evidence_root / ".staging")
            staging_parent = evidence_root / ".staging"
            publication_directories = _directory_chain(
                self.root, (lock.parent, final.parent, staging_parent),
            )
            final_parent_fd = _open_directory_no_follow(
                self.root, authority_root_fd, final.parent,
            )
            staging_parent_fd = _open_directory_no_follow(
                self.root, authority_root_fd, staging_parent,
            )
            final_parent_info = os.fstat(final_parent_fd)
            staging_parent_info = os.fstat(staging_parent_fd)
            parent_identity = (final_parent_info.st_dev, final_parent_info.st_ino)
            staging_parent_identity = (
                staging_parent_info.st_dev, staging_parent_info.st_ino,
            )
            if (
                root_identity(final.parent) != parent_identity
                or root_identity(staging_parent) != staging_parent_identity
                or _directory_chain(
                    self.root, (lock.parent, final.parent, staging_parent),
                ) != publication_directories
            ):
                raise PathBoundaryError("publication parent changed while opening")
            lock_parent_fd = _open_directory_no_follow(
                self.root, authority_root_fd, lock.parent,
            )
            parent_info = os.fstat(lock_parent_fd)
            lock_parent_identity = (parent_info.st_dev, parent_info.st_ino)
            lock_fd = os.open(lock.name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600, dir_fd=lock_parent_fd)
            lock_owned = True
            os.write(lock_fd, request_digest.value.encode("ascii"))
            os.fsync(lock_fd)
            os.fsync(lock_parent_fd)
            lock_info = os.fstat(lock_fd)
            lock_identity = (lock_info.st_dev, lock_info.st_ino, lock_info.st_size, lock_info.st_mtime_ns)
            self.fault_hook("after_lock")
            if not _lock_continuous(lock, lock_parent_fd, lock_fd, lock_parent_identity, lock_identity):
                return _result(
                    request.cycle_id, "blocked", False, None, None,
                    tuple(draft.problems + [_problem(
                        "lock_continuity_lost", "publication",
                        "evidence lock identity changed after acquisition", blocking=True,
                    )]),
                )
            if final.exists():
                if final_observer is None:
                    final_observer = SourceMutationObserver(
                        self.root, (final.relative_to(self.root).as_posix(),),
                    )
                if not final_observer.supported:
                    return _result(
                        request.cycle_id, "blocked", False, None, None,
                        tuple(draft.problems + [_problem(
                            "existing_final_observer_unavailable", "publication",
                            "concurrent final namespace continuity is unavailable",
                            blocking=True,
                        )]),
                    )
                final_before = os.lstat(str(final))
                final_identity = (final_before.st_dev, final_before.st_ino)
                existing = self._existing(final, request.cycle_id, request_digest)
                if existing is not None and existing.status == "adopted":
                    sources_continuous = self._adoption_sources_continuous(
                        request, draft, observer, data_observer,
                    )
                    confirmed = (
                        self._existing(final, request.cycle_id, request_digest)
                        if sources_continuous else None
                    )
                    try:
                        final_after = os.lstat(str(final))
                        final_continuous = (
                            stat.S_ISDIR(final_after.st_mode)
                            and not stat.S_ISLNK(final_after.st_mode)
                            and (final_after.st_dev, final_after.st_ino) == final_identity
                        )
                    except OSError:
                        final_continuous = False
                    if (
                        final_continuous
                        and not final_observer.mutated()
                        and root_identity(self.root) == root_before
                        and _directory_chain(
                            self.root, (lock.parent, final.parent, staging_parent),
                        ) == publication_directories
                        and confirmed is not None
                        and confirmed.status == "adopted"
                        and confirmed.seal_digest == existing.seal_digest
                        and confirmed.sealed_status == existing.sealed_status
                        and _lock_continuous(
                            lock, lock_parent_fd, lock_fd,
                            lock_parent_identity, lock_identity,
                        )
                        and self._adoption_sources_continuous(
                            request, draft, observer, data_observer,
                        )
                    ):
                        if _release_owned_lock(
                            lock, lock_parent_fd, lock_fd,
                            lock_parent_identity, lock_identity,
                        ):
                            lock_owned = False
                            return confirmed
                        return _result(
                            request.cycle_id, "blocked", False, None, None,
                            tuple(draft.problems + [_problem(
                                "lock_cleanup_failed", "publication",
                                "evidence lock could not be removed after adoption",
                                blocking=True,
                            )]),
                        )
                    return _result(
                        request.cycle_id, "blocked", False, None, None,
                        tuple(draft.problems + [_problem(
                            "adoption_continuity_lost", "capture",
                            "source or lock identity changed while adopting evidence",
                            blocking=True,
                        )]),
                    )
                if existing is not None:
                    return existing
                return _result(request.cycle_id, "conflict", False, None, None, tuple(draft.problems))
            if root_identity(self.root) != root_before:
                return _result(request.cycle_id, "blocked", False, None, None, tuple(draft.problems))
            for _attempt in range(128):
                stage_name = request.cycle_id + "." + secrets.token_hex(12)
                try:
                    os.mkdir(stage_name, 0o700, dir_fd=staging_parent_fd)
                    break
                except FileExistsError:
                    continue
            else:
                raise OSError(errno.EEXIST, "unable to allocate unique staging name")
            os.fsync(staging_parent_fd)
            stage = staging_parent / stage_name
            stage_fd = os.open(
                stage_name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=staging_parent_fd,
            )
            stage_info = os.fstat(stage_fd)
            stage_identity = (stage_info.st_dev, stage_info.st_ino)
            wrote_staging = True
            if (
                not stat.S_ISDIR(stage_info.st_mode)
                or stat.S_ISLNK(stage_info.st_mode)
                or stage_info.st_dev != parent_identity[0]
                or root_identity(staging_parent) != staging_parent_identity
            ):
                raise PathBoundaryError("staging directory identity is invalid")
            self.fault_hook("after_stage_created")
            for digest, data in draft.objects.items():
                _stage_bytes(stage_fd, "objects/%s/%s" % (digest[:2], digest), data)
            for name, data in draft.named_files.items():
                _stage_bytes(stage_fd, name, data)
            _stage_bytes(stage_fd, "manifest.json", manifest_data)
            _stage_bytes(stage_fd, "seal.json", seal_data)
            os.fsync(stage_fd)
            self.fault_hook("before_publish")
            if (
                observer.mutated()
                or (data_observer is not None and data_observer.mutated())
                or not self._sources_continuous(request, draft)
                or not self._observations_continuous(draft)
            ):
                return _result(
                    request.cycle_id, "failed_no_final", True, None, None,
                    tuple(draft.problems + [_problem(
                        "source_continuity_lost", "capture",
                        "source changed before namespace publication", blocking=True,
                    )]),
                )
            if _directory_chain(
                self.root, (lock.parent, final.parent, staging_parent),
            ) != publication_directories:
                return _result(
                    request.cycle_id, "failed_no_final", True, None, None,
                    tuple(draft.problems + [_problem(
                        "publish_ancestor_continuity_lost", "publication",
                        "publication directory identity changed before namespace publication",
                        blocking=True,
                    )]),
                )
            if not _lock_continuous(lock, lock_parent_fd, lock_fd, lock_parent_identity, lock_identity):
                return _result(
                    request.cycle_id, "failed_no_final", True, None, None,
                    tuple(draft.problems + [_problem(
                        "lock_continuity_lost", "publication",
                        "evidence lock identity changed before namespace publication", blocking=True,
                    )]),
                )
            try:
                stage_public = os.lstat(str(stage))
                stage_continuous = (
                    stat.S_ISDIR(stage_public.st_mode)
                    and not stat.S_ISLNK(stage_public.st_mode)
                    and stage_identity == (stage_public.st_dev, stage_public.st_ino)
                    and root_identity(stage.parent) == staging_parent_identity
                    and stage_public.st_dev == parent_identity[0]
                )
            except OSError:
                stage_continuous = False
            if not stage_continuous:
                return _result(
                    request.cycle_id, "failed_no_final", True, None, None,
                    tuple(draft.problems + [_problem(
                        "staging_continuity_lost", "publication",
                        "staging public identity changed before namespace publication",
                        blocking=True,
                    )]),
                )
            staged = self._existing(stage, request.cycle_id, request_digest)
            if (
                staged is None or staged.status != "adopted"
                or staged.seal_digest != TypedDigest.raw(seal_data)
                or staged.sealed_status != draft.manifest["status"]
            ):
                return _result(
                    request.cycle_id, "failed_no_final", True, None, None,
                    tuple(draft.problems + [_problem(
                        "staging_verification_failed", "publication",
                        "staged evidence does not match its canonical seal",
                        blocking=True,
                    )]),
                )
            if root_identity(self.root) != root_before or root_identity(final.parent) != parent_identity:
                return _result(
                    request.cycle_id, "failed_no_final", True, None, None,
                    tuple(draft.problems + [_problem(
                        "publish_parent_continuity_lost", "publication",
                        "workspace or final parent changed before namespace publication",
                        blocking=True,
                    )]),
                )
            _rename_noreplace(
                staging_parent_fd, stage.name,
                final_parent_fd, final.name,
            )
            published = True
            stage = None
            if final_observer is None:
                final_observer = SourceMutationObserver(
                    self.root, (final.relative_to(self.root).as_posix(),),
                )
            if not final_observer.supported:
                return _result(
                    request.cycle_id, "uncertain", True, None, None,
                    tuple(draft.problems + [_problem(
                        "final_observer_unavailable", "publication",
                        "post-publish final namespace continuity is unavailable",
                        blocking=True,
                    )]),
                )
            os.fsync(final_parent_fd)
            self.fault_hook("after_publish")
            try:
                final_info = os.lstat(str(final))
                final_continuous = (
                    not os.path.islink(str(final))
                    and os.path.isdir(str(final))
                    and stage_identity == (final_info.st_dev, final_info.st_ino)
                )
            except OSError:
                final_continuous = False
            if (
                not final_continuous
                or root_identity(self.root) != root_before
                or root_identity(final.parent) != parent_identity
                or not _lock_continuous(lock, lock_parent_fd, lock_fd, lock_parent_identity, lock_identity)
                or _directory_chain(
                    self.root, (lock.parent, final.parent, staging_parent),
                ) != publication_directories
            ):
                return _result(
                    request.cycle_id, "uncertain", True, None, None,
                    tuple(draft.problems + [_problem(
                        "post_publish_continuity_lost", "publication",
                        "final, root, parent, or lock identity changed after publication",
                        blocking=True,
                    )]),
                )
            adopted = self._existing(final, request.cycle_id, request_digest)
            if (
                adopted is None
                or adopted.status != "adopted"
                or adopted.cycle_id != request.cycle_id
                or adopted.seal_digest != TypedDigest.raw(seal_data)
                or adopted.sealed_status != draft.manifest["status"]
            ):
                return _result(
                    request.cycle_id, "uncertain", True, None, None,
                    tuple(draft.problems + [_problem(
                        "post_publish_verification_failed", "publication",
                        "canonical final bundle did not verify against the staged seal",
                        blocking=True,
                    )]),
                )
            final_info_after = os.lstat(str(final))
            if (
                stage_identity != (final_info_after.st_dev, final_info_after.st_ino)
                or root_identity(self.root) != root_before
                or root_identity(final.parent) != parent_identity
                or not _lock_continuous(lock, lock_parent_fd, lock_fd, lock_parent_identity, lock_identity)
                or observer.mutated()
                or (data_observer is not None and data_observer.mutated())
                or not self._sources_continuous(request, draft)
                or not self._observations_continuous(draft)
                or _directory_chain(
                    self.root, (lock.parent, final.parent, staging_parent),
                ) != publication_directories
            ):
                return _result(
                    request.cycle_id, "uncertain", True, None, None,
                    tuple(draft.problems + [_problem(
                        "post_publish_source_continuity_lost", "capture",
                        "source, final, root, parent, or lock continuity was lost",
                        blocking=True,
                    )]),
                )
            confirmed = self._existing(final, request.cycle_id, request_digest)
            try:
                final_confirmed = os.lstat(str(final))
                confirmation_continuous = (
                    stage_identity == (final_confirmed.st_dev, final_confirmed.st_ino)
                    and confirmed is not None
                    and confirmed.status == "adopted"
                    and confirmed.seal_digest == adopted.seal_digest
                    and confirmed.sealed_status == adopted.sealed_status
                    and not final_observer.mutated()
                    and not observer.mutated()
                    and (data_observer is None or not data_observer.mutated())
                )
            except OSError:
                confirmation_continuous = False
            if not confirmation_continuous:
                return _result(
                    request.cycle_id, "uncertain", True, None, None,
                    tuple(draft.problems + [_problem(
                        "post_publish_confirmation_lost", "publication",
                        "final bundle changed during terminal capability confirmation",
                        blocking=True,
                    )]),
                )
            if not _release_owned_lock(
                lock, lock_parent_fd, lock_fd,
                lock_parent_identity, lock_identity,
            ):
                return _result(
                    request.cycle_id, "uncertain", True, None, None,
                    tuple(draft.problems + [_problem(
                        "lock_cleanup_failed", "publication",
                        "evidence lock could not be removed after publication",
                        blocking=True,
                    )]),
                )
            lock_owned = False
            return _result(
                request.cycle_id, draft.manifest["status"], True,
                final.relative_to(self.root).as_posix(), TypedDigest.raw(seal_data),
                tuple(draft.problems),
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except PathBoundaryError as exc:
            status = (
                "uncertain" if published
                else "failed_no_final" if wrote_staging
                else "blocked"
            )
            return _result(
                request.cycle_id, status, published or wrote_staging, None, None,
                tuple(draft.problems + [_problem("write_boundary", "publication", str(exc), blocking=True)]),
            )
        except FileExistsError:
            if not lock_owned:
                return _result(
                    request.cycle_id, "blocked", False, None, None,
                    tuple(draft.problems),
                )
            if final.exists():
                if final_observer is None:
                    final_observer = SourceMutationObserver(
                        self.root, (final.relative_to(self.root).as_posix(),),
                    )
                if not final_observer.supported:
                    return _result(
                        request.cycle_id, "blocked", False, None, None,
                        tuple(draft.problems + [_problem(
                            "existing_final_observer_unavailable", "publication",
                            "concurrent final namespace continuity is unavailable",
                            blocking=True,
                        )]),
                    )
                existing = self._existing(final, request.cycle_id, request_digest)
                sources_continuous = self._adoption_sources_continuous(
                    request, draft, observer, data_observer,
                )
                confirmed = (
                    self._existing(final, request.cycle_id, request_digest)
                    if sources_continuous else None
                )
                exact_existing = (
                    existing is not None and existing.status == "adopted"
                    and confirmed is not None and confirmed.status == "adopted"
                    and confirmed.seal_digest == existing.seal_digest
                    and confirmed.sealed_status == existing.sealed_status
                )
                if exact_existing and (
                    not final_observer.mutated()
                    and root_identity(self.root) == root_before
                    and publication_directories is not None
                    and _directory_chain(
                        self.root, (lock.parent, final.parent, staging_parent),
                    ) == publication_directories
                    and _lock_continuous(
                        lock, lock_parent_fd, lock_fd,
                        lock_parent_identity, lock_identity,
                    )
                    and self._adoption_sources_continuous(
                        request, draft, observer, data_observer,
                    )
                ):
                    if _release_owned_lock(
                        lock, lock_parent_fd, lock_fd,
                        lock_parent_identity, lock_identity,
                    ):
                        lock_owned = False
                        return confirmed
                    return _result(
                        request.cycle_id, "blocked", False, None, None,
                        tuple(draft.problems + [_problem(
                            "lock_cleanup_failed", "publication",
                            "evidence lock could not be removed after concurrent adoption",
                            blocking=True,
                        )]),
                    )
                if exact_existing:
                    return _result(
                        request.cycle_id, "blocked", False, None, None,
                        tuple(draft.problems + [_problem(
                            "adoption_continuity_lost", "capture",
                            "concurrent exact evidence lost source or namespace continuity",
                            blocking=True,
                        )]),
                    )
                return _result(
                    request.cycle_id, "conflict", False, None, None,
                    tuple(draft.problems),
                )
            return _result(
                request.cycle_id,
                "failed_no_final" if wrote_staging else "blocked",
                wrote_staging, None, None,
                tuple(draft.problems + [_problem(
                    "staging_member_conflict" if wrote_staging else "staging_allocation_failed",
                    "publication",
                    "a staged evidence member already exists" if wrote_staging
                    else "a unique staging directory could not be allocated",
                    blocking=True,
                )]),
            )
        except Exception as exc:
            status = "uncertain" if published else "failed_no_final" if wrote_staging else "blocked"
            return _result(
                request.cycle_id, status, published or wrote_staging, None, None,
                tuple(draft.problems + [_problem("publication_failed", "publication", _exception_detail(exc), blocking=True)]),
            )
        finally:
            # A public-name identity check followed by path-based recursive
            # deletion has an unavoidable replacement race. Failed staging is
            # retained for owner-controlled cleanup instead of risking deletion
            # of a foreign replacement tree.
            try:
                if lock_parent_fd is not None:
                    try:
                        if (
                            lock_owned and lock_fd is not None
                            and lock_parent_identity is not None and lock_identity is not None
                            and publication_directories is not None
                            and _directory_chain(
                                self.root, (lock.parent, final.parent, staging_parent),
                            ) == publication_directories
                            and _lock_continuous(
                                lock, lock_parent_fd, lock_fd,
                                lock_parent_identity, lock_identity,
                            )
                        ):
                            _release_owned_lock(
                                lock, lock_parent_fd, lock_fd,
                                lock_parent_identity, lock_identity,
                            )
                    except (FileNotFoundError, PathBoundaryError):
                        pass
                    finally:
                        if lock_fd is not None:
                            os.close(lock_fd)
                        os.close(lock_parent_fd)
            except OSError:
                pass
            if final_observer is not None:
                final_observer.close()
            for descriptor in (
                stage_fd, staging_parent_fd, final_parent_fd, authority_root_fd,
            ):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
