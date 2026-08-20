"""Read-only, continuity-aware evidence inspection helpers."""

from __future__ import annotations

import json
import hashlib
import os
import stat
import subprocess
import ctypes
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Tuple

from quantpits.evidence.contracts import TypedDigest


class PathBoundaryError(ValueError):
    pass


def _exception_detail(exc: BaseException) -> str:
    detail = type(exc).__name__
    error_number = getattr(exc, "errno", None)
    return "%s(errno=%s)" % (detail, error_number) if error_number is not None else detail


class SourceMutationObserver:
    """Linux inotify guard for transient replace/move/write source events."""

    _EVENT = struct.Struct("iIII")
    _SELF_MASK = 0x00000002 | 0x00000004 | 0x00000008 | 0x00000400 | 0x00000800
    _PARENT_MASK = 0x00000040 | 0x00000080 | 0x00000100 | 0x00000200
    _BAD_GLOBAL = 0x00004000 | 0x00008000
    _MASK_ADD = 0x20000000

    def __init__(self, root: Path, paths: Iterable[str]) -> None:
        self.root = root.resolve(strict=True)
        self.fd = -1
        self.supported = False
        self._watches = {}
        self._mutated = False
        libc = ctypes.CDLL(None, use_errno=True)
        init = getattr(libc, "inotify_init1", None)
        add = getattr(libc, "inotify_add_watch", None)
        if init is None or add is None:
            return
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add.restype = ctypes.c_int
        self.fd = init(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        if self.fd < 0:
            return
        self.supported = True
        self._add = add
        self._watch(self.root, self._SELF_MASK, None, True)
        self._watch(self.root.parent, self._PARENT_MASK, {self.root.name}, False)
        self.add_paths(paths)

    def add_paths(self, paths: Iterable[str]) -> None:
        """Extend the observation window before reading newly discovered evidence."""
        if self.fd < 0:
            self.supported = False
            return
        for logical in paths:
            if not logical:
                continue
            try:
                target = contained_path(self.root, logical)
            except FileNotFoundError:
                raw = Path(logical)
                current = self.root
                for index, part in enumerate(raw.parts):
                    candidate = current / part
                    try:
                        info = os.lstat(str(candidate))
                    except FileNotFoundError:
                        self._watch(current, self._PARENT_MASK, {part}, False)
                        break
                    if stat.S_ISLNK(info.st_mode):
                        raise PathBoundaryError("path contains a symlink alias")
                    if index != len(raw.parts) - 1 and not stat.S_ISDIR(info.st_mode):
                        raise PathBoundaryError("path parent is not a directory")
                    current = candidate
                continue
            if target.is_dir():
                self._watch(target.parent, self._PARENT_MASK, {target.name}, False)
                for directory, _dirs, _files in os.walk(str(target), followlinks=False):
                    self._watch(Path(directory), self._SELF_MASK | self._PARENT_MASK, None, True)
            else:
                self._watch(target, self._SELF_MASK, None, True)
                self._watch(target.parent, self._PARENT_MASK, {target.name}, False)

    def _watch(self, path: Path, mask: int, names, any_event: bool) -> None:
        wd = self._add(self.fd, os.fsencode(str(path)), mask | self._MASK_ADD)
        if wd < 0:
            self.supported = False
            return
        current = self._watches.setdefault(wd, {"names": set(), "any": False})
        if names:
            current["names"].update(names)
        current["any"] = current["any"] or any_event

    def mutated(self) -> bool:
        if self.fd < 0:
            return self._mutated
        while True:
            try:
                data = os.read(self.fd, 64 * 1024)
            except BlockingIOError:
                break
            if not data:
                break
            offset = 0
            while offset + self._EVENT.size <= len(data):
                wd, mask, _cookie, length = self._EVENT.unpack_from(data, offset)
                offset += self._EVENT.size
                name = data[offset:offset + length].split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
                offset += length
                rule = self._watches.get(wd)
                if mask & self._BAD_GLOBAL:
                    self._mutated = True
                elif rule and (rule["any"] or name in rule["names"]):
                    self._mutated = True
        return self._mutated

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()


@dataclass(frozen=True)
class FileSnapshot:
    logical_path: str
    status: str
    data: Optional[bytes]
    digest: Optional[TypedDigest]
    identity: Optional[Tuple[int, int, int, int]]
    detail: str = ""

    def to_public_dict(self, preservation: str) -> dict:
        return {
            "path": self.logical_path,
            "status": self.status,
            "digest": self.digest.to_dict() if self.digest else None,
            "preservation_status": preservation,
            "detail": self.detail,
        }


def root_identity(root: Path) -> Tuple[int, int]:
    info = os.lstat(str(root))
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PathBoundaryError("workspace root is not a directory")
    return info.st_dev, info.st_ino


def contained_path(root: Path, logical_path: str, *, must_exist: bool = True) -> Path:
    if not isinstance(logical_path, str) or not logical_path:
        raise PathBoundaryError("path must be non-empty workspace-relative text")
    pure = PurePosixPath(logical_path)
    if (
        "\\" in logical_path or "\0" in logical_path
        or pure.is_absolute() or pure.as_posix() != logical_path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise PathBoundaryError("path must be canonical and workspace-relative")
    raw = Path(*pure.parts)
    canonical_root = root.absolute()
    root_info = os.lstat(str(canonical_root))
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise PathBoundaryError("authority root is not a canonical directory")
    candidate = canonical_root.joinpath(*raw.parts)
    current = canonical_root
    for index, part in enumerate(raw.parts):
        current = current / part
        try:
            current_info = os.lstat(str(current))
        except FileNotFoundError:
            if must_exist or index != len(raw.parts) - 1:
                raise
            break
        if stat.S_ISLNK(current_info.st_mode):
            raise PathBoundaryError("path contains a symlink alias")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        if must_exist:
            raise
        resolved = candidate.parent.resolve(strict=True) / candidate.name
    try:
        resolved.relative_to(canonical_root)
    except ValueError as exc:
        raise PathBoundaryError("path escapes the physical workspace") from exc
    return resolved


def _open_no_follow(root: Path, logical_path: str, *, directory: bool = False) -> int:
    """Open a file from its authority root without following parent aliases."""
    raw = Path(logical_path)
    descriptors = []
    try:
        current_fd = os.open(
            str(root.absolute()),
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptors.append(current_fd)
        for part in raw.parts[:-1]:
            current_fd = os.open(
                part, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=current_fd,
            )
            descriptors.append(current_fd)
        descriptor = os.open(
            raw.parts[-1],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            | (getattr(os, "O_DIRECTORY", 0) if directory else 0),
            dir_fd=current_fd,
        )
        return descriptor
    except OSError as exc:
        if exc.errno in {getattr(os, "ELOOP", 40), getattr(os, "ENOTDIR", 20)}:
            raise PathBoundaryError("path parent identity became noncanonical") from exc
        raise
    finally:
        for current in reversed(descriptors):
            os.close(current)


def inspect_file(root: Path, logical_path: str) -> FileSnapshot:
    """Read one regular file and prove its public-name identity stayed stable."""
    try:
        path = contained_path(root, logical_path)
        before = os.lstat(str(path))
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            return FileSnapshot(logical_path, "incomparable", None, None, None, "not a regular file")
        if before.st_nlink != 1:
            raise PathBoundaryError("hard-linked source has ambiguous physical ownership")
        descriptor = _open_no_follow(root, logical_path)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                return FileSnapshot(
                    logical_path, "changed", None, None, None,
                    "identity changed while opening",
                )
            if opened.st_size <= 2 * 1024 * 1024:
                data = handle.read()
                digest = TypedDigest.raw(data)
            else:
                hasher = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
                    size += len(chunk)
                data = None
                digest = TypedDigest("sha256", "raw_bytes", hasher.hexdigest(), size)
        after = os.lstat(str(contained_path(root, logical_path)))
        identities = [
            (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
            for item in (before, opened, after)
        ]
        if identities[0] != identities[1] or identities[0] != identities[2]:
            return FileSnapshot(logical_path, "changed", None, None, identities[2], "identity changed while reading")
        return FileSnapshot(logical_path, "observed", data, digest, identities[0])
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except PathBoundaryError:
        raise
    except FileNotFoundError:
        return FileSnapshot(logical_path, "missing", None, None, None, "file is missing")
    except Exception as exc:
        return FileSnapshot(logical_path, "incomparable", None, None, None, _exception_detail(exc))


def inspect_many(root: Path, members: Iterable[Tuple[str, str]]) -> Tuple[Tuple[str, FileSnapshot], ...]:
    """Preserve requested identity/order/cardinality after ordinary failures."""
    results = []
    for name, logical_path in members:
        try:
            snapshot = inspect_file(root, logical_path)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except PathBoundaryError:
            raise
        except Exception as exc:  # protects a caller-supplied inspector wrapper
            snapshot = FileSnapshot(
                logical_path, "incomparable", None, None, None,
                _exception_detail(exc),
            )
        results.append((name, snapshot))
    return tuple(results)


def inspect_tree(root: Path, logical_path: str) -> Tuple[FileSnapshot, ...]:
    base = contained_path(root, logical_path)
    if base.is_file():
        return (inspect_file(root, logical_path),)
    if not base.is_dir() or base.is_symlink():
        return (FileSnapshot(logical_path, "incomparable", None, None, None, "not a regular tree"),)
    members = []
    root_fd = _open_no_follow(root, logical_path, directory=True)

    def walk(directory_fd: int, parent: Path) -> None:
        with os.scandir(directory_fd) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
        for entry in entries:
            relative = (parent / entry.name).as_posix()
            try:
                if entry.is_symlink():
                    members.append(FileSnapshot(
                        relative, "incomparable", None, None, None,
                        "symlink is not evidence",
                    ))
                elif entry.is_file(follow_symlinks=False):
                    members.append(inspect_file(root, relative))
                elif entry.is_dir(follow_symlinks=False):
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                    try:
                        walk(child_fd, Path(relative))
                    finally:
                        os.close(child_fd)
                else:
                    members.append(FileSnapshot(
                        relative, "incomparable", None, None, None,
                        "special node is not evidence",
                    ))
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except PathBoundaryError:
                raise
            except OSError as exc:
                members.append(FileSnapshot(
                    relative, "changed", None, None, None,
                    _exception_detail(exc),
                ))

    try:
        walk(root_fd, Path(logical_path))
    finally:
        os.close(root_fd)
    return tuple(members)


def strict_json_object(data: bytes) -> Optional[dict]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("non-finite JSON number")

    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def parse_json(snapshot: FileSnapshot) -> Optional[dict]:
    if snapshot.status != "observed" or snapshot.data is None:
        return None
    return strict_json_object(snapshot.data)


def _git(repo: Path, *args: str) -> bytes:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
        timeout=30,
    ).stdout


def git_control_specs(start: Path) -> Tuple[Tuple[Path, Tuple[str, ...]], ...]:
    """Return private observer roots for transient Git identity mutations."""
    top = Path(
        _git(start, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve(strict=True)
    git_dir = Path(
        _git(start, "rev-parse", "--absolute-git-dir").decode("utf-8").strip()
    ).resolve(strict=True)
    common_raw = Path(
        _git(start, "rev-parse", "--git-common-dir").decode("utf-8").strip()
    )
    common_dir = (
        common_raw if common_raw.is_absolute() else top / common_raw
    ).resolve(strict=True)
    members = {}
    members.setdefault(git_dir, set()).update({"HEAD", "index"})
    members.setdefault(common_dir, set()).update({"packed-refs", "refs"})
    public_git_marker = top / ".git"
    try:
        marker_info = os.lstat(str(public_git_marker))
    except FileNotFoundError:
        marker_info = None
    if marker_info is not None and not stat.S_ISDIR(marker_info.st_mode):
        members.setdefault(top, set()).add(".git")
    return tuple(
        (root, tuple(sorted(paths)))
        for root, paths in sorted(members.items(), key=lambda item: str(item[0]))
    )


def inspect_git(start: Path) -> dict:
    """Observe a containing Git repository without exposing its absolute path."""
    try:
        top = Path(_git(start, "rev-parse", "--show-toplevel").decode().strip()).resolve()
        before = (_git(top, "rev-parse", "HEAD").decode().strip(), _git(top, "rev-parse", "HEAD^{tree}").decode().strip())
        status_bytes = _git(top, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        diff_bytes = _git(top, "diff", "--binary", "HEAD", "--")
        after = (_git(top, "rev-parse", "HEAD").decode().strip(), _git(top, "rev-parse", "HEAD^{tree}").decode().strip())
        status_after = _git(top, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        diff_after = _git(top, "diff", "--binary", "HEAD", "--")
        if before != after or status_bytes != status_after or diff_bytes != diff_after:
            return {"status": "dirty_unresolved", "detail": "Git identity changed during observation"}
        dirty = bool(status_bytes)
        status_inventory = [
            item.decode("utf-8", "backslashreplace")
            for item in status_bytes.split(b"\0") if item
        ]
        try:
            upstream = _git(top, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}").decode().strip()
            counts = _git(top, "rev-list", "--left-right", "--count", "HEAD..." + upstream).decode().split()
            ahead, behind = int(counts[0]), int(counts[1])
            remote_relation = (
                "aligned" if (ahead, behind) == (0, 0) else
                "ahead" if behind == 0 else "behind" if ahead == 0 else "diverged"
            )
        except Exception:
            remote_relation = "no_upstream"
        return {
            "status": "dirty_observed_and_fingerprinted" if dirty else "clean",
            "commit": before[0], "tree": before[1],
            "status_inventory_digest": TypedDigest.raw(status_bytes).to_dict(),
            "status_inventory": status_inventory,
            "tracked_diff_digest": TypedDigest.raw(diff_bytes).to_dict(),
            "repository_scope": "workspace" if top == start.resolve() else "containing_repository",
            "remote_relation": remote_relation,
        }
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        return {"status": "dirty_unresolved", "detail": _exception_detail(exc)}
