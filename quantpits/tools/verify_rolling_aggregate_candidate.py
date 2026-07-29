"""Reusable preflight for the bounded Rolling aggregate candidate gate.

The command is deliberately no-write unless ``--execute`` is present.  The
execute route creates deterministic execution-bound source fixtures without
training, materializes exactly one candidate, and proves zero-write reuse in a
second process.  Substantive scenario policy stays here rather than in a
candidate-specific wrapper.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import select
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
from unittest.mock import patch


GATE_PROTOCOL = "rolling_aggregate_candidate_gate_v2"
EXECUTE_AUTHORIZATION = "authorize-rolling-aggregate-candidate-gate-v2"
CLEANUP_AUTHORIZATION = "authorize-rolling-aggregate-gate-cleanup-v1"
WALL_SECONDS = 300
MIN_FREE_BYTES = 2 * 1024 ** 3
MAX_WRITE_BYTES = 512 * 1024 ** 2


class AggregateGateError(RuntimeError):
    pass


class _WorkspaceMutationObserver:
    """Observe every mutation below the gate's pre-created write parents."""

    _MASK = (
        0x00000002  # IN_MODIFY
        | 0x00000004  # IN_ATTRIB
        | 0x00000008  # IN_CLOSE_WRITE
        | 0x00000040  # IN_MOVED_FROM
        | 0x00000080  # IN_MOVED_TO
        | 0x00000100  # IN_CREATE
        | 0x00000200  # IN_DELETE
        | 0x00000400  # IN_DELETE_SELF
        | 0x00000800  # IN_MOVE_SELF
    )
    _ANCESTOR_MASK = (
        0x00000004  # IN_ATTRIB
        | 0x00000400  # IN_DELETE_SELF
        | 0x00000800  # IN_MOVE_SELF
    )
    _EVENT = struct.Struct("iIII")
    _registry_lock = threading.RLock()
    _active_observers = set()
    _saved_mkdir = None
    _saved_rename = None
    _saved_replace = None
    _pathlib_accessor = None
    _saved_pathlib_mkdir = None
    _saved_pathlib_rename = None
    _saved_pathlib_replace = None

    def __init__(self, root, excluded_relative_paths=()):
        self.root = Path(root).absolute()
        self._exclusions = frozenset(excluded_relative_paths)
        self._fd = -1
        self._root_fd = -1
        self._root_identity = None
        self._root_ancestors = []
        self._paths = {}
        self._observed = set()
        self._error = None
        self._stop_requested = threading.Event()
        self._thread = None
        self._add_watch = None
        self._lock = threading.RLock()
        self._registered = False

    def _excluded(self, path):
        if path == self.root:
            return False
        relative = path.relative_to(self.root)
        return bool(relative.parts and relative.parts[0] in self._exclusions)

    def _record_tree(self, directory):
        with self._lock:
            if directory.is_symlink():
                return
            for current, names, files in os.walk(
                str(directory), topdown=True, followlinks=False,
            ):
                current_path = Path(current)
                names[:] = [
                    name for name in names
                    if (
                        not (current_path / name).is_symlink()
                        and not (
                            current_path == self.root
                            and name in self._exclusions
                        )
                    )
                ]
                for name in names + files:
                    path = current_path / name
                    try:
                        relative = path.relative_to(self.root).as_posix()
                    except ValueError:
                        continue
                    self._observed.add(relative)
                self._add_directory_watch(current_path)

    def _add_directory_watch(self, directory):
        relative = (
            ""
            if directory == self.root
            else directory.relative_to(self.root).as_posix()
        )
        if self._excluded(directory):
            return
        watch = self._add_watch(
            self._fd, os.fsencode(str(directory)), self._MASK,
        )
        if watch < 0:
            error = ctypes.get_errno()
            if error == 2:  # The directory disappeared before observation.
                with self._lock:
                    self._observed.add(relative)
                return
            raise AggregateGateError(
                "gate lifecycle write parent is unavailable"
            )
        with self._lock:
            self._paths[watch] = relative

    @staticmethod
    def _absolute_path(path, directory_fd=None):
        candidate = Path(os.fsdecode(os.fspath(path)))
        if candidate.is_absolute():
            return candidate
        base = (
            Path(os.readlink("/proc/self/fd/%d" % directory_fd))
            if directory_fd is not None else Path.cwd()
        )
        return (base / candidate).absolute()

    def _observe_created_directory(self, path, directory_fd=None):
        try:
            candidate = self._absolute_path(path, directory_fd)
            relative = candidate.relative_to(self.root)
        except (OSError, TypeError, ValueError):
            return
        if not relative.parts or relative.parts[0] in self._exclusions:
            return
        with self._lock:
            self._observed.add(relative.as_posix())
        try:
            candidate.resolve(strict=True).relative_to(
                self.root.resolve(strict=True)
            )
        except (OSError, ValueError):
            return
        self._record_tree(candidate)

    @classmethod
    def _notify_created_directory(cls, path, directory_fd=None):
        with cls._registry_lock:
            observers = tuple(cls._active_observers)
        for observer in observers:
            observer._observe_created_directory(path, directory_fd)

    @classmethod
    def _mkdir(cls, path, mode=0o777, *, dir_fd=None):
        result = cls._saved_mkdir(path, mode, dir_fd=dir_fd)
        cls._notify_created_directory(path, dir_fd)
        return result

    @classmethod
    def _rename(
        cls, source, destination, *, src_dir_fd=None, dst_dir_fd=None,
    ):
        result = cls._saved_rename(
            source, destination,
            src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
        )
        try:
            candidate = cls._absolute_path(destination, dst_dir_fd)
            if candidate.is_dir() and not candidate.is_symlink():
                cls._notify_created_directory(destination, dst_dir_fd)
        except (OSError, TypeError, ValueError):
            pass
        return result

    @classmethod
    def _replace(
        cls, source, destination, *, src_dir_fd=None, dst_dir_fd=None,
    ):
        result = cls._saved_replace(
            source, destination,
            src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
        )
        try:
            candidate = cls._absolute_path(destination, dst_dir_fd)
            if candidate.is_dir() and not candidate.is_symlink():
                cls._notify_created_directory(destination, dst_dir_fd)
        except (OSError, TypeError, ValueError):
            pass
        return result

    @classmethod
    def _pathlib_mkdir(cls, path, mode=0o777):
        result = cls._saved_pathlib_mkdir(path, mode)
        cls._notify_created_directory(path)
        return result

    @classmethod
    def _pathlib_rename(cls, source, destination):
        result = cls._saved_pathlib_rename(source, destination)
        try:
            candidate = cls._absolute_path(destination)
            if candidate.is_dir() and not candidate.is_symlink():
                cls._notify_created_directory(destination)
        except (OSError, TypeError, ValueError):
            pass
        return result

    @classmethod
    def _pathlib_replace(cls, source, destination):
        result = cls._saved_pathlib_replace(source, destination)
        try:
            candidate = cls._absolute_path(destination)
            if candidate.is_dir() and not candidate.is_symlink():
                cls._notify_created_directory(destination)
        except (OSError, TypeError, ValueError):
            pass
        return result

    def _register_directory_creation_barrier(self):
        cls = type(self)
        with cls._registry_lock:
            if not cls._active_observers:
                cls._saved_mkdir = os.mkdir
                cls._saved_rename = os.rename
                cls._saved_replace = os.replace
                accessor = getattr(Path("."), "_accessor", None)
                try:
                    if accessor is not None:
                        cls._pathlib_accessor = accessor
                        cls._saved_pathlib_mkdir = accessor.mkdir
                        cls._saved_pathlib_rename = accessor.rename
                        cls._saved_pathlib_replace = accessor.replace
                        accessor.mkdir = cls._pathlib_mkdir
                        accessor.rename = cls._pathlib_rename
                        accessor.replace = cls._pathlib_replace
                    os.mkdir = cls._mkdir
                    os.rename = cls._rename
                    os.replace = cls._replace
                except Exception as exc:
                    os.mkdir = cls._saved_mkdir
                    os.rename = cls._saved_rename
                    os.replace = cls._saved_replace
                    if cls._pathlib_accessor is not None:
                        cls._pathlib_accessor.mkdir = (
                            cls._saved_pathlib_mkdir
                        )
                        cls._pathlib_accessor.rename = (
                            cls._saved_pathlib_rename
                        )
                        cls._pathlib_accessor.replace = (
                            cls._saved_pathlib_replace
                        )
                    cls._saved_mkdir = None
                    cls._saved_rename = None
                    cls._saved_replace = None
                    cls._pathlib_accessor = None
                    cls._saved_pathlib_mkdir = None
                    cls._saved_pathlib_rename = None
                    cls._saved_pathlib_replace = None
                    raise AggregateGateError(
                        "gate lifecycle directory barrier could not start"
                    ) from exc
            cls._active_observers.add(self)
            self._registered = True

    def _unregister_directory_creation_barrier(self):
        cls = type(self)
        with cls._registry_lock:
            if not self._registered:
                return
            cls._active_observers.discard(self)
            self._registered = False
            if not cls._active_observers:
                os.mkdir = cls._saved_mkdir
                os.rename = cls._saved_rename
                os.replace = cls._saved_replace
                if cls._pathlib_accessor is not None:
                    cls._pathlib_accessor.mkdir = cls._saved_pathlib_mkdir
                    cls._pathlib_accessor.rename = cls._saved_pathlib_rename
                    cls._pathlib_accessor.replace = cls._saved_pathlib_replace
                cls._saved_mkdir = None
                cls._saved_rename = None
                cls._saved_replace = None
                cls._pathlib_accessor = None
                cls._saved_pathlib_mkdir = None
                cls._saved_pathlib_rename = None
                cls._saved_pathlib_replace = None

    def _drain(self):
        while True:
            try:
                data = os.read(self._fd, 1024 * 1024)
            except BlockingIOError:
                return
            if not data:
                return
            offset = 0
            while offset < len(data):
                watch, mask, _cookie, name_length = self._EVENT.unpack_from(
                    data, offset,
                )
                offset += self._EVENT.size
                raw_name = data[offset:offset + name_length]
                offset += name_length
                if mask & 0x00004000:
                    raise AggregateGateError(
                        "gate lifecycle write observer overflowed"
                    )
                name = os.fsdecode(raw_name.rstrip(b"\0"))
                with self._lock:
                    parent = self._paths.get(watch, "")
                relative = "/".join(
                    item for item in (parent, name) if item
                )
                if relative or (not name and mask & self._MASK):
                    with self._lock:
                        self._observed.add(relative or ".")
                if (
                    relative
                    and mask & (0x00000100 | 0x00000080)
                    and mask & 0x40000000
                ):
                    self._record_tree(self.root / relative)

    def _consume(self):
        try:
            while not self._stop_requested.is_set():
                ready, _write, _error = select.select(
                    (self._fd,), (), (), 0.05,
                )
                if ready:
                    self._drain()
            self._drain()
        except BaseException as exc:
            self._error = exc

    @staticmethod
    def _directory_identity(observation):
        if not stat.S_ISDIR(observation.st_mode):
            raise AggregateGateError(
                "gate lifecycle root is not a directory"
            )
        return (
            observation.st_dev, observation.st_ino,
            stat.S_IFMT(observation.st_mode),
        )

    def _open_root_identity(self):
        flags = (
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            descriptor = os.open(str(self.root), flags)
            opened = os.fstat(descriptor)
            public = os.lstat(str(self.root))
            opened_identity = self._directory_identity(opened)
            public_identity = self._directory_identity(public)
        except BaseException as exc:
            if "descriptor" in locals():
                os.close(descriptor)
            if isinstance(
                exc, (KeyboardInterrupt, SystemExit, GeneratorExit),
            ):
                raise
            raise AggregateGateError(
                "gate lifecycle root identity is unavailable"
            ) from exc
        if opened_identity != public_identity:
            os.close(descriptor)
            raise AggregateGateError(
                "gate lifecycle root identity drifted"
            )
        self._root_fd = descriptor
        self._root_identity = opened_identity
        ancestors = []
        try:
            for path in reversed(self.root.parents):
                ancestor_fd = os.open(str(path), flags)
                try:
                    opened = self._directory_identity(
                        os.fstat(ancestor_fd)
                    )
                    public = self._directory_identity(
                        os.lstat(str(path))
                    )
                    if opened != public:
                        raise AggregateGateError(
                            "gate lifecycle root ancestor identity drifted"
                        )
                except BaseException:
                    os.close(ancestor_fd)
                    raise
                ancestors.append((ancestor_fd, path, opened))
        except BaseException as exc:
            for ancestor_fd, _path, _identity in ancestors:
                os.close(ancestor_fd)
            self._root_ancestors = []
            os.close(self._root_fd)
            self._root_fd = -1
            self._root_identity = None
            if isinstance(
                exc, (KeyboardInterrupt, SystemExit, GeneratorExit),
            ):
                raise
            raise AggregateGateError(
                "gate lifecycle root ancestor identity is unavailable"
            ) from exc
        self._root_ancestors = ancestors

    def _watch_root_ancestors(self):
        for _descriptor, path, _identity in self._root_ancestors:
            watch = self._add_watch(
                self._fd, os.fsencode(str(path)), self._ANCESTOR_MASK,
            )
            if watch < 0:
                raise AggregateGateError(
                    "gate lifecycle root ancestor is unavailable"
                )
            with self._lock:
                self._paths[watch] = ""

    def _assert_root_identity(self):
        if self._root_fd < 0 or self._root_identity is None:
            raise AggregateGateError(
                "gate lifecycle root identity is unavailable"
            )
        try:
            opened = self._directory_identity(os.fstat(self._root_fd))
            public = self._directory_identity(os.lstat(str(self.root)))
        except (OSError, AggregateGateError) as exc:
            raise AggregateGateError(
                "gate lifecycle root identity drifted"
            ) from exc
        if opened != self._root_identity or public != self._root_identity:
            raise AggregateGateError(
                "gate lifecycle root identity drifted"
            )
        for descriptor, path, identity in self._root_ancestors:
            try:
                opened = self._directory_identity(os.fstat(descriptor))
                public = self._directory_identity(os.lstat(str(path)))
            except (OSError, AggregateGateError) as exc:
                raise AggregateGateError(
                    "gate lifecycle root ancestor identity drifted"
                ) from exc
            if opened != identity or public != identity:
                raise AggregateGateError(
                    "gate lifecycle root ancestor identity drifted"
                )

    def start(self):
        self._open_root_identity()
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            init = libc.inotify_init1
            add_watch = libc.inotify_add_watch
            init.argtypes = (ctypes.c_int,)
            init.restype = ctypes.c_int
            add_watch.argtypes = (
                ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32,
            )
            add_watch.restype = ctypes.c_int
            fd = init(os.O_NONBLOCK | os.O_CLOEXEC)
        except AttributeError as exc:
            self.close()
            raise AggregateGateError(
                "gate lifecycle write observation requires inotify"
            ) from exc
        except BaseException as exc:
            self.close()
            if isinstance(
                exc, (KeyboardInterrupt, SystemExit, GeneratorExit),
            ):
                raise
            raise AggregateGateError(
                "gate lifecycle write observer could not start"
            ) from exc
        if fd < 0:
            self.close()
            raise AggregateGateError(
                "gate lifecycle write observer could not start"
            )
        self._fd = fd
        self._add_watch = add_watch
        try:
            self._watch_root_ancestors()
            self._add_directory_watch(self.root)
            self._record_tree(self.root)
            self._assert_root_identity()
            self._observed.clear()
            self._thread = threading.Thread(
                target=self._consume,
                name="rolling-aggregate-gate-write-observer",
                daemon=True,
            )
            self._thread.start()
            self._register_directory_creation_barrier()
        except BaseException:
            self.close()
            raise
        return self

    def stop(self):
        self._stop_requested.set()
        try:
            if self._thread is not None:
                self._thread.join(timeout=5)
                if self._thread.is_alive():
                    raise AggregateGateError(
                        "gate lifecycle write observer did not stop"
                    )
            if self._error is not None:
                raise self._error
            self._assert_root_identity()
        finally:
            self.close()
        with self._lock:
            return tuple(sorted(self._observed))

    def close(self):
        self._stop_requested.set()
        if (
            self._thread is not None
            and self._thread is not threading.current_thread()
            and self._thread.is_alive()
        ):
            self._thread.join(timeout=1)
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        if self._root_fd >= 0:
            os.close(self._root_fd)
            self._root_fd = -1
        self._root_identity = None
        for descriptor, _path, _identity in self._root_ancestors:
            os.close(descriptor)
        self._root_ancestors = []
        self._unregister_directory_creation_barrier()
        self._paths = {}
        self._thread = None
        self._add_watch = None


class _GateActivityObserver:
    """Deny network activity and count the fixed runner's actual actions."""

    def __init__(self, runtime_root=None):
        self.runner_calls = 0
        self.training_calls = 0
        self.gpu_calls = 0
        self.network_calls = 0
        self.runtime_root = (
            Path(runtime_root)
            if runtime_root is not None else Path("/tmp")
        )

    def observe_runner(self):
        self.runner_calls += 1

    def _profile(self, frame, event, _argument):
        if event != "call":
            return self._profile
        module = str(frame.f_globals.get("__name__", ""))
        function = frame.f_code.co_name
        if (
            function in (
                "fit", "train", "train_single_model", "train_cpcv_model",
            )
            and module.startswith((
                "qlib.contrib.model", "qlib.model.trainer",
                "quantpits.utils.train_utils",
            ))
        ):
            self.training_calls += 1
            raise AggregateGateError("training activity is forbidden")
        if module.startswith(("cupy.cuda", "torch.cuda")):
            self.gpu_calls += 1
            raise AggregateGateError("GPU activity is forbidden")
        return self._profile

    def _deny_network(self, *_args, **_kwargs):
        self.network_calls += 1
        raise AggregateGateError("network activity is forbidden")

    def disable_qlib_repository_logging(self):
        from qlib.workflow.recorder import MLflowRecorder

        self._stack.enter_context(patch.object(
            MLflowRecorder, "_log_uncommitted_code",
            lambda _recorder: None,
        ))

    def __enter__(self):
        self._stack = ExitStack()
        self._prior_profile = sys.getprofile()
        self._prior_thread_profile = getattr(
            threading, "_profile_hook", None,
        )
        self._prior_cwd = Path.cwd()
        self._stack.enter_context(
            patch.object(socket.socket, "connect", self._deny_network)
        )
        self._stack.enter_context(
            patch.object(socket.socket, "connect_ex", self._deny_network)
        )
        self._stack.enter_context(
            patch.object(socket.socket, "sendto", self._deny_network)
        )
        self._stack.enter_context(
            patch.object(socket, "create_connection", self._deny_network)
        )
        self._stack.enter_context(
            patch.object(socket, "getaddrinfo", self._deny_network)
        )
        self._stack.enter_context(
            patch.object(socket, "gethostbyname", self._deny_network)
        )
        self._stack.enter_context(patch.dict(os.environ, {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "HOME": str(self.runtime_root),
            "TMPDIR": str(self.runtime_root),
            "TMP": str(self.runtime_root),
            "TEMP": str(self.runtime_root),
            "XDG_CACHE_HOME": str(self.runtime_root / "cache"),
            "MPLCONFIGDIR": str(self.runtime_root / "matplotlib"),
            "USER": "quantpits-gate",
            "LOGNAME": "quantpits-gate",
        }))
        self._stack.enter_context(
            patch.object(tempfile, "tempdir", str(self.runtime_root))
        )
        self._stack.enter_context(
            patch.object(sys, "dont_write_bytecode", True)
        )
        self._stack.enter_context(
            patch.object(sys, "argv", ["quantpits-r35-gate"])
        )
        os.chdir("/")
        sys.setprofile(self._profile)
        threading.setprofile(self._profile)
        return self

    def __exit__(self, exc_type, exc, traceback):
        sys.setprofile(self._prior_profile)
        threading.setprofile(self._prior_thread_profile)
        try:
            return self._stack.__exit__(exc_type, exc, traceback)
        finally:
            os.chdir(str(self._prior_cwd))


def _process_physical_write_bytes():
    try:
        rows = {}
        for line in Path("/proc/self/io").read_text(
            encoding="ascii",
        ).splitlines():
            key, value = line.split(":", 1)
            rows[key] = int(value.strip())
        return rows["write_bytes"]
    except (OSError, KeyError, ValueError) as exc:
        raise AggregateGateError(
            "physical write-byte observer is unavailable"
        ) from exc


def _strict_int(value, field):
    if type(value) is not int:
        raise AggregateGateError("%s must be an exact integer" % field)
    return value


def _strict_bool(value, field):
    if type(value) is not bool:
        raise AggregateGateError("%s must be an exact boolean" % field)
    return value


@dataclass(frozen=True)
class AggregateGateScenario:
    protocol: str
    family: str
    target_count: int
    window_count: int
    source_unit_count: int
    training: bool
    expected_new_recorders: int
    expected_artifacts: tuple
    second_process_writer_calls: int
    second_process_new_recorders: int
    cleanup_default: str

    def __post_init__(self):
        if self.protocol != GATE_PROTOCOL:
            raise AggregateGateError("wrong gate protocol")
        if self.family != "rolling":
            raise AggregateGateError("gate family must be rolling")
        if (
            _strict_int(self.target_count, "target_count") != 1
            or _strict_int(self.window_count, "window_count") != 2
            or _strict_int(self.source_unit_count, "source_unit_count") != 2
        ):
            raise AggregateGateError(
                "selector must be exactly one target and two windows"
            )
        if _strict_bool(self.training, "training"):
            raise AggregateGateError("training is forbidden")
        if _strict_int(
            self.expected_new_recorders, "expected_new_recorders"
        ) != 1:
            raise AggregateGateError("gate requires exactly one new candidate")
        if self.expected_artifacts != (
            "pred.pkl", "aggregate_manifest.json",
        ):
            raise AggregateGateError("candidate artifact set is not exact")
        if (
            _strict_int(
                self.second_process_writer_calls,
                "second_process_writer_calls",
            ) != 0
            or _strict_int(
                self.second_process_new_recorders,
                "second_process_new_recorders",
            ) != 0
        ):
            raise AggregateGateError("reuse must perform zero writes")
        if self.cleanup_default != "preserve":
            raise AggregateGateError("gate cleanup must default to preserve")

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(
                self.to_public_dict(), sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def to_public_dict(self):
        return {
            "protocol": self.protocol,
            "family": self.family,
            "target_count": self.target_count,
            "window_count": self.window_count,
            "source_unit_count": self.source_unit_count,
            "training": self.training,
            "expected_new_recorders": self.expected_new_recorders,
            "expected_artifacts": list(self.expected_artifacts),
            "second_process_writer_calls": self.second_process_writer_calls,
            "second_process_new_recorders":
                self.second_process_new_recorders,
            "cleanup_default": self.cleanup_default,
        }


def frozen_scenario():
    return AggregateGateScenario(
        GATE_PROTOCOL, "rolling", 1, 2, 2, False, 1,
        ("pred.pkl", "aggregate_manifest.json"), 0, 0, "preserve",
    )


def scenario_from_mapping(payload):
    if not isinstance(payload, Mapping):
        raise AggregateGateError("scenario root must be a mapping")
    expected = frozenset(frozen_scenario().to_public_dict())
    if frozenset(payload) != expected:
        raise AggregateGateError("scenario fields are not exact")
    artifacts = payload["expected_artifacts"]
    if not isinstance(artifacts, list):
        raise AggregateGateError("expected_artifacts must be a JSON array")
    return AggregateGateScenario(
        payload["protocol"], payload["family"], payload["target_count"],
        payload["window_count"], payload["source_unit_count"],
        payload["training"], payload["expected_new_recorders"],
        tuple(artifacts), payload["second_process_writer_calls"],
        payload["second_process_new_recorders"],
        payload["cleanup_default"],
    )


def _real_directory(path, field, allow_linked_entry=False):
    path = Path(path).expanduser().absolute()
    if not path.exists() or not path.is_dir():
        raise AggregateGateError("%s must be a real existing directory" % field)
    resolved = path.resolve(strict=True)
    if not allow_linked_entry and (path.is_symlink() or resolved != path):
        raise AggregateGateError("%s contains a symlink" % field)
    return resolved


def _stable_regular_file_bytes(path, initial, field):
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
    ):
        raise AggregateGateError(
            "%s is not one canonical regular file" % field
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise AggregateGateError(
            "%s could not be opened canonically" % field
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino)
            != (initial.st_dev, initial.st_ino)
        ):
            raise AggregateGateError(
                "%s identity drifted before read" % field
            )
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        completed = os.fstat(descriptor)
        terminal = os.lstat(str(path))
        expected = (
            opened.st_dev, opened.st_ino, opened.st_nlink,
            opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(terminal.st_mode)
            or terminal.st_nlink != 1
            or (
                completed.st_dev, completed.st_ino, completed.st_nlink,
                completed.st_size, completed.st_mtime_ns,
                completed.st_ctime_ns,
            ) != expected
            or (
                terminal.st_dev, terminal.st_ino, terminal.st_nlink,
                terminal.st_size, terminal.st_mtime_ns,
                terminal.st_ctime_ns,
            ) != expected
        ):
            raise AggregateGateError(
                "%s identity drifted during read" % field
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stable_symlink_target(path, initial):
    if not stat.S_ISLNK(initial.st_mode):
        raise AggregateGateError("snapshot node is not a symlink")
    expected = (
        initial.st_dev, initial.st_ino, initial.st_mode, initial.st_nlink,
        initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns,
    )
    try:
        target = os.readlink(str(path))
        terminal = os.lstat(str(path))
    except OSError as exc:
        raise AggregateGateError(
            "snapshot symlink could not be observed canonically"
        ) from exc
    if (
        not stat.S_ISLNK(terminal.st_mode)
        or (
            terminal.st_dev, terminal.st_ino, terminal.st_mode,
            terminal.st_nlink, terminal.st_size, terminal.st_mtime_ns,
            terminal.st_ctime_ns,
        ) != expected
    ):
        raise AggregateGateError(
            "snapshot symlink identity drifted during observation"
        )
    return os.fsencode(target)


def snapshot_tree(root, allow_symlinks=False):
    if type(allow_symlinks) is not bool:
        raise AggregateGateError(
            "snapshot symlink policy must be an exact boolean"
        )
    root = _real_directory(root, "snapshot root")
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        node = path.lstat()
        if stat.S_ISLNK(node.st_mode):
            if not allow_symlinks:
                raise AggregateGateError("snapshot encountered a symlink")
            target = _stable_symlink_target(path, node)
            rows.append((
                relative, "symlink", len(target),
                hashlib.sha256(target).hexdigest(),
            ))
        elif stat.S_ISREG(node.st_mode):
            data = _stable_regular_file_bytes(
                path, node, "snapshot node",
            )
            rows.append((relative, "file", len(data), hashlib.sha256(data).hexdigest()))
        elif stat.S_ISDIR(node.st_mode):
            rows.append((relative, "directory", None, None))
        else:
            raise AggregateGateError("snapshot encountered a special node")
    return tuple(rows)


def validate_binding(
    scenario, workspace, protected_workspace, commit, tree,
    execute=False, authorization=None,
):
    if not isinstance(scenario, AggregateGateScenario):
        raise AggregateGateError("binding requires a validated scenario")
    workspace = _real_directory(workspace, "disposable workspace")
    protected_values = (
        tuple(protected_workspace)
        if isinstance(protected_workspace, (tuple, list))
        else (protected_workspace,)
    )
    if not protected_values:
        raise AggregateGateError(
            "at least one protected workspace is required"
        )
    protected = tuple(_real_directory(
        value, "protected workspace", allow_linked_entry=True,
    ) for value in protected_values)
    if len(protected) != len(set(protected)):
        raise AggregateGateError("protected workspaces must be unique")
    repository = Path(__file__).resolve().parents[2]
    if any(
        _path_contains(workspace, boundary)
        or _path_contains(boundary, workspace)
        for boundary in protected + (repository,)
    ):
        raise AggregateGateError(
            "disposable workspace overlaps a protected boundary"
        )
    for value, field in ((commit, "commit"), (tree, "tree")):
        if (
            not isinstance(value, str) or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise AggregateGateError("%s must be a full lowercase git id" % field)
    try:
        actual_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repository),
            text=True, timeout=10,
        ).strip()
        actual_tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=str(repository),
            text=True, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AggregateGateError("candidate git identity is unavailable") from exc
    if commit != actual_commit or tree != actual_tree:
        raise AggregateGateError("binding does not match the checked-out candidate")
    if shutil.disk_usage(str(workspace)).free < MIN_FREE_BYTES:
        raise AggregateGateError("disposable filesystem has insufficient capacity")
    if type(execute) is not bool:
        raise AggregateGateError("execute must be an exact boolean")
    if execute and authorization != EXECUTE_AUTHORIZATION:
        raise AggregateGateError("execute authorization is missing or invalid")
    if execute:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repository), text=True, timeout=10,
        )
        untracked_runtime = subprocess.check_output(
            [
                "git", "ls-files", "--others", "--exclude-standard",
                "--", "quantpits",
            ],
            cwd=str(repository), text=True, timeout=10,
        )
        if dirty.strip() or untracked_runtime.strip():
            raise AggregateGateError(
                "execute requires a clean tracked candidate"
            )
    return {
        "scenario_fingerprint": scenario.fingerprint,
        "workspace": workspace,
        "protected_workspace": protected[0],
        "protected_workspaces": protected,
        "commit": commit,
        "tree": tree,
        "execute": execute,
    }


def preflight_evidence(binding):
    protected = tuple(
        snapshot_tree(root, allow_symlinks=True)
        for root in _binding_protected_workspaces(binding)
    )
    return {
        "protocol": GATE_PROTOCOL,
        "status": "preflight_passed",
        "reason_code": "rolling_aggregate_gate_preflight_passed",
        "scenario_fingerprint": binding["scenario_fingerprint"],
        "commit": binding["commit"],
        "tree": binding["tree"],
        "budgets": {
            "wall_seconds": WALL_SECONDS,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "maximum_write_bytes": MAX_WRITE_BYTES,
            "network": 0,
            "gpu": 0,
            "training_calls": 0,
        },
        "protected_snapshot_fingerprint": hashlib.sha256(
            json.dumps(protected, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "execute_authorized": binding["execute"],
        "cleanup": "preserve",
    }


def _initialize_fixture_tracking(context):
    import qlib
    from qlib.constant import REG_CN

    qlib.init(
        provider_uri=context.qlib_data_dir,
        region=REG_CN,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": context.mlflow_uri,
                "default_exp_name": "Experiment",
            },
        },
    )


def _build_fixture_scope(context):
    from quantpits.model_capabilities.catalog import AUTHORITATIVE_CATALOG
    from quantpits.model_capabilities.inspector import ModelCapabilityInspector
    from quantpits.model_capabilities.probes import (
        ImportObservation,
        ProtocolProbeFailure,
    )
    from quantpits.rolling import (
        PreparedRollingRun,
        ResolvedRollingRun,
        RollingAnchorPolicy,
        RollingRunIdentity,
        RollingRunOptions,
        RollingTarget,
        RollingTargetIdentity,
        RollingWindowDescriptor,
        RollingWindowIdentity,
        build_rolling_execution_scope,
        map_workflow_capability,
        observe_rolling_business_sessions,
        workspace_fingerprint,
    )
    from quantpits.runtime.command import CommandPlan, fingerprint_command_plan
    from quantpits.utils.workspace import fingerprint_value

    declaration = next(
        item for item in AUTHORITATIVE_CATALOG
        if item.model_module == "qlib.contrib.model.linear"
        and item.model_class == "LinearModel"
        and item.dataset_module == "qlib.data.dataset"
        and item.dataset_class == "DatasetH"
        and item.action == "train"
        and item.execution_family == "rolling"
    )
    matrix = ModelCapabilityInspector._with_probes(
        lambda _module, _class: ImportObservation(
            False, False, False, False, False, False, True,
            "gate_fixture_model_probe_forbidden",
        ),
        lambda _declaration: ProtocolProbeFailure(
            "gate_fixture_protocol_probe_forbidden",
        ),
    ).inspect((declaration,))
    if matrix.results[0].preflight_allowed:
        raise AggregateGateError(
            "deterministic fixture cannot claim model capability"
        )
    relative = "config/demo_linear_gate.yaml"
    workflow = context.root / relative
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow_bytes = (
        "task:\n"
        "  model: {class: LinearModel, module_path: qlib.contrib.model.linear}\n"
        "  dataset:\n"
        "    class: DatasetH\n"
        "    module_path: qlib.data.dataset\n"
        "    kwargs:\n"
        "      handler:\n"
        "        class: Alpha158\n"
        "        module_path: qlib.contrib.data.handler\n"
        "        kwargs: {}\n"
        "      segments: {}\n"
    ).encode("utf-8")
    if not workflow.exists():
        workflow.write_bytes(workflow_bytes)
    elif workflow.read_bytes() != workflow_bytes:
        raise AggregateGateError("deterministic fixture workflow drifted")
    target_identity = RollingTargetIdentity("demo_linear", "rolling")
    mapped = map_workflow_capability(
        context, target_identity.target_key, relative, matrix,
    )
    prepared_target = RollingTarget(
        target_identity, relative, mapped.workflow_fingerprint,
        "gate_fixture", {},
    )
    identities = (
        RollingWindowIdentity(
            "rolling", "2023-01-01", "2023-12-31",
            "2025-01-05", "2025-01-06", "a" * 64,
            valid_start="2024-01-01", valid_end="2024-12-31",
        ),
        RollingWindowIdentity(
            "rolling", "2023-02-01", "2024-01-31",
            "2025-02-05", "2025-02-06", "b" * 64,
            valid_start="2024-02-01", valid_end="2025-01-31",
        ),
    )
    windows = tuple(
        RollingWindowDescriptor(index, identity, {
            "window_idx": index,
            "train_start": identity.train_start,
            "train_end": identity.train_end,
            "valid_start": identity.valid_start,
            "valid_end": identity.valid_end,
            "test_start": identity.test_start,
            "test_end": identity.test_end,
        })
        for index, identity in enumerate(identities)
    )
    plan = CommandPlan(
        "rolling_train", context.root.name, "aggregate-gate",
        mode="rolling:merge",
    )
    plan_fingerprint = fingerprint_command_plan(plan, length=64)
    prepared = PreparedRollingRun(
        context, RollingRunOptions(action="merge"), (), {},
        (prepared_target,), None, RollingAnchorPolicy("gate_fixture"),
        plan, plan_fingerprint, {},
    )
    runtime_params = {"market": "demo", "benchmark": "demo"}
    identity = RollingRunIdentity(
        workspace_fingerprint(context.root), "rolling", "merge",
        plan_fingerprint, fingerprint_value({}), identities[-1].test_end,
        (mapped.target_key,), tuple(item.window_key for item in windows),
        fingerprint_value(runtime_params),
    )
    resolved = ResolvedRollingRun(
        prepared, identity.anchor_date, runtime_params, windows, identity,
    )
    observed = observe_rolling_business_sessions(
        windows, lambda start, end: (start, end),
    )
    return build_rolling_execution_scope(
        prepared, resolved, tuple(item.window_key for item in observed),
        (mapped,), observed,
    ), runtime_params


class _FixtureExecutionBackend:
    """Delegate real evidence/MLflow work while freezing a tiny calendar."""

    def __init__(self, delegate):
        self._delegate = delegate

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    @staticmethod
    def calendar_sessions(start, end):
        return (start, end)


class _FixtureRunner:
    def __init__(self, context, runtime_params, activity):
        self.context = context
        self.runtime_params = runtime_params
        self.activity = activity

    @property
    def runtime_params_fingerprint(self):
        from quantpits.utils.workspace import fingerprint_value
        return fingerprint_value(self.runtime_params)

    def execute(self, scope, unit, attempt_id):
        import pandas as pd
        from qlib.workflow import R
        from quantpits.rolling import RollingUnitRunnerObservation

        self.activity.observe_runner()
        experiment_name = "Rolling_Aggregate_Gate_Sources"
        with R.start(experiment_name=experiment_name):
            recorder = R.get_recorder()
            R.set_tags(
                execution_protocol=scope.execution_protocol_version,
                run_fingerprint=scope.run_identity.fingerprint,
                attempt_id=attempt_id,
                target_key=unit.unit_key[0],
                window_key=unit.unit_key[1],
                source_operation=scope.run_identity.action,
                fixture_kind="deterministic_no_training",
            )
            rows = [
                (pd.Timestamp(session), "DEMO")
                for session in unit.window.expected_sessions
            ]
            prediction = pd.DataFrame(
                {"score": [float(index + 1) for index in range(len(rows))]},
                index=pd.MultiIndex.from_tuples(
                    rows, names=("datetime", "instrument"),
                ),
            )
            recorder.save_objects(
                **{"model.pkl": {"fixture": True}, "pred.pkl": prediction}
            )
            recorder_id = str(recorder.info["id"])
            experiment_id = str(
                R.get_exp(
                    experiment_name=experiment_name, create=False,
                ).id
            )
        return RollingUnitRunnerObservation(
            unit.unit_key, attempt_id, "candidate_success",
            experiment_name, experiment_id, recorder_id,
        )


def _materialize_no_training_source_fixtures(
    scope, repository, source, runner,
):
    """Create exact Phase 32/34 fixture evidence without capability probing."""

    from quantpits.rolling.execution_backend import (
        _claim,
        _snapshot,
        _source_extensions,
    )

    attempt_id = "gate-execution-attempt"
    initial = repository.inspect_readonly()
    if initial.inspection.classification != "missing":
        raise AggregateGateError("source fixture state is not disposable")
    receipt = repository.commit(
        _snapshot(scope, "prepared", None, ()),
        initial.baseline,
    )
    if receipt.status != "committed" or receipt.cas_baseline is None:
        raise AggregateGateError("source fixture prepared State failed")
    baseline = receipt.cas_baseline
    requests = []
    successful_claims = []
    for unit in scope.units:
        recorder_baseline = source.capture_recorder_inventory(
            scope, unit, attempt_id,
        )
        observation = runner.execute(scope, unit, attempt_id)
        request = source.commit_execution_manifest(
            scope, unit, observation, recorder_baseline,
        )
        evidence = source.inspect(scope, (request,))
        if (
            evidence.status != "all_valid"
            or len(evidence.unit_results) != 1
            or evidence.unit_results[0].classification != "valid"
        ):
            raise AggregateGateError(
                "source fixture evidence did not become valid"
            )
        requests.append(request)
        successful_claims.append(_claim(
            unit, "success", observation.recorder_id,
            evidence.unit_results[0].evidence_fingerprint,
            _source_extensions(request, attempt_id),
        ))
    claims = [_claim(unit, "running") for unit in scope.units]
    receipt = repository.commit(
        _snapshot(scope, "executing", attempt_id, claims),
        baseline,
    )
    if receipt.status != "committed" or receipt.cas_baseline is None:
        raise AggregateGateError(
            "source fixture executing State failed: %s"
            % receipt.reason_code
        )
    baseline = receipt.cas_baseline
    for position, success_claim in enumerate(successful_claims):
        claims[position] = success_claim
        evidence_set = source.inspect(
            scope, tuple(requests[:position + 1]),
        )
        receipt = repository.commit_evidence_authorized(
            _snapshot(scope, "executing", attempt_id, claims),
            baseline, evidence_set,
        )
        if receipt.status != "committed" or receipt.cas_baseline is None:
            raise AggregateGateError(
                "source fixture success State failed: %s"
                % receipt.reason_code
            )
        baseline = receipt.cas_baseline
    evidence_set = source.inspect(scope, tuple(requests))
    if (
        evidence_set.status != "all_valid"
        or evidence_set.requested_unit_keys != scope.requested_unit_keys
    ):
        raise AggregateGateError("source fixture evidence set is incomplete")
    receipt = repository.commit_evidence_authorized(
        _snapshot(scope, "units_complete", attempt_id, claims),
        baseline, evidence_set,
    )
    if receipt.status != "committed":
        raise AggregateGateError("source fixture terminal State failed")
    return repository.inspect_readonly()


def _workspace_bytes(snapshot):
    return sum(
        size for _path, kind, size, _digest_value in snapshot
        if kind == "file"
    )


def _assert_gate_budgets(elapsed_seconds, write_bytes):
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or elapsed_seconds < 0
    ):
        raise AggregateGateError("gate elapsed time is invalid")
    if _strict_int(write_bytes, "write_bytes") < 0:
        raise AggregateGateError("gate write-byte count is invalid")
    if elapsed_seconds > WALL_SECONDS:
        raise AggregateGateError("gate wall-clock budget exceeded")
    if write_bytes > MAX_WRITE_BYTES:
        raise AggregateGateError("gate write-byte budget exceeded")


_RUN_RESULT_FIELDS = frozenset({
    "status", "candidate_fingerprint", "candidate_row_count",
    "new_candidate_recorders", "training_calls", "gpu_calls",
    "network_calls", "runner_calls", "protected_unchanged",
    "repository_unchanged", "elapsed_seconds", "workspace_bytes",
    "write_bytes", "changed_path_count",
})


def _validate_run_result(result, reuse_only):
    if not isinstance(result, Mapping) or frozenset(result) != _RUN_RESULT_FIELDS:
        raise AggregateGateError("gate run evidence fields are not exact")
    expected_status = (
        "reused_success" if reuse_only else "materialized_success"
    )
    expected_new = 0 if reuse_only else 1
    expected_runners = 0 if reuse_only else 2
    if (
        result["status"] != expected_status
        or _strict_int(
            result["new_candidate_recorders"],
            "new_candidate_recorders",
        ) != expected_new
        or _strict_int(result["runner_calls"], "runner_calls")
        != expected_runners
        or any(
            _strict_int(result[field], field) != 0
            for field in ("training_calls", "gpu_calls", "network_calls")
        )
        or _strict_int(
            result["candidate_row_count"], "candidate_row_count",
        ) != 4
        or not _strict_bool(
            result["protected_unchanged"], "protected_unchanged",
        )
        or not _strict_bool(
            result["repository_unchanged"], "repository_unchanged",
        )
        or not isinstance(result["candidate_fingerprint"], str)
        or len(result["candidate_fingerprint"]) != 64
        or any(
            char not in "0123456789abcdef"
            for char in result["candidate_fingerprint"]
        )
    ):
        raise AggregateGateError("gate run evidence is inconsistent")
    for field in ("workspace_bytes", "write_bytes", "changed_path_count"):
        if _strict_int(result[field], field) < 0:
            raise AggregateGateError("gate run count is negative")
    elapsed = result["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed < 0
    ):
        raise AggregateGateError("gate elapsed time is invalid")
    _assert_gate_budgets(float(elapsed), result["write_bytes"])
    if (
        result["workspace_bytes"] <= 0
        or (
            reuse_only
            and (
                result["write_bytes"] != 0
                or result["changed_path_count"] != 0
            )
        )
        or (
            not reuse_only
            and result["changed_path_count"] <= 0
        )
    ):
        raise AggregateGateError("gate write evidence is inconsistent")
    return result


def _assert_snapshot_unchanged(before, after, field):
    if before != after:
        raise AggregateGateError("%s drifted" % field)


def _assert_workspace_write_allowlist(
    before, after, mutation_paths=(), observed_write_bytes=None,
    reuse_only=None,
):
    before_map = {item[0]: item[1:] for item in before}
    after_map = {item[0]: item[1:] for item in after}
    changed = {
        path for path in set(before_map) | set(after_map)
        if before_map.get(path) != after_map.get(path)
    }
    allowed_exact = {
        "config",
        "config/demo_linear_gate.yaml",
        "data",
        "data/rolling_state.json",
        "data/rolling_state.json.lock",
        "data/locks",
        "data/locks/rolling_aggregate_candidate.lock",
        "data/locks/training_execution.lock",
        "data/aggregate_gate_scenario.json",
        "data/aggregate_gate_runtime",
        "data/rolling_aggregate_candidates_rolling",
        "mlruns",
        "mlruns/.aggregate-gate",
        "mlruns/.trash",
        "mlruns/filelock",
        "output",
        "qlib_data",
    }
    transient_patterns = (
        r"data/aggregate_gate_runtime/"
        r"quantpits-execution-manifest-[a-z0-9_]+",
        r"data/aggregate_gate_runtime/"
        r"quantpits-execution-manifest-[a-z0-9_]+/"
        r"execution_manifest\.json",
        r"data/aggregate_gate_runtime/tmp[a-z0-9_]+",
        r"data/aggregate_gate_runtime/tmp[a-z0-9_]+/"
        r"(?:model\.pkl|pred\.pkl)",
    )
    after_paths = frozenset(after_map)
    experiment_ids = frozenset(
        match.group(1)
        for path in after_paths
        for match in [re.fullmatch(r"mlruns/([0-9]+)/meta\.yaml", path)]
        if match is not None
    )
    run_ids = frozenset(
        (match.group(1), match.group(2))
        for path in after_paths
        for match in [
            re.fullmatch(
                r"mlruns/([0-9]+)/([0-9a-f]{32})/meta\.yaml", path,
            )
        ]
        if match is not None
    )
    source_run_ids = frozenset(
        (match.group(1), match.group(2))
        for path in after_paths
        for match in [
            re.fullmatch(
                r"mlruns/([0-9]+)/([0-9a-f]{32})/"
                r"artifacts/execution_manifest\.json",
                path,
            )
        ]
        if match is not None
    )
    candidate_ids = frozenset(
        match.group(1)
        for path in after_paths
        for match in [
            re.fullmatch(
                r"data/rolling_aggregate_candidates_rolling/"
                r"([0-9a-f]{32})/artifacts/"
                r"(?:pred\.pkl|aggregate_manifest\.json)",
                path,
            )
        ]
        if match is not None
    )

    def persisted_gate_path(path):
        candidate = re.fullmatch(
            r"data/rolling_aggregate_candidates_rolling/"
            r"([0-9a-f]{32})(?:|/artifacts|/artifacts/"
            r"(?:pred\.pkl|aggregate_manifest\.json))",
            path,
        )
        if candidate is not None:
            return candidate.group(1) in candidate_ids
        experiment = re.fullmatch(
            r"mlruns/([0-9]+)(/meta\.yaml)?", path,
        )
        if experiment is not None:
            return experiment.group(1) in experiment_ids
        run = re.fullmatch(
            r"mlruns/([0-9]+)/([0-9a-f]{32})"
            r"(?:/(artifacts|metrics|params|tags|meta\.yaml))?",
            path,
        )
        if run is not None:
            return (run.group(1), run.group(2)) in run_ids
        artifact = re.fullmatch(
            r"mlruns/([0-9]+)/([0-9a-f]{32})/artifacts/"
            r"(execution_manifest\.json|model\.pkl|pred\.pkl)",
            path,
        )
        if artifact is not None:
            return (artifact.group(1), artifact.group(2)) in source_run_ids
        parameter = re.fullmatch(
            r"mlruns/([0-9]+)/([0-9a-f]{32})/params/cmd-sys\.argv",
            path,
        )
        if parameter is not None:
            return (parameter.group(1), parameter.group(2)) in source_run_ids
        tag = re.fullmatch(
            r"mlruns/([0-9]+)/([0-9a-f]{32})/tags/"
            r"(aggregate_attempt_id|aggregate_protocol|attempt_id|"
            r"candidate_key|candidate_kind|execution_protocol|fixture_kind|"
            r"mlflow\.runName|mlflow\.source\.name|mlflow\.source\.type|"
            r"mlflow\.user|run_fingerprint|scope_fingerprint|"
            r"source_operation|target_key|window_key)",
            path,
        )
        if tag is None:
            return False
        identity = (tag.group(1), tag.group(2))
        name = tag.group(3)
        if name == "mlflow.runName":
            return identity in run_ids
        if identity in source_run_ids:
            return name in {
                "attempt_id", "execution_protocol", "fixture_kind",
                "mlflow.source.name", "mlflow.source.type", "mlflow.user",
                "run_fingerprint", "source_operation", "target_key",
                "window_key",
            }
        return (
            identity in run_ids
            and identity[1] in candidate_ids
            and name in {
                "aggregate_attempt_id", "aggregate_protocol",
                "candidate_key", "candidate_kind", "scope_fingerprint",
                "target_key",
            }
        )

    def allowed(path):
        return (
            path in allowed_exact
            or any(
                re.fullmatch(pattern, path) is not None
                for pattern in transient_patterns
            )
            or persisted_gate_path(path)
            or re.fullmatch(
                r"data/\.rolling_state\.json\.[0-9a-f]{16}\.tmp",
                path,
            ) is not None
            or re.fullmatch(
                r"data/\.quantpits-aggregate-[a-z0-9_]+",
                path,
            ) is not None
            or re.fullmatch(
                r"data/\.quantpits-aggregate-[a-z0-9_]+/"
                r"(?:pred\.pkl|aggregate_manifest\.json)",
                path,
            ) is not None
        )

    observed_paths = changed | set(mutation_paths)
    if reuse_only is not None and type(reuse_only) is not bool:
        raise AggregateGateError("reuse_only must be an exact boolean")
    if reuse_only is not None:
        run_counts = sorted(
            sum(1 for item in run_ids if item[0] == experiment_id)
            for experiment_id in experiment_ids
        )
        candidate_runs = {
            identity for identity in run_ids
            if identity[1] in candidate_ids
        }
        source_experiments = {
            experiment_id for experiment_id, _run_id in source_run_ids
        }
        candidate_experiments = {
            experiment_id for experiment_id, _run_id in candidate_runs
        }
        if (
            len(experiment_ids) != 2
            or len(run_ids) != 3
            or run_counts != [1, 2]
            or len(source_run_ids) != 2
            or len(source_experiments) != 1
            or len(candidate_ids) != 1
            or len(candidate_runs) != 1
            or candidate_runs & source_run_ids
            or len(candidate_experiments) != 1
            or candidate_experiments & source_experiments
        ):
            raise AggregateGateError(
                "gate persisted write namespace is not exact"
            )
        transient_roots = {
            "state": {
                path for path in mutation_paths
                if re.fullmatch(
                    r"data/\.rolling_state\.json\.[0-9a-f]{16}\.tmp",
                    path,
                )
            },
            "manifest": {
                path for path in mutation_paths
                if re.fullmatch(
                    r"data/aggregate_gate_runtime/"
                    r"quantpits-execution-manifest-[a-z0-9_]+",
                    path,
                )
            },
            "objects": {
                path for path in mutation_paths
                if re.fullmatch(
                    r"data/aggregate_gate_runtime/tmp[a-z0-9_]+",
                    path,
                )
            },
            "candidate": {
                path for path in mutation_paths
                if re.fullmatch(
                    r"data/\.quantpits-aggregate-[a-z0-9_]+",
                    path,
                )
            },
        }
        expected_counts = (
            {"state": 0, "manifest": 0, "objects": 0, "candidate": 0}
            if reuse_only else
            {"state": 5, "manifest": 2, "objects": 2, "candidate": 1}
        )
        if {
            key: len(value) for key, value in transient_roots.items()
        } != expected_counts:
            raise AggregateGateError(
                "gate transient write namespace is not exact"
            )
    unexpected = tuple(sorted(
        path for path in observed_paths
        if not allowed(path)
    ))
    if unexpected:
        raise AggregateGateError(
            "gate observed an undeclared write: %s"
            % ", ".join(unexpected[:5])
        )
    before_bytes = _workspace_bytes(before)
    after_bytes = _workspace_bytes(after)
    write_bytes = (
        max(0, after_bytes - before_bytes)
        if observed_write_bytes is None
        else _strict_int(observed_write_bytes, "observed_write_bytes")
    )
    if write_bytes < 0:
        raise AggregateGateError("observed_write_bytes cannot be negative")
    _assert_gate_budgets(0, write_bytes)
    return len(observed_paths), write_bytes


def _snapshot_tracked_repository(root):
    try:
        names = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=str(root), timeout=20,
        ).split(b"\0")
    except (OSError, subprocess.SubprocessError) as exc:
        raise AggregateGateError("tracked repository inventory failed") from exc
    rows = []
    for raw in names:
        if not raw:
            continue
        relative = raw.decode("utf-8")
        path = root / relative
        try:
            node = os.lstat(str(path))
        except OSError as exc:
            raise AggregateGateError(
                "tracked repository node is unavailable"
            ) from exc
        data = _stable_regular_file_bytes(
            path, node, "tracked repository node",
        )
        rows.append((relative, len(data), hashlib.sha256(data).hexdigest()))
    return tuple(rows)


def _run_real_gate_action(binding, reuse_only, activity):
    workspace = binding["workspace"]
    marker = workspace / "mlruns" / ".aggregate-gate"
    if not marker.exists():
        marker.touch()
    scenario_marker = workspace / "data" / "aggregate_gate_scenario.json"
    if not scenario_marker.exists():
        scenario_marker.write_text(json.dumps({
            "protocol": GATE_PROTOCOL,
            "scenario_fingerprint": binding["scenario_fingerprint"],
        }, sort_keys=True), encoding="utf-8")
    try:
        marker_payload = json.loads(
            scenario_marker.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError):
        raise AggregateGateError("disposable scenario marker is invalid")
    if marker_payload != {
        "protocol": GATE_PROTOCOL,
        "scenario_fingerprint": binding["scenario_fingerprint"],
    }:
        raise AggregateGateError("disposable scenario marker is stale")
    from quantpits.utils.workspace import WorkspaceContext
    context = WorkspaceContext.from_root(
        workspace, qlib_data_dir=workspace / "qlib_data",
        qlib_region="cn",
    )
    if not str(context.mlflow_uri).startswith("file://"):
        raise AggregateGateError("fixture tracking backend is not file-contained")
    _initialize_fixture_tracking(context)
    activity.disable_qlib_repository_logging()
    scope, runtime_params = _build_fixture_scope(context)
    from quantpits.rolling import (
        QlibMlflowAggregateBackend,
        QlibMlflowExecutionBackend,
        RollingStateRepository,
        build_rolling_aggregate_scope,
        materialize_rolling_aggregate_candidates,
    )
    repository = RollingStateRepository.for_workspace(context, "rolling")
    source = _FixtureExecutionBackend(QlibMlflowExecutionBackend(context))
    state_view = repository.inspect_readonly()
    if state_view.inspection.classification == "missing":
        if reuse_only:
            raise AggregateGateError("reuse process found no fixture state")
        runner = _FixtureRunner(context, runtime_params, activity)
        state_view = _materialize_no_training_source_fixtures(
            scope, repository, source, runner,
        )
        if activity.runner_calls != 2:
            raise AggregateGateError("deterministic source fixture failed")
    elif not reuse_only:
        raise AggregateGateError("disposable workspace is stale")
    aggregate = build_rolling_aggregate_scope(
        scope, state_view, "gate-aggregate-attempt",
    )
    candidate_backend = QlibMlflowAggregateBackend(context)
    before_count = candidate_backend.inventory(aggregate)["raw_count"]
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, candidate_backend,
    )
    after_count = candidate_backend.inventory(aggregate)["raw_count"]
    if reuse_only:
        if (
            result.status != "success"
            or result.target_results[0].status != "reused_success"
            or after_count != before_count
            or activity.runner_calls != 0
        ):
            raise AggregateGateError("second-process reuse was not zero-write")
    elif (
        result.status != "success"
        or result.target_results[0].status != "materialized_success"
        or after_count != before_count + 1
    ):
        raise AggregateGateError("candidate materialization was not exact-one")
    candidate = result.target_results[0].candidate
    if candidate.row_count != 4:
        raise AggregateGateError("candidate row cardinality is unexpected")
    return candidate, before_count, after_count


def _run_real_gate(binding, reuse_only=False):
    started = time.monotonic()
    workspace = binding["workspace"]
    repository_root = Path(__file__).resolve().parents[2]
    write_before = _process_physical_write_bytes()
    declared_directories = (
        "config", "data", "data/locks",
        "data/aggregate_gate_runtime",
        "data/rolling_aggregate_candidates_rolling",
        "mlruns", "output", "qlib_data",
    )
    observers = []
    protected_observers = []
    protected_roots = _binding_protected_workspaces(binding)

    def close_observers():
        for active_observer in observers:
            active_observer.close()

    try:
        workspace_observer = _WorkspaceMutationObserver(workspace).start()
        observers.append(workspace_observer)
        for protected_root in protected_roots:
            observer = _WorkspaceMutationObserver(
                protected_root,
            ).start()
            protected_observers.append(observer)
            observers.append(observer)
        repository_observer = _WorkspaceMutationObserver(
            repository_root, (".git", "plan", "workspaces"),
        ).start()
        observers.append(repository_observer)
        workspace_before = snapshot_tree(workspace)
        for name in declared_directories:
            path = workspace / name
            path.mkdir(parents=True, exist_ok=True)
            try:
                path.resolve(strict=True).relative_to(workspace)
            except ValueError:
                raise AggregateGateError("fixture directory is not contained")
            if (
                path.is_symlink()
                or path.resolve(strict=True) != path.absolute()
            ):
                raise AggregateGateError("fixture directory is not contained")
    except BaseException:
        close_observers()
        raise
    try:
        protected_before = tuple(
            snapshot_tree(root, allow_symlinks=True)
            for root in protected_roots
        )
        repository_before = _snapshot_tracked_repository(
            repository_root,
        )
    except BaseException:
        close_observers()
        raise
    activity = _GateActivityObserver(
        workspace / "data" / "aggregate_gate_runtime",
    )
    try:
        with activity:
            candidate, before_count, after_count = _run_real_gate_action(
                binding, reuse_only, activity,
            )
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        close_observers()
        raise
    except BaseException:
        close_observers()
        raise
    try:
        _assert_snapshot_unchanged(
            protected_before,
            tuple(
                snapshot_tree(root, allow_symlinks=True)
                for root in protected_roots
            ),
            "protected workspaces",
        )
        _assert_snapshot_unchanged(
            repository_before,
            _snapshot_tracked_repository(repository_root),
            "tracked repository",
        )
        workspace_after = snapshot_tree(workspace)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        close_observers()
        raise
    except BaseException:
        close_observers()
        raise
    try:
        mutation_paths = workspace_observer.stop()
        protected_mutations = tuple(
            path
            for observer in protected_observers
            for path in observer.stop()
        )
        repository_mutations = repository_observer.stop()
    finally:
        close_observers()
    if protected_mutations:
        raise AggregateGateError(
            "protected workspace lifecycle observer detected a write"
        )
    if repository_mutations:
        raise AggregateGateError(
            "repository lifecycle observer detected a write"
        )
    write_bytes = max(
        0, _process_physical_write_bytes() - write_before,
    )
    if (
        activity.training_calls != 0
        or activity.gpu_calls != 0
        or activity.network_calls != 0
    ):
        raise AggregateGateError(
            "gate observed forbidden training, GPU, or network activity"
        )
    elapsed = time.monotonic() - started
    changed_file_count, write_bytes = _assert_workspace_write_allowlist(
        workspace_before, workspace_after, mutation_paths, write_bytes,
        reuse_only=reuse_only,
    )
    _assert_gate_budgets(elapsed, write_bytes)
    total_bytes = _workspace_bytes(workspace_after)
    return {
        "status": "reused_success" if reuse_only else "materialized_success",
        "candidate_fingerprint": candidate.content_fingerprint,
        "candidate_row_count": candidate.row_count,
        "new_candidate_recorders": after_count - before_count,
        "training_calls": activity.training_calls,
        "gpu_calls": activity.gpu_calls,
        "network_calls": activity.network_calls,
        "runner_calls": activity.runner_calls,
        "protected_unchanged": True,
        "repository_unchanged": True,
        "elapsed_seconds": round(elapsed, 6),
        "workspace_bytes": total_bytes,
        "write_bytes": write_bytes,
        "changed_path_count": changed_file_count,
    }


def _parse_internal_envelope(child, binding, reason_code):
    if child.returncode != 0:
        raise AggregateGateError("separate gate process failed")
    try:
        envelope = json.loads(child.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise AggregateGateError("separate gate process evidence is invalid")
    if (
        not isinstance(envelope, Mapping)
        or frozenset(envelope) != {
            "status", "reason_code", "protocol",
            "scenario_fingerprint", "commit", "tree", "result",
        }
        or envelope.get("status") != "gate_passed"
        or envelope.get("reason_code") != reason_code
        or envelope.get("protocol") != GATE_PROTOCOL
        or envelope.get("scenario_fingerprint")
        != binding["scenario_fingerprint"]
        or envelope.get("commit") != binding["commit"]
        or envelope.get("tree") != binding["tree"]
    ):
        raise AggregateGateError("separate gate process envelope failed")
    return envelope


def execute_gate(binding):
    gate_started = time.monotonic()
    command = [
        sys.executable, "-m",
        "quantpits.tools.verify_rolling_aggregate_candidate",
        "--workspace", str(binding["workspace"]),
        "--commit", binding["commit"], "--tree", binding["tree"],
        "--execute", "--authorization", EXECUTE_AUTHORIZATION,
    ]
    for protected_root in _binding_protected_workspaces(binding):
        command.extend((
            "--protected-workspace", str(protected_root),
        ))
    try:
        primary_child = subprocess.run(
            command + ["--internal-primary"],
            capture_output=True, text=True, timeout=WALL_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AggregateGateError(
            "primary process exceeded the gate wall-clock budget"
        ) from exc
    primary_envelope = _parse_internal_envelope(
        primary_child, binding, "rolling_aggregate_gate_primary_passed",
    )
    primary = _validate_run_result(
        primary_envelope["result"], reuse_only=False,
    )
    remaining_seconds = WALL_SECONDS - (time.monotonic() - gate_started)
    if remaining_seconds <= 0:
        raise AggregateGateError("aggregate gate wall-clock budget exceeded")
    try:
        child = subprocess.run(
            command + ["--internal-reuse"],
            capture_output=True, text=True, timeout=remaining_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise AggregateGateError(
            "reuse process exceeded the gate wall-clock budget"
        ) from exc
    reuse = _parse_internal_envelope(
        child, binding, "rolling_aggregate_gate_reuse_passed",
    )
    reuse_result = _validate_run_result(
        reuse["result"], reuse_only=True,
    )
    if (
        reuse_result.get("candidate_fingerprint")
        != primary.get("candidate_fingerprint")
        or reuse_result.get("candidate_row_count")
        != primary.get("candidate_row_count")
    ):
        raise AggregateGateError("separate-process reuse evidence failed")
    total_elapsed = time.monotonic() - gate_started
    _assert_gate_budgets(total_elapsed, primary["write_bytes"])
    return {
        "status": "gate_passed",
        "reason_code": "rolling_aggregate_gate_passed",
        "scenario_fingerprint": binding["scenario_fingerprint"],
        "commit": binding["commit"],
        "tree": binding["tree"],
        "result": primary,
        "reuse": reuse_result,
        "total_elapsed_seconds": round(total_elapsed, 6),
        "cleanup": "preserved",
    }


def _binding_protected_workspaces(binding):
    values = binding.get("protected_workspaces")
    if values is None:
        values = (binding["protected_workspace"],)
    return tuple(values)


def _path_contains(parent, child):
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _read_regular_child_bytes(directory_fd, name):
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(descriptor)
        public = os.stat(
            name, dir_fd=directory_fd, follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(public.st_mode)
            or opened.st_nlink != 1
            or public.st_nlink != 1
            or (opened.st_dev, opened.st_ino)
            != (public.st_dev, public.st_ino)
        ):
            raise AggregateGateError(
                "cleanup marker is not one canonical regular file"
            )
        chunks = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        completed = os.fstat(descriptor)
        terminal = os.stat(
            name, dir_fd=directory_fd, follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(terminal.st_mode)
            or terminal.st_nlink != 1
            or (
                completed.st_dev, completed.st_ino,
                completed.st_size, completed.st_mtime_ns,
                completed.st_ctime_ns, completed.st_nlink,
            ) != (
                opened.st_dev, opened.st_ino,
                opened.st_size, opened.st_mtime_ns,
                opened.st_ctime_ns, opened.st_nlink,
            )
            or (
                terminal.st_dev, terminal.st_ino,
                terminal.st_size, terminal.st_mtime_ns,
                terminal.st_ctime_ns, terminal.st_nlink,
            ) != (
                opened.st_dev, opened.st_ino,
                opened.st_size, opened.st_mtime_ns,
                opened.st_ctime_ns, opened.st_nlink,
            )
        ):
            raise AggregateGateError(
                "cleanup marker drifted during observation"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _delete_exact_tree_contents(directory_fd):
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for name in os.listdir(directory_fd):
        node = os.stat(
            name, dir_fd=directory_fd, follow_symlinks=False,
        )
        if stat.S_ISDIR(node.st_mode):
            child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
            try:
                opened = os.fstat(child_fd)
                identity = (opened.st_dev, opened.st_ino)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or identity != (node.st_dev, node.st_ino)
                ):
                    raise AggregateGateError(
                        "cleanup directory identity drifted"
                    )
                _delete_exact_tree_contents(child_fd)
                public = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(public.st_mode)
                    or (public.st_dev, public.st_ino) != identity
                ):
                    raise AggregateGateError(
                        "cleanup directory public name drifted"
                    )
                os.rmdir(name, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(node.st_mode) and node.st_nlink == 1:
            os.unlink(name, dir_fd=directory_fd)
        else:
            raise AggregateGateError(
                "cleanup tree contains an aliased or special node"
            )


def _validate_exact_tree(directory_fd):
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for name in os.listdir(directory_fd):
        try:
            node = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False,
            )
        except OSError as exc:
            raise AggregateGateError(
                "cleanup tree inventory is unavailable"
            ) from exc
        if stat.S_ISREG(node.st_mode):
            if node.st_nlink != 1:
                raise AggregateGateError(
                    "cleanup tree contains an aliased or special node"
                )
            continue
        if not stat.S_ISDIR(node.st_mode):
            raise AggregateGateError(
                "cleanup tree contains an aliased or special node"
            )
        try:
            child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
        except OSError as exc:
            raise AggregateGateError(
                "cleanup directory identity drifted"
            ) from exc
        try:
            opened = os.fstat(child_fd)
            identity = (opened.st_dev, opened.st_ino)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or identity != (node.st_dev, node.st_ino)
            ):
                raise AggregateGateError(
                    "cleanup directory identity drifted"
                )
            _validate_exact_tree(child_fd)
            public = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(public.st_mode)
                or (public.st_dev, public.st_ino) != identity
            ):
                raise AggregateGateError(
                    "cleanup directory public name drifted"
                )
        finally:
            os.close(child_fd)


def cleanup_gate_workspace(
    workspace, protected_workspace, scenario_fingerprint, authorization,
):
    if authorization != CLEANUP_AUTHORIZATION:
        raise AggregateGateError("cleanup authorization is missing or invalid")
    workspace = _real_directory(workspace, "cleanup workspace")
    protected_values = (
        tuple(protected_workspace)
        if isinstance(protected_workspace, (tuple, list))
        else (protected_workspace,)
    )
    protected = tuple(_real_directory(
        value, "protected workspace", allow_linked_entry=True,
    ) for value in protected_values)
    repository = Path(__file__).resolve().parents[2]
    if (
        workspace == workspace.parent
        or any(
            _path_contains(workspace, boundary)
            or _path_contains(boundary, workspace)
            for boundary in protected + (repository,)
        )
    ):
        raise AggregateGateError("cleanup target is protected or broad")
    parent = workspace.parent
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(str(parent), directory_flags)
    workspace_fd = None
    try:
        parent_node = os.fstat(parent_fd)
        parent_public = os.lstat(str(parent))
        if (
            not stat.S_ISDIR(parent_node.st_mode)
            or not stat.S_ISDIR(parent_public.st_mode)
            or (parent_node.st_dev, parent_node.st_ino)
            != (parent_public.st_dev, parent_public.st_ino)
        ):
            raise AggregateGateError("cleanup parent identity drifted")
        workspace_public = os.stat(
            workspace.name, dir_fd=parent_fd, follow_symlinks=False,
        )
        workspace_fd = os.open(
            workspace.name, directory_flags, dir_fd=parent_fd,
        )
        workspace_node = os.fstat(workspace_fd)
        workspace_identity = (
            workspace_node.st_dev, workspace_node.st_ino,
        )
        if (
            not stat.S_ISDIR(workspace_node.st_mode)
            or workspace_identity
            != (workspace_public.st_dev, workspace_public.st_ino)
        ):
            raise AggregateGateError("cleanup target identity drifted")
        try:
            data_fd = os.open(
                "data", directory_flags, dir_fd=workspace_fd,
            )
        except OSError as exc:
            raise AggregateGateError(
                "cleanup scenario marker is unavailable"
            ) from exc
        try:
            try:
                marker_bytes = _read_regular_child_bytes(
                    data_fd, "aggregate_gate_scenario.json",
                )
            except OSError as exc:
                raise AggregateGateError(
                    "cleanup scenario marker is unavailable"
                ) from exc
        finally:
            os.close(data_fd)
        try:
            payload = json.loads(marker_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise AggregateGateError(
                "cleanup scenario marker is unavailable"
            ) from exc
        if payload != {
            "protocol": GATE_PROTOCOL,
            "scenario_fingerprint": scenario_fingerprint,
        }:
            raise AggregateGateError(
                "cleanup scenario identity disagrees"
            )
        current = os.stat(
            workspace.name, dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != workspace_identity
        ):
            raise AggregateGateError(
                "cleanup target drifted before deletion"
            )
        _validate_exact_tree(workspace_fd)
        _delete_exact_tree_contents(workspace_fd)
        current = os.stat(
            workspace.name, dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != workspace_identity
        ):
            raise AggregateGateError(
                "cleanup target drifted before root removal"
            )
        os.rmdir(workspace.name, dir_fd=parent_fd)
    finally:
        if workspace_fd is not None:
            os.close(workspace_fd)
        os.close(parent_fd)
    if workspace.exists():
        raise AggregateGateError("cleanup postcondition is uncertain")
    return {
        "status": "cleanup_completed",
        "reason_code": "rolling_aggregate_gate_cleanup_completed",
        "scenario_fingerprint": scenario_fingerprint,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="Preflight the bounded immutable Rolling aggregate gate",
    )
    parser.add_argument("--scenario")
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--protected-workspace", required=True, action="append",
    )
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--internal-reuse", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--internal-primary", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--cleanup-authorization")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.cleanup and (
            args.execute or args.internal_reuse or args.internal_primary
        ):
            raise AggregateGateError(
                "cleanup and execute/reuse modes are mutually exclusive"
            )
        if (args.internal_reuse or args.internal_primary) and not args.execute:
            raise AggregateGateError(
                "internal execution requires explicit execute mode"
            )
        if args.internal_reuse and args.internal_primary:
            raise AggregateGateError("internal execution modes are exclusive")
        if args.scenario:
            payload = json.loads(
                Path(args.scenario).read_text(encoding="utf-8"),
            )
            scenario = scenario_from_mapping(payload)
        else:
            scenario = frozen_scenario()
        binding = validate_binding(
            scenario, args.workspace, args.protected_workspace,
            args.commit, args.tree,
            args.execute or args.internal_reuse or args.internal_primary,
            args.authorization,
        )
        evidence = (
            cleanup_gate_workspace(
                binding["workspace"],
                _binding_protected_workspaces(binding),
                binding["scenario_fingerprint"],
                args.cleanup_authorization,
            )
            if args.cleanup
            else
            {
                "status": "gate_passed",
                "reason_code": "rolling_aggregate_gate_reuse_passed",
                "protocol": GATE_PROTOCOL,
                "scenario_fingerprint": binding["scenario_fingerprint"],
                "commit": binding["commit"],
                "tree": binding["tree"],
                "result": _validate_run_result(
                    _run_real_gate(binding, reuse_only=True),
                    reuse_only=True,
                ),
            }
            if args.internal_reuse
            else
            {
                "status": "gate_passed",
                "reason_code": "rolling_aggregate_gate_primary_passed",
                "protocol": GATE_PROTOCOL,
                "scenario_fingerprint": binding["scenario_fingerprint"],
                "commit": binding["commit"],
                "tree": binding["tree"],
                "result": _validate_run_result(
                    _run_real_gate(binding, reuse_only=False),
                    reuse_only=False,
                ),
            }
            if args.internal_primary
            else execute_gate(binding) if args.execute
            else preflight_evidence(binding)
        )
        print(json.dumps(evidence, sort_keys=True))
        return 0
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        print(json.dumps({
            "protocol": GATE_PROTOCOL,
            "status": "blocked",
            "reason_code": "rolling_aggregate_gate_blocked",
            "error": exc.__class__.__name__,
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
