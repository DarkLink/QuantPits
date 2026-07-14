"""Coherent file baselines and durable local filesystem helpers."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class FileBaseline:
    path: str
    existed: bool
    fingerprint: Optional[str]
    size_bytes: Optional[int]

    def to_dict(self):
        return {
            "path": self.path,
            "existed": self.existed,
            "fingerprint": self.fingerprint,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            path=str(value["path"]),
            existed=bool(value["existed"]),
            fingerprint=value.get("fingerprint"),
            size_bytes=value.get("size_bytes"),
        )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_with_baseline(path: Path, *, display_path: Optional[str] = None) -> Tuple[Optional[bytes], FileBaseline]:
    """Read one coherent version and derive its baseline from those bytes."""

    path = Path(path)
    public_path = display_path or path.name
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, FileBaseline(public_path, False, None, None)
    return raw, FileBaseline(public_path, True, sha256_bytes(raw), len(raw))


def baseline_matches(path: Path, baseline: FileBaseline) -> bool:
    _raw, observed = read_with_baseline(path, display_path=baseline.path)
    return (
        observed.existed == baseline.existed
        and observed.fingerprint == baseline.fingerprint
        and observed.size_bytes == baseline.size_bytes
    )


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json_bytes(path: Path, payload: bytes) -> None:
    if not payload.endswith(b"\n"):
        raise ValueError("durable JSON payload must end with a newline")
    atomic_write_bytes(path, payload)
