"""Workspace runtime context and stable fingerprint helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _resolve_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def _has_legacy_mlruns_data(workspace_root: Path) -> bool:
    mlruns = workspace_root / "mlruns"
    if not mlruns.is_dir():
        return False
    return any(entry.name != ".gitkeep" for entry in mlruns.iterdir())


def _default_mlflow_uri(workspace_root: Path) -> str:
    if _has_legacy_mlruns_data(workspace_root):
        return f"file://{workspace_root / 'mlruns'}"
    return f"sqlite:///{workspace_root / 'mlflow.db'}"


@dataclass(frozen=True)
class WorkspaceContext:
    """Explicit workspace runtime paths and lightweight metadata."""

    root: Path
    config_dir: Path
    data_dir: Path
    output_dir: Path
    mlruns_dir: Path
    mlflow_uri: str
    qlib_data_dir: Path | None = None
    qlib_region: str | None = None

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        mlflow_uri: str | None = None,
        qlib_data_dir: str | Path | None = None,
        qlib_region: str | None = None,
    ) -> "WorkspaceContext":
        workspace_root = _resolve_root(root)
        qlib_path = (
            Path(qlib_data_dir).expanduser().resolve()
            if qlib_data_dir is not None
            else Path(os.environ.get("QLIB_DATA_DIR", "~/.qlib/qlib_data/cn_data")).expanduser().resolve()
        )
        return cls(
            root=workspace_root,
            config_dir=workspace_root / "config",
            data_dir=workspace_root / "data",
            output_dir=workspace_root / "output",
            mlruns_dir=workspace_root / "mlruns",
            mlflow_uri=mlflow_uri or _default_mlflow_uri(workspace_root),
            qlib_data_dir=qlib_path,
            qlib_region=qlib_region or os.environ.get("QLIB_REGION", "cn"),
        )

    @classmethod
    def from_env(cls) -> "WorkspaceContext":
        from quantpits.utils import env

        return env.get_workspace_context()

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def config_path(self, *parts: str) -> Path:
        return self.config_dir.joinpath(*parts)

    def data_path(self, *parts: str) -> Path:
        return self.data_dir.joinpath(*parts)

    def output_path(self, *parts: str) -> Path:
        return self.output_dir.joinpath(*parts)


def _normalize_for_json(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize_for_json(dataclasses.asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_for_json(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    return value


def stable_json_dumps(value: Any) -> str:
    """Serialize values deterministically for metadata fingerprints."""

    normalized = _normalize_for_json(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint_value(value: Any, *, algorithm: str = "sha256") -> str:
    payload = stable_json_dumps(value).encode("utf-8")
    hasher = hashlib.new(algorithm)
    hasher.update(payload)
    return hasher.hexdigest()


def short_fingerprint(value: Any, *, length: int = 12, algorithm: str = "sha256") -> str:
    return fingerprint_value(value, algorithm=algorithm)[:length]


def fingerprint_file(path: str | Path, *, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
