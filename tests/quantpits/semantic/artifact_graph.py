"""Independent, symlink-aware filesystem observer for semantic scenarios."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ObservedArtifact:
    kind: str
    fingerprint: str
    link_target: str = ""


@dataclass(frozen=True)
class ObservedArtifactGraph:
    root: Path
    artifacts: Dict[str, ObservedArtifact]
    physical_escapes: Tuple[str, ...]

    @property
    def files(self):
        return {
            path: artifact.fingerprint
            for path, artifact in self.artifacts.items()
            if artifact.kind == "file"
        }

    def changed_paths(self, baseline):
        keys = set(self.artifacts) | set(baseline.artifacts)
        changed = {
            key for key in keys
            if self.artifacts.get(key) != baseline.artifacts.get(key)
        }
        changed.update(set(self.physical_escapes) ^ set(baseline.physical_escapes))
        return tuple(sorted(changed))

    def json(self, relative):
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def jsonl(self, relative):
        path = self.root / relative
        return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _is_excluded(relative, excluded_roots):
    return any(
        relative == item.rstrip("/") or relative.startswith(item.rstrip("/") + "/")
        for item in excluded_roots
    )


def _walk_paths(root, excluded_roots):
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(str(directory)) as entries:
            children = sorted(entries, key=lambda item: item.name, reverse=True)
        for entry in children:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if _is_excluded(relative, excluded_roots):
                continue
            yield path
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)


def observe_artifact_graph(root, *, excluded_roots=()):
    root = Path(root).resolve()
    artifacts = {}
    escapes = []
    if not root.exists():
        return ObservedArtifactGraph(root, artifacts, ())
    for path in _walk_paths(root, tuple(excluded_roots)):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(str(path))
            artifact = ObservedArtifact("symlink", hashlib.sha256(target.encode("utf-8")).hexdigest(), target)
        elif path.is_dir():
            artifact = ObservedArtifact("directory", "")
        elif path.is_file():
            artifact = ObservedArtifact("file", _sha256(path))
        else:
            artifact = ObservedArtifact("other", "")
        artifacts[relative] = artifact
        try:
            path.resolve(strict=False).relative_to(root)
        except (OSError, ValueError):
            escapes.append(relative)
    return ObservedArtifactGraph(root, artifacts, tuple(sorted(escapes)))


def _is_allowed(path, declaration):
    if declaration.endswith("/"):
        directory = declaration.rstrip("/")
        return path == directory or path.startswith(directory + "/")
    return path == declaration


def assert_declared_writes(changed_paths: Iterable[str], allowed_paths: Iterable[str]):
    declarations = tuple(allowed_paths)
    unexpected = [
        path for path in changed_paths
        if not any(_is_allowed(path, declaration) for declaration in declarations)
    ]
    assert not unexpected, "undeclared semantic writes: %s" % unexpected


def assert_write_conservation(observed, baseline, allowed_paths):
    assert not observed.physical_escapes, "physical workspace escapes: %s" % (observed.physical_escapes,)
    assert_declared_writes(observed.changed_paths(baseline), allowed_paths)
