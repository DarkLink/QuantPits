"""Independent filesystem observer for semantic scenarios."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ObservedArtifactGraph:
    root: Path
    files: Dict[str, str]
    physical_escapes: Tuple[str, ...]

    def changed_paths(self, baseline):
        keys = set(self.files) | set(baseline.files)
        return tuple(sorted(key for key in keys if self.files.get(key) != baseline.files.get(key)))

    def json(self, relative):
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def jsonl(self, relative):
        path = self.root / relative
        return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def observe_artifact_graph(root):
    root = Path(root).resolve()
    files = {}
    escapes = []
    if not root.exists():
        return ObservedArtifactGraph(root, files, ())
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(root)
        except ValueError:
            escapes.append(relative)
        if path.is_file() and not path.is_symlink():
            files[relative] = _sha256(path)
    return ObservedArtifactGraph(root, files, tuple(escapes))


def assert_declared_writes(changed_paths: Iterable[str], allowed_prefixes: Iterable[str]):
    prefixes = tuple(allowed_prefixes)
    unexpected = [path for path in changed_paths if not any(path == item or path.startswith(item) for item in prefixes)]
    assert not unexpected, "undeclared semantic writes: %s" % unexpected
