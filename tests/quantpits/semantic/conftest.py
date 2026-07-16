"""Semantic-lane marker and isolation checks."""

import hashlib
import warnings
from pathlib import Path

import pytest


_REPOSITORY_BASELINE = None


def _repository_snapshot(repository):
    roots = [
        repository / name
        for name in ("quantpits", "tests", "docs", ".pytest_cache", "__pycache__", "mlruns", "output")
    ]
    files = [path for path in repository.iterdir() if path.is_file()]
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return {
        path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(files))
    }


def _semantic_only(config):
    args = tuple(str(value).replace("\\", "/") for value in config.args)
    return bool(args) and all("tests/quantpits/semantic" in value for value in args)


def pytest_collection_finish(session):
    global _REPOSITORY_BASELINE
    if _semantic_only(session.config):
        _REPOSITORY_BASELINE = _repository_snapshot(Path(str(session.config.rootpath)).resolve())


def pytest_sessionfinish(session, exitstatus):
    del exitstatus
    if _REPOSITORY_BASELINE is None:
        return
    repository = Path(str(session.config.rootpath)).resolve()
    after = _repository_snapshot(repository)
    if after != _REPOSITORY_BASELINE:
        changed = sorted(set(after) ^ set(_REPOSITORY_BASELINE))
        changed.extend(
            key for key in set(after) & set(_REPOSITORY_BASELINE)
            if after[key] != _REPOSITORY_BASELINE[key]
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        warnings.warn(
            pytest.PytestWarning("semantic lane wrote repository files: %s" % sorted(set(changed)))
        )


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/quantpits/semantic/"):
            item.add_marker(pytest.mark.semantic)


@pytest.fixture(autouse=True)
def semantic_repository_guard(request, monkeypatch, tmp_path):
    """Reject the historical root diagnostic leak for every semantic test."""

    repository = Path(str(request.config.rootpath)).resolve()
    diagnostic = repository / "unmocked_mlruns.log"
    before = diagnostic.read_bytes() if diagnostic.exists() else None
    environment_root = tmp_path / "semantic_environment"
    for relative in ("config", "data", "output"):
        (environment_root / relative).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(environment_root))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    yield
    after = diagnostic.read_bytes() if diagnostic.exists() else None
    assert after == before
