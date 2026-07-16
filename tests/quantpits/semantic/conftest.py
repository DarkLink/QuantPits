"""Semantic-lane marker and isolation checks."""

import warnings
from pathlib import Path

import pytest

from .artifact_graph import observe_artifact_graph


_REPOSITORY_BASELINE = None


def _repository_snapshot(repository):
    # The repository guard covers every source/document/build path. Private,
    # uncommitted workspaces are an explicit non-observation boundary.
    return observe_artifact_graph(repository, excluded_roots=(".git/", "workspaces/"))


def _semantic_only(config):
    args = tuple(str(value).replace("\\", "/") for value in config.args)
    return bool(args) and all("tests/quantpits/semantic" in value for value in args)


def pytest_configure(config):
    """Snapshot before collection imports semantic tests and production modules."""

    global _REPOSITORY_BASELINE
    if _semantic_only(config):
        _REPOSITORY_BASELINE = _repository_snapshot(Path(str(config.rootpath)).resolve())


def pytest_sessionfinish(session, exitstatus):
    del exitstatus
    if _REPOSITORY_BASELINE is None:
        return
    repository = Path(str(session.config.rootpath)).resolve()
    after = _repository_snapshot(repository)
    changed = after.changed_paths(_REPOSITORY_BASELINE)
    if changed or after.physical_escapes != _REPOSITORY_BASELINE.physical_escapes:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        warnings.warn(
            pytest.PytestWarning("semantic lane wrote repository paths: %s" % list(changed))
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
