import pytest

from .artifact_graph import (
    assert_declared_writes,
    assert_write_conservation,
    observe_artifact_graph,
)


def test_exact_file_declaration_rejects_prefix_collision(tmp_path):
    with pytest.raises(AssertionError, match="operator_log.jsonl.evil"):
        assert_declared_writes(
            ("data/operator_log.jsonl.evil",),
            ("data/operator_log.jsonl",),
        )


def test_observer_detects_empty_directory_delta(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    baseline = observe_artifact_graph(root)
    (root / "unexpected" / "empty").mkdir(parents=True)
    observed = observe_artifact_graph(root)

    assert observed.changed_paths(baseline) == ("unexpected", "unexpected/empty")
    with pytest.raises(AssertionError, match="unexpected"):
        assert_write_conservation(observed, baseline, ())


def test_observer_rejects_external_symlink_even_when_path_is_declared(tmp_path):
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    baseline = observe_artifact_graph(root)
    (root / "output").symlink_to(outside, target_is_directory=True)
    (root / "output" / "escaped.txt").write_text("escaped", encoding="utf-8")
    observed = observe_artifact_graph(root)

    assert (outside / "escaped.txt").read_text(encoding="utf-8") == "escaped"
    assert observed.artifacts["output"].kind == "symlink"
    assert observed.physical_escapes == ("output",)
    with pytest.raises(AssertionError, match="physical workspace escapes"):
        assert_write_conservation(observed, baseline, ("output/",))


def test_repository_observer_covers_arbitrary_nested_subdirectory(tmp_path):
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    baseline = observe_artifact_graph(repository, excluded_roots=(".git/",))
    path = repository / "arbitrary" / "nested" / "leak.txt"
    path.parent.mkdir(parents=True)
    path.write_text("leak", encoding="utf-8")
    observed = observe_artifact_graph(repository, excluded_roots=(".git/",))

    assert observed.changed_paths(baseline) == (
        "arbitrary", "arbitrary/nested", "arbitrary/nested/leak.txt",
    )
