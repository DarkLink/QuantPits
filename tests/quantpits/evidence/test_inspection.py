import pytest
import os

import quantpits.evidence.inspection as inspection
from quantpits.evidence.inspection import (
    PathBoundaryError, SourceMutationObserver, inspect_file, inspect_many,
    inspect_tree,
)


def test_symlink_and_parent_escape_are_blocked(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (root / "alias").symlink_to(outside)
    with pytest.raises(PathBoundaryError):
        inspect_file(root, "alias")
    with pytest.raises(PathBoundaryError):
        inspect_file(root, "../outside.txt")
    with pytest.raises(PathBoundaryError):
        inspect_file(root, "nested//source")


def test_internal_symlink_alias_is_blocked_even_when_target_is_contained(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "target"
    target.write_text("evidence")
    (root / "alias").symlink_to(target)

    with pytest.raises(PathBoundaryError, match="symlink"):
        inspect_file(root, "alias")


def test_parent_replacement_during_open_cannot_redirect_source_read(
    tmp_path, monkeypatch,
):
    root = tmp_path / "workspace"
    parent = root / "evidence"
    parent.mkdir(parents=True)
    (parent / "source.json").write_text("original")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.json").write_text("private-outside")
    displaced = root / "displaced"
    original_open = inspection.os.open
    replaced = False

    def replace_parent_before_final_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        if path == "source.json" and dir_fd is not None and not replaced:
            parent.rename(displaced)
            parent.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(inspection.os, "open", replace_parent_before_final_open)
    with pytest.raises(PathBoundaryError):
        inspect_file(root, "evidence/source.json")
    assert replaced is True


def test_tree_inventory_retains_special_nodes_as_incomparable(tmp_path):
    root = tmp_path / "workspace"
    tree = root / "tree"
    tree.mkdir(parents=True)
    os.mkfifo(str(tree / "unexpected"))

    members = inspect_tree(root, "tree")
    assert len(members) == 1
    assert members[0].status == "incomparable"
    assert "special node" in members[0].detail


def test_hardlink_source_is_blocked_as_ambiguous_physical_ownership(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "source"
    source.write_text("evidence")
    os.link(str(source), str(root / "alias"))
    with pytest.raises(PathBoundaryError):
        inspect_file(root, "alias")


def test_source_observer_detects_transient_move_away_and_back(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "source"
    displaced = root / "displaced"
    source.write_text("evidence")
    with SourceMutationObserver(root, ("source",)) as observer:
        if not observer.supported:
            pytest.skip("inotify is unavailable")
        source.rename(displaced)
        displaced.rename(source)
        assert observer.mutated() is True


def test_source_observer_detects_transient_creation_under_missing_nested_path(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    with SourceMutationObserver(root, ("missing/nested/source",)) as observer:
        if not observer.supported:
            pytest.skip("inotify is unavailable")
        nested = root / "missing"
        nested.mkdir()
        nested.rmdir()
        assert observer.mutated() is True


def test_git_observer_derives_clean_identity_and_raw_inventory_digests(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()

    def fake_git(_repo, *args):
        command = tuple(args)
        if command == ("rev-parse", "--show-toplevel"):
            return str(root).encode() + b"\n"
        if command == ("rev-parse", "HEAD"):
            return b"a" * 40 + b"\n"
        if command == ("rev-parse", "HEAD^{tree}"):
            return b"b" * 40 + b"\n"
        if command[:2] == ("status", "--porcelain=v1"):
            return b""
        if command[:2] == ("diff", "--binary"):
            return b""
        raise RuntimeError("no upstream")

    monkeypatch.setattr(inspection, "_git", fake_git)
    result = inspection.inspect_git(root)
    assert result["status"] == "clean"
    assert result["commit"] == "a" * 40
    assert result["status_inventory_digest"]["domain"] == "raw_bytes"
    assert result["remote_relation"] == "no_upstream"


def test_git_observer_failure_diagnostic_does_not_expose_absolute_path(
    tmp_path, monkeypatch,
):
    root = tmp_path / "private-workspace"
    root.mkdir()

    def fail(_repo, *_args):
        raise OSError(5, "synthetic", str(root))

    monkeypatch.setattr(inspection, "_git", fail)
    result = inspection.inspect_git(root)
    assert result["status"] == "dirty_unresolved"
    assert str(root) not in result["detail"]
    assert result["detail"] == "OSError(errno=5)"


def test_git_observer_rechecks_status_and_diff_for_one_observation(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    status_calls = 0

    def changing_git(_repo, *args):
        nonlocal status_calls
        command = tuple(args)
        if command == ("rev-parse", "--show-toplevel"):
            return str(root).encode() + b"\n"
        if command == ("rev-parse", "HEAD"):
            return b"a" * 40 + b"\n"
        if command == ("rev-parse", "HEAD^{tree}"):
            return b"b" * 40 + b"\n"
        if command[:2] == ("status", "--porcelain=v1"):
            status_calls += 1
            return b"" if status_calls == 1 else b"?? changed\0"
        if command[:2] == ("diff", "--binary"):
            return b""
        raise RuntimeError("no upstream")

    monkeypatch.setattr(inspection, "_git", changing_git)
    assert inspection.inspect_git(root) == {
        "status": "dirty_unresolved",
        "detail": "Git identity changed during observation",
    }


def test_ordinary_member_failure_preserves_later_identity_order_and_cardinality(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "first").write_text("1")
    (root / "last").write_text("2")
    original = inspection.inspect_file

    def faulty(workspace, path):
        if path == "first":
            raise OSError("synthetic failure")
        return original(workspace, path)

    monkeypatch.setattr(inspection, "inspect_file", faulty)
    result = inspect_many(root, (("one", "first"), ("two", "last")))
    assert [name for name, _item in result] == ["one", "two"]
    assert [item.status for _name, item in result] == ["incomparable", "observed"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_interrupt_propagates(tmp_path, monkeypatch, interrupt):
    root = tmp_path / "workspace"
    root.mkdir()

    def stop(_workspace, _path):
        raise interrupt()

    monkeypatch.setattr(inspection, "inspect_file", stop)
    with pytest.raises(interrupt):
        inspect_many(root, (("one", "anything"),))
