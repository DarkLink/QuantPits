import pytest
import os

import quantpits.evidence.inspection as inspection
from quantpits.evidence.inspection import (
    PathBoundaryError, SourceMutationObserver, inspect_file, inspect_many,
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
