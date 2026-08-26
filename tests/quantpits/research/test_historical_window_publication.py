from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path

import pytest

import quantpits.research.historical_window_publication as publication
from quantpits.research.historical_cycle import ARM_IDS
from quantpits.research.historical_window_publication import (
    HistoricalWindowPublicationContractError,
    WindowPublicationReceipt,
    build_historical_window_artifacts,
    write_historical_window_output,
)
from tests.quantpits.research.test_historical_window import build_window


def _root(tmp_path, suffix="one"):
    return Path("/tmp") / ("quantpits_b3b_%s_%s" % (tmp_path.name, suffix))


def _clean(*roots):
    for root in roots:
        shutil.rmtree(str(root), ignore_errors=True)


@pytest.mark.parametrize("window_size", [4, 5, 6])
def test_four_five_six_window_artifact_sets_are_exact_and_deterministic(tmp_path, window_size):
    runner = build_window(tmp_path, window_size)[0]
    result = runner.run()
    first = build_historical_window_artifacts(runner, result)
    second = build_historical_window_artifacts(runner, result)
    assert tuple(first) == tuple(second)
    assert first == second
    assert len(first) == 4 + window_size * 5


def test_artifacts_join_exact_runner_result_and_ranking_digests_before_create(tmp_path):
    runner = build_window(tmp_path, 4)[0]
    result = runner.run()
    assert len(build_historical_window_artifacts(runner, result)) == 24
    result._payload["terminal_cycles"][0]["terminal_arms"][0]["ranking_digest"] = "0" * 64
    with pytest.raises(HistoricalWindowPublicationContractError):
        build_historical_window_artifacts(runner, result)


def test_private_result_and_rankings_are_exact_while_operator_views_are_normalized_only(tmp_path):
    runner = build_window(tmp_path, 4)[0]
    result = runner.run()
    members = build_historical_window_artifacts(runner, result)
    assert members["result.json"] == result.to_canonical_json_bytes()
    safe = members["summary.json"] + members["metrics.csv"] + members["report.md"]
    assert b'"cash"' not in safe and b'"positions"' not in safe and b'"orders"' not in safe
    assert b"promotion capability" in members["report.md"].lower()


def test_fresh_direct_tmp_publication_is_manifest_last_exact_and_committed(tmp_path):
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    root = _root(tmp_path)
    _clean(root)
    try:
        receipt = write_historical_window_output(runner, result, root)
        assert receipt.status == "COMMITTED" and receipt.member_count == 24
        manifest = json.loads((root / "output_manifest.json").read_text())
        assert manifest["member_count"] == 24 and len(manifest["members"]) == 24
        assert set(manifest) == {
            "schema_version", "bundle_kind", "evidence_class", "prospective_claim",
            "promotion_capability", "window_result_digest", "requested_anchor_ids",
            "requested_arm_ids", "member_count", "members",
        }
        assert [item["logical_path"] for item in manifest["members"]] == sorted(
            item["logical_path"] for item in manifest["members"]
        )
        for item in manifest["members"]:
            data = (root / item["logical_path"]).read_bytes()
            assert item["size_bytes"] == len(data)
            assert item["digest"] == publication._raw_digest(data)
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in root.rglob("*") if path.is_file())
        assert set(path.name for path in root.iterdir()) == {
            "result.json", "summary.json", "metrics.csv", "report.md", "rankings", "output_manifest.json",
        }
        before = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        conflict = write_historical_window_output(runner, result, root)
        after = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        assert conflict.status == "CONFLICT" and before == after
    finally:
        _clean(root)


def test_blocked_foreign_or_tampered_result_is_denied_before_root_create(tmp_path):
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    root = _root(tmp_path)
    result._payload["metric_capability"] = False
    with pytest.raises(HistoricalWindowPublicationContractError):
        write_historical_window_output(runner, result, root)
    assert not root.exists()
    with pytest.raises(HistoricalWindowPublicationContractError):
        build_historical_window_artifacts(object(), result)


def test_existing_root_is_conflict_unread_and_unchanged(tmp_path):
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    root = _root(tmp_path)
    _clean(root)
    root.mkdir(mode=0o700)
    marker = root / "private"
    marker.write_bytes(b"untouched")
    try:
        receipt = write_historical_window_output(runner, result, root)
        assert receipt.status == "CONFLICT" and receipt.did_write is False
        assert marker.read_bytes() == b"untouched" and set(root.iterdir()) == {marker}
    finally:
        _clean(root)


def test_relative_nested_workspace_and_symlink_parent_are_denied_before_write(tmp_path):
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    for root in (Path("relative"), tmp_path / "nested", Path("/tmp") / "a" / "nested"):
        with pytest.raises(HistoricalWindowPublicationContractError, match="direct child"):
            write_historical_window_output(runner, result, root)
    alias = Path("/tmp") / ("quantpits_b3b_alias_%s" % tmp_path.name)
    alias.symlink_to(tmp_path, target_is_directory=True)
    try:
        with pytest.raises(HistoricalWindowPublicationContractError, match="direct child"):
            write_historical_window_output(runner, result, alias / "nested")
        assert not (tmp_path / "nested").exists()
    finally:
        alias.unlink()
    assert not (tmp_path / "nested").exists()


def test_member_failure_is_uncertain_with_exact_observed_prefix_and_no_manifest(tmp_path, monkeypatch):
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    root = _root(tmp_path)
    _clean(root)
    original = publication._write_file
    calls = []
    def fail_second(fd, name, data):
        calls.append(name)
        if len(calls) == 2:
            raise OSError("private")
        return original(fd, name, data)
    monkeypatch.setattr(publication, "_write_file", fail_second)
    try:
        receipt = write_historical_window_output(runner, result, root)
        assert receipt.status == "UNCERTAIN" and receipt.member_count == 1
        assert not (root / "output_manifest.json").exists() and len(calls) == 2
    finally:
        _clean(root)


def test_manifest_or_final_member_failure_is_uncertain_without_commit_capability(tmp_path, monkeypatch):
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    root = _root(tmp_path)
    _clean(root)
    original = publication._read_exact_regular
    def fail_manifest(path, logical):
        if logical == "output_manifest.json":
            raise OSError("private")
        return original(path, logical)
    monkeypatch.setattr(publication, "_read_exact_regular", fail_manifest)
    try:
        receipt = write_historical_window_output(runner, result, root)
        assert receipt.status == "UNCERTAIN" and receipt.member_count == 24
        assert receipt.manifest_digest is None and (root / "output_manifest.json").exists()
    finally:
        _clean(root)


@pytest.mark.parametrize("target", ["parent", "root", "rankings"])
def test_parent_identity_drift_and_root_or_directory_replacement_are_uncertain(tmp_path, monkeypatch, target):
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    root, displaced = _root(tmp_path), _root(tmp_path, "displaced")
    _clean(root, displaced)
    original = publication._inventory
    changed = {"done": False}
    def replace_root(fd, expected, logical):
        if not changed["done"]:
            changed["done"] = True
            victim = root if target == "root" else root / "rankings"
            moved = displaced if target == "root" else root / "rankings-displaced"
            victim.rename(moved)
            victim.mkdir(mode=0o700)
        return original(fd, expected, logical)
    monkeypatch.setattr(publication, "_inventory", replace_root)
    if target == "parent":
        original_identity = publication._root_identity
        parent_calls = {"count": 0}
        def drift_parent(path):
            identity = original_identity(path)
            if Path(path) == Path("/tmp"):
                parent_calls["count"] += 1
                if parent_calls["count"] == 2:
                    return (identity[0], identity[1] + 1, identity[2], 0, 0)
            return identity
        monkeypatch.setattr(publication, "_root_identity", drift_parent)
    try:
        receipt = write_historical_window_output(runner, result, root)
        assert receipt.status == "UNCERTAIN"
        if target in ("parent", "root"):
            assert receipt.output_root_identity is None
    finally:
        _clean(root, displaced)


@pytest.mark.parametrize("mutation", ["missing", "symlink", "hardlink", "fifo", "foreign"])
def test_missing_foreign_symlink_hardlink_and_special_final_members_are_uncertain(tmp_path, monkeypatch, mutation):
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    root = _root(tmp_path)
    _clean(root)
    original_inventory = publication._inventory
    original_read = publication._read_exact_regular
    injected = {"done": False}
    def add_foreign(fd, expected, logical):
        if mutation == "foreign" and not injected["done"]:
            injected["done"] = True
            (root / "foreign").write_bytes(b"x")
        return original_inventory(fd, expected, logical)
    reads = {"result.json": 0}
    def mutate_member(path, logical):
        if logical == "result.json":
            reads[logical] += 1
            if mutation != "foreign" and reads[logical] == 2:
                target = root / logical
                target.unlink()
                if mutation == "symlink":
                    target.symlink_to(root / "summary.json")
                elif mutation == "hardlink":
                    os.link(str(root / "summary.json"), str(target))
                elif mutation == "fifo":
                    os.mkfifo(str(target), 0o600)
        return original_read(path, logical)
    monkeypatch.setattr(publication, "_inventory", add_foreign)
    monkeypatch.setattr(publication, "_read_exact_regular", mutate_member)
    try:
        receipt = write_historical_window_output(runner, result, root)
        assert receipt.status == "UNCERTAIN" and receipt.manifest_digest is None
    finally:
        _clean(root)


@pytest.mark.parametrize("stage", ["validation", "create", "member", "manifest", "final_read"])
def test_process_control_propagates_from_validation_create_member_manifest_and_final_read(tmp_path, monkeypatch, stage):
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    root = _root(tmp_path)
    _clean(root)
    if stage == "validation":
        monkeypatch.setattr(publication, "revalidate_historical_window_result", lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
    elif stage == "create":
        original_mkdir = publication.os.mkdir
        monkeypatch.setattr(publication.os, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()) if args[0] == root.name else original_mkdir(*args, **kwargs))
    elif stage in ("member", "manifest"):
        original_write = publication._write_file
        monkeypatch.setattr(
            publication, "_write_file",
            lambda fd, name, data: ((_ for _ in ()).throw(KeyboardInterrupt())
                                    if (stage == "member" or name == "output_manifest.json")
                                    else original_write(fd, name, data)),
        )
    else:
        original_read = publication._read_exact_regular
        calls = {"result.json": 0}
        def stop_final(path, logical):
            if logical == "result.json":
                calls[logical] += 1
                if calls[logical] == 2:
                    raise KeyboardInterrupt()
            return original_read(path, logical)
        monkeypatch.setattr(publication, "_read_exact_regular", stop_final)
    try:
        with pytest.raises(KeyboardInterrupt):
            write_historical_window_output(runner, result, root)
        if stage in ("validation", "create"):
            assert not root.exists()
        elif stage in ("member", "manifest"):
            assert root.exists() and not (root / "output_manifest.json").exists()
        else:
            assert (root / "output_manifest.json").exists()
    finally:
        _clean(root)


def test_receipt_public_construction_and_impossible_cross_fields_are_denied(tmp_path):
    with pytest.raises(HistoricalWindowPublicationContractError, match="writer-owned"):
        WindowPublicationReceipt(
            operation_id="x", status="COMMITTED", reason_code="COMMITTED", did_write=True,
            result_digest={}, manifest_digest={}, member_count=24,
            output_root_identity=(1, 2, 3, 0, 0), root_parent_identity_before=(1, 2, 3, 0, 0),
            root_parent_identity_after=(1, 2, 3, 0, 0),
        )
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    root = _root(tmp_path)
    _clean(root)
    try:
        receipt = write_historical_window_output(runner, result, root)
        with pytest.raises(HistoricalWindowPublicationContractError, match="writer-owned"):
            replace(receipt, status="CONFLICT")
        fields = receipt.to_dict()
        fields.update({
            "_authority": publication._PUBLICATION_AUTHORITY,
            "status": "CONFLICT", "reason_code": "OUTPUT_ALREADY_EXISTS",
            "did_write": False, "manifest_digest": None, "member_count": 24,
            "output_root_identity": None,
            "root_parent_identity_before": tuple(fields["root_parent_identity_before"]),
            "root_parent_identity_after": tuple(fields["root_parent_identity_after"]),
        })
        with pytest.raises(HistoricalWindowPublicationContractError, match="cross-fields"):
            WindowPublicationReceipt(**fields)
    finally:
        _clean(root)


def test_two_fresh_roots_have_exact_member_and_manifest_bytes(tmp_path):
    runner = build_window(tmp_path / "source", 4)[0]
    result = runner.run()
    left, right = _root(tmp_path, "left"), _root(tmp_path, "right")
    _clean(left, right)
    try:
        assert write_historical_window_output(runner, result, left).status == "COMMITTED"
        assert write_historical_window_output(runner, result, right).status == "COMMITTED"
        left_files = {p.relative_to(left).as_posix(): p.read_bytes() for p in left.rglob("*") if p.is_file()}
        right_files = {p.relative_to(right).as_posix(): p.read_bytes() for p in right.rglob("*") if p.is_file()}
        assert left_files == right_files
    finally:
        _clean(left, right)
