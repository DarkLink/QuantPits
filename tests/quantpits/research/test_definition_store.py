from __future__ import annotations

import json
import hashlib
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path

import pytest

import quantpits.research.definition_store as definition_store
from quantpits.research.definition_store import (
    DEFINITION_PATHS,
    MANIFEST_NAME,
    CreateOnlyDefinitionBundleStore,
    DefinitionBundleMember,
    DefinitionBundleRequest,
    DefinitionStoreContractError,
    DefinitionStoreInputError,
    DefinitionStoreReceipt,
    revalidate_definition_bundle_request,
)


def _bytes(label):
    return json.dumps({"label": label}, sort_keys=True, separators=(",", ":")).encode()


def _request(identifier="candidate-1", changed=None):
    changed = changed or {}
    return DefinitionBundleRequest(
        identifier,
        tuple(
            DefinitionBundleMember(path, changed.get(path, _bytes(path)))
            for path in DEFINITION_PATHS
        ),
    )


def _root(tmp_path, name="store"):
    root = tmp_path / name
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _files(bundle):
    return {path.name: path.read_bytes() for path in bundle.iterdir() if path.is_file()}


def test_request_revalidates_exact_four_canonical_json_members_and_digest():
    first = _request()
    second = _request()
    rebuilt = revalidate_definition_bundle_request(first)
    assert rebuilt is not first
    assert rebuilt.members is not first.members
    assert rebuilt.request_digest == first.request_digest == second.request_digest
    digest_payload = {
        "schema_version": 1,
        "bundle_kind": "SHADOW_FORWARD_DEFINITIONS_V1",
        "definition_set_id": "candidate-1",
        "member_count": 4,
        "members": [
            {
                "logical_path": member.logical_path,
                "size_bytes": len(member.canonical_json_bytes),
                "digest": {
                    "algorithm": "sha256",
                    "domain": "raw_bytes",
                    "value": hashlib.sha256(member.canonical_json_bytes).hexdigest(),
                    "size_bytes": len(member.canonical_json_bytes),
                },
            }
            for member in first.members
        ],
    }
    digest_bytes = json.dumps(
        digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()
    assert dict(first.request_digest) == {
        "algorithm": "sha256", "domain": "canonical_json",
        "value": hashlib.sha256(digest_bytes).hexdigest(), "size_bytes": len(digest_bytes),
    }
    changed = _request(changed={"protocol.json": b'{"label":"changed"}'})
    assert changed.request_digest != first.request_digest
    with pytest.raises(TypeError):
        second.request_digest["value"] = "0" * 64
    object.__setattr__(first, "request_digest", {"caller": "forged"})
    assert revalidate_definition_bundle_request(first).request_digest == second.request_digest


def test_request_rejects_closest_invalid_member_id_set_order_and_size(monkeypatch):
    invalid_payloads = (
        None, True, "{}", b"", b"[]", b'{"x":1,"x":2}', b'{"x":NaN}',
        b'{ "x":1}', b'{"z":1,"a":2}',
    )
    for value in invalid_payloads:
        with pytest.raises(DefinitionStoreContractError):
            DefinitionBundleMember("protocol.json", value)
    with pytest.raises(DefinitionStoreContractError):
        DefinitionBundleMember("foreign.json", b"{}")
    base = list(_request().members)
    for members in (base[:-1], base + [base[0]], list(reversed(base)), [base[0]] * 4, object()):
        with pytest.raises(DefinitionStoreContractError):
            DefinitionBundleRequest("candidate-1", members)
    for identifier in (None, True, "", ".", "..", " Candidate", "UPPER", "a/b", "a\\b", "é"):
        with pytest.raises(DefinitionStoreContractError):
            DefinitionBundleRequest(identifier, base)
    for foreign in (None, False, object()):
        with pytest.raises(DefinitionStoreContractError):
            CreateOnlyDefinitionBundleStore(Path("relative")).publish(foreign)
    class ForeignMember(DefinitionBundleMember):
        pass
    with pytest.raises(DefinitionStoreContractError):
        DefinitionBundleRequest(
            "candidate-1", [ForeignMember(item.logical_path, item.canonical_json_bytes) for item in base],
        )
    class ForeignRequest(DefinitionBundleRequest):
        pass
    foreign_request = ForeignRequest("candidate-1", base)
    with pytest.raises(DefinitionStoreContractError):
        revalidate_definition_bundle_request(foreign_request)
    monkeypatch.setattr(definition_store, "MAX_MEMBER_BYTES", 1)
    with pytest.raises(DefinitionStoreContractError):
        DefinitionBundleMember("protocol.json", b"{}")


def test_first_publication_commits_manifest_last_with_exact_private_layout(tmp_path, monkeypatch):
    root = _root(tmp_path)
    request = _request()
    writes = []
    original = definition_store._write_exclusive

    def observe(descriptor, name, data):
        writes.append(name)
        return original(descriptor, name, data)

    monkeypatch.setattr(definition_store, "_write_exclusive", observe)
    receipt = CreateOnlyDefinitionBundleStore(root).publish(request)
    bundle = root / request.definition_set_id
    assert receipt.status == "COMMITTED" and receipt.publication_capability
    assert receipt.did_write is True and receipt.member_count == 4
    assert writes == sorted(DEFINITION_PATHS) + [MANIFEST_NAME]
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o700
    assert set(path.name for path in bundle.iterdir()) == set(DEFINITION_PATHS) | {MANIFEST_NAME}
    assert set(path.name for path in root.iterdir()) == {"candidate-1"}
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in bundle.iterdir())
    manifest = json.loads((bundle / MANIFEST_NAME).read_bytes())
    assert set(manifest) == {
        "schema_version", "bundle_kind", "storage_claim", "semantic_claim",
        "prospective_claim", "promotion_capability", "definition_set_id",
        "request_digest", "member_count", "members",
    }
    assert [row["logical_path"] for row in manifest["members"]] == sorted(DEFINITION_PATHS)
    for row in manifest["members"]:
        data = (bundle / row["logical_path"]).read_bytes()
        assert row["size_bytes"] == len(data)
        assert row["digest"] == {
            "algorithm": "sha256", "value": hashlib.sha256(data).hexdigest(),
        }
    assert manifest["semantic_claim"] is False
    assert manifest["prospective_claim"] is False
    assert manifest["promotion_capability"] is False


def test_identical_replay_is_adopted_without_write_or_byte_change(tmp_path, monkeypatch):
    root = _root(tmp_path)
    store = CreateOnlyDefinitionBundleStore(root)
    assert store.publish(_request()).status == "COMMITTED"
    before = _files(root / "candidate-1")
    monkeypatch.setattr(
        definition_store, "_write_exclusive",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected write")),
    )
    receipt = store.publish(_request())
    assert receipt.status == "ADOPTED" and receipt.did_write is False
    assert receipt.publication_capability and receipt.member_count == 4
    assert _files(root / "candidate-1") == before


def test_same_id_different_bytes_is_conflict_and_existing_bundle_is_unchanged(tmp_path):
    root = _root(tmp_path)
    store = CreateOnlyDefinitionBundleStore(root)
    assert store.publish(_request()).status == "COMMITTED"
    bundle = root / "candidate-1"
    before = _files(bundle)
    receipt = store.publish(_request(changed={"protocol.json": b'{"different":true}'}))
    assert receipt.status == "CONFLICT" and receipt.did_write is False
    assert receipt.member_count == 0 and not receipt.publication_capability
    assert receipt.manifest_digest is None and _files(bundle) == before


@pytest.mark.parametrize(
    "mutation", [
        "partial", "malformed", "manifest", "foreign", "symlink", "hardlink", "fifo",
        "wrong_mode", "target_symlink", "target_hardlink", "target_fifo", "target_file",
        "target_wrong_mode",
    ],
)
def test_partial_malformed_foreign_symlink_hardlink_special_or_wrong_mode_existing_bundle_is_not_adopted(
    tmp_path, mutation,
):
    root = _root(tmp_path)
    store = CreateOnlyDefinitionBundleStore(root)
    assert store.publish(_request()).status == "COMMITTED"
    bundle = root / "candidate-1"
    target = bundle / "protocol.json"
    if mutation == "partial":
        (bundle / MANIFEST_NAME).unlink()
    elif mutation == "malformed":
        target.write_bytes(b"not-json")
    elif mutation == "manifest":
        (bundle / MANIFEST_NAME).write_bytes(b"{}")
    elif mutation == "foreign":
        (bundle / "foreign").write_bytes(b"x")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(bundle / "champion_strategy.json")
    elif mutation == "hardlink":
        target.unlink()
        os.link(str(bundle / "champion_strategy.json"), str(target))
    elif mutation == "fifo":
        target.unlink()
        os.mkfifo(str(target), 0o600)
    elif mutation == "wrong_mode":
        target.chmod(0o644)
    elif mutation == "target_wrong_mode":
        bundle.chmod(0o755)
    else:
        shutil.rmtree(str(bundle))
        if mutation == "target_symlink":
            bundle.symlink_to(root, target_is_directory=True)
        elif mutation == "target_hardlink":
            source = root / "source-file"
            source.write_bytes(b"x")
            os.link(str(source), str(bundle))
        elif mutation == "target_fifo":
            os.mkfifo(str(bundle), 0o600)
        else:
            bundle.write_bytes(b"x")
    before = os.lstat(str(bundle))
    before_inventory = (
        tuple(sorted(path.name for path in bundle.iterdir()))
        if stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode) else None
    )
    receipt = store.publish(_request())
    assert receipt.status == "CONFLICT" and receipt.did_write is False
    assert receipt.member_count == 0 and not receipt.publication_capability
    after = os.lstat(str(bundle))
    assert definition_store._file_identity(after) == definition_store._file_identity(before)
    if before_inventory is not None:
        assert tuple(sorted(path.name for path in bundle.iterdir())) == before_inventory


def test_invalid_store_root_is_denied_before_bundle_create(tmp_path):
    request = _request()
    missing = (tmp_path / "missing").resolve()
    with pytest.raises(DefinitionStoreContractError):
        CreateOnlyDefinitionBundleStore(Path("relative")).publish(request)
    with pytest.raises(DefinitionStoreInputError):
        CreateOnlyDefinitionBundleStore(missing).publish(request)
    wrong = tmp_path / "wrong"
    wrong.mkdir(mode=0o755)
    wrong.chmod(0o755)
    with pytest.raises(DefinitionStoreInputError):
        CreateOnlyDefinitionBundleStore(wrong).publish(request)
    assert stat.S_IMODE(wrong.stat().st_mode) == 0o755
    regular = tmp_path / "regular"
    regular.write_bytes(b"x")
    with pytest.raises(DefinitionStoreInputError):
        CreateOnlyDefinitionBundleStore(regular).publish(request)
    real = _root(tmp_path, "real")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(DefinitionStoreInputError):
        CreateOnlyDefinitionBundleStore(alias).publish(request)
    assert not (real / "candidate-1").exists()


@pytest.mark.parametrize("stage", ["member", "manifest", "manifest_read", "final_member", "final_inventory"])
def test_member_and_manifest_failure_are_uncertain_without_publication_capability(
    tmp_path, monkeypatch, stage,
):
    root = _root(tmp_path)
    original = definition_store._write_exclusive
    calls = []

    def fail(descriptor, name, data):
        calls.append(name)
        if (stage == "member" and len(calls) == 2) or (stage == "manifest" and name == MANIFEST_NAME):
            raise OSError("injected")
        return original(descriptor, name, data)

    monkeypatch.setattr(definition_store, "_write_exclusive", fail)
    if stage in ("manifest_read", "final_member"):
        original_read = definition_store._read_regular_at
        reads = {"protocol.json": 0}

        def fail_read(descriptor, name, conflict=False):
            if stage == "manifest_read" and name == MANIFEST_NAME:
                raise OSError("injected")
            if stage == "final_member" and name == "protocol.json":
                reads[name] += 1
                if reads[name] == 2:
                    target = root / "candidate-1" / name
                    target.write_bytes(b'{"changed":true}')
                    target.chmod(0o600)
            return original_read(descriptor, name, conflict=conflict)

        monkeypatch.setattr(definition_store, "_read_regular_at", fail_read)
    if stage == "final_inventory":
        original_listdir = definition_store.os.listdir
        injected = {"value": False}

        def foreign_inventory(descriptor):
            if not injected["value"]:
                injected["value"] = True
                foreign = root / "candidate-1" / "foreign"
                foreign.write_bytes(b"x")
                foreign.chmod(0o600)
            return original_listdir(descriptor)

        monkeypatch.setattr(definition_store.os, "listdir", foreign_inventory)
    receipt = CreateOnlyDefinitionBundleStore(root).publish(_request())
    assert receipt.status == "UNCERTAIN" and receipt.did_write is True
    assert receipt.member_count == (1 if stage == "member" else 4)
    assert receipt.manifest_digest is None and not receipt.publication_capability
    if stage == "member":
        assert calls == sorted(DEFINITION_PATHS)[:2]
    if stage in ("member", "manifest"):
        assert not (root / "candidate-1" / MANIFEST_NAME).exists()
    else:
        assert (root / "candidate-1" / MANIFEST_NAME).exists()


@pytest.mark.parametrize("node", ["store", "bundle", "existing_store", "existing_bundle"])
def test_store_or_bundle_checkpoint_identity_drift_is_uncertain(tmp_path, monkeypatch, node):
    root = _root(tmp_path)
    store = CreateOnlyDefinitionBundleStore(root)
    if node.startswith("existing_"):
        assert store.publish(_request()).status == "COMMITTED"
        original_read = definition_store._read_regular_at
        changed = {"value": False}

        def replace_namespace(descriptor, name, conflict=False):
            if not changed["value"]:
                changed["value"] = True
                victim = root if node == "existing_store" else root / "candidate-1"
                displaced = tmp_path / (node + "_displaced")
                victim.rename(displaced)
                victim.mkdir(mode=0o700)
                victim.chmod(0o700)
            return original_read(descriptor, name, conflict=conflict)

        monkeypatch.setattr(definition_store, "_read_regular_at", replace_namespace)
        receipt = store.publish(_request())
        assert receipt.status == "UNCERTAIN" and receipt.did_write is False
        assert receipt.member_count == 0 and not receipt.publication_capability
        return
    original = definition_store._path_directory_identity
    original_fd = definition_store._fd_directory_identity
    original_write = definition_store._write_exclusive
    active = {"value": False}

    def activate(descriptor, name, data):
        result = original_write(descriptor, name, data)
        if name == MANIFEST_NAME:
            active["value"] = True
        return result

    def drift(path):
        identity = original(path)
        if active["value"] and node == "bundle" and Path(path) != root:
            return (identity[0], identity[1] + 1, identity[2], 0, 0)
        return identity

    def drift_fd(descriptor):
        identity = original_fd(descriptor)
        if active["value"] and node == "store" and os.fstat(descriptor).st_ino == root.stat().st_ino:
            return (identity[0], identity[1] + 1, identity[2], 0, 0)
        return identity

    monkeypatch.setattr(definition_store, "_write_exclusive", activate)
    monkeypatch.setattr(definition_store, "_path_directory_identity", drift)
    monkeypatch.setattr(definition_store, "_fd_directory_identity", drift_fd)
    receipt = store.publish(_request())
    assert receipt.status == "UNCERTAIN" and receipt.did_write is True
    assert not receipt.publication_capability and receipt.manifest_digest is None


@pytest.mark.parametrize("control", [KeyboardInterrupt, SystemExit, GeneratorExit])
@pytest.mark.parametrize("stage", ["validation", "create", "member", "manifest", "final_read"])
def test_process_control_propagates_from_validation_create_member_manifest_and_final_read(
    tmp_path, monkeypatch, stage, control,
):
    root = _root(tmp_path)
    request = _request()
    if stage == "validation":
        monkeypatch.setattr(
            definition_store, "revalidate_definition_bundle_request",
            lambda *_args: (_ for _ in ()).throw(control()),
        )
    elif stage == "create":
        original = definition_store.os.mkdir
        monkeypatch.setattr(
            definition_store.os, "mkdir",
            lambda *args, **kwargs: (
                (_ for _ in ()).throw(control())
                if args[0] == "candidate-1" else original(*args, **kwargs)
            ),
        )
    elif stage in ("member", "manifest"):
        original = definition_store._write_exclusive
        monkeypatch.setattr(
            definition_store, "_write_exclusive",
            lambda fd, name, data: (
                (_ for _ in ()).throw(control())
                if (stage == "member" or name == MANIFEST_NAME) else original(fd, name, data)
            ),
        )
    else:
        original = definition_store._read_regular_at
        calls = {"protocol.json": 0}

        def stop_final(descriptor, name, conflict=False):
            if name == "protocol.json":
                calls[name] += 1
                if calls[name] == 2:
                    raise control()
            return original(descriptor, name, conflict=conflict)

        monkeypatch.setattr(definition_store, "_read_regular_at", stop_final)
    with pytest.raises(control):
        CreateOnlyDefinitionBundleStore(root).publish(request)
    bundle = root / "candidate-1"
    if stage in ("validation", "create"):
        assert not bundle.exists()
    elif stage in ("member", "manifest"):
        assert bundle.exists() and not (bundle / MANIFEST_NAME).exists()
    else:
        assert (bundle / MANIFEST_NAME).exists()


def test_receipt_is_writer_owned_and_rejects_impossible_cross_fields(tmp_path):
    fields = dict(
        operation_id="0" * 64,
        status="COMMITTED",
        reason_code="COMMITTED",
        did_write=True,
        definition_set_id="candidate-1",
        request_digest={"algorithm": "sha256", "domain": "canonical_json", "value": "0" * 64, "size_bytes": 1},
        manifest_digest={"algorithm": "sha256", "domain": "raw_bytes", "value": "0" * 64, "size_bytes": 1},
        member_count=4,
        store_root_identity_before=(1, 2, stat.S_IFDIR | 0o700, 0, 0),
        store_root_identity_after=(1, 2, stat.S_IFDIR | 0o700, 0, 0),
        bundle_root_identity=(1, 3, stat.S_IFDIR | 0o700, 0, 0),
    )
    with pytest.raises(DefinitionStoreContractError, match="writer-owned"):
        DefinitionStoreReceipt(**fields)
    root = _root(tmp_path)
    receipt = CreateOnlyDefinitionBundleStore(root).publish(_request())
    with pytest.raises(TypeError):
        receipt.request_digest["value"] = "0" * 64
    with pytest.raises(TypeError):
        receipt.manifest_digest["value"] = "0" * 64
    assert set(receipt.to_dict()) == {
        "operation_id", "status", "reason_code", "did_write", "definition_set_id",
        "request_digest", "manifest_digest", "member_count", "store_root_identity_before",
        "store_root_identity_after", "bundle_root_identity",
    }
    operation_payload = {
        "domain": "C0_DEFINITION_PUBLICATION_V1",
        "definition_set_id": receipt.definition_set_id,
        "request_digest": dict(receipt.request_digest),
        "store_root_identity_before": list(receipt.store_root_identity_before),
    }
    assert receipt.operation_id == hashlib.sha256(json.dumps(
        operation_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    with pytest.raises(DefinitionStoreContractError, match="writer-owned"):
        replace(receipt, status="CONFLICT")
    fields = receipt.to_dict()
    fields.update({
        "_authority": definition_store._RECEIPT_AUTHORITY,
        "status": "CONFLICT",
        "reason_code": "DEFINITION_SET_ID_CONFLICT",
        "did_write": False,
        "store_root_identity_before": tuple(fields["store_root_identity_before"]),
        "store_root_identity_after": tuple(fields["store_root_identity_after"]),
        "bundle_root_identity": tuple(fields["bundle_root_identity"]),
    })
    with pytest.raises(DefinitionStoreContractError, match="cross-fields"):
        DefinitionStoreReceipt(**fields)


def test_two_store_roots_produce_exact_same_member_and_manifest_bytes(tmp_path):
    left, right = _root(tmp_path, "left"), _root(tmp_path, "right")
    request = _request()
    assert CreateOnlyDefinitionBundleStore(left).publish(request).status == "COMMITTED"
    assert CreateOnlyDefinitionBundleStore(right).publish(request).status == "COMMITTED"
    assert _files(left / "candidate-1") == _files(right / "candidate-1")


def test_definition_store_import_preserves_environment_cwd_and_optional_dependency_state(monkeypatch):
    import importlib
    import sys

    before_env = dict(os.environ)
    before_cwd = os.getcwd()
    watched = {name: name in sys.modules for name in ("qlib", "mlflow", "quantpits.utils.env")}
    sys.modules.pop("quantpits.research.definition_store", None)
    module = importlib.import_module("quantpits.research.definition_store")
    assert module.BUNDLE_KIND == "SHADOW_FORWARD_DEFINITIONS_V1"
    assert dict(os.environ) == before_env and os.getcwd() == before_cwd
    assert {name: name in sys.modules for name in watched} == watched
