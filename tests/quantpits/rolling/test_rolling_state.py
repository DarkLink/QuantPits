"""Contracts for strict read-only Rolling state classification."""

import hashlib
import json
import dataclasses

import pytest

from quantpits.rolling.identity import workspace_fingerprint
from quantpits.rolling.state import (
    LegacyRollingStateSnapshot,
    RollingStateExpectation,
    RollingStateUnitClaim,
    RollingStateV2Snapshot,
    build_legacy_migration_proposal,
    inspect_rolling_state,
    parse_rolling_state_v2_bytes,
    serialize_rolling_state_v2,
)
from quantpits.rolling.errors import RollingIdentityError
from quantpits.utils.workspace import fingerprint_value


def _root(tmp_path):
    root = tmp_path / "Demo_Workspace"
    (root / "data").mkdir(parents=True)
    return root


def _legacy(family="rolling"):
    method = "cpcv" if family == "cpcv_rolling" else "slide"
    return {
        "started_at": "2024-04-01 00:00:00",
        "rolling_config": {},
        "anchor_date": "2024-03-31",
        "training_method": method,
        "completed_windows": {"0": {"demo": "recorder-demo"}},
        "current_window_idx": 0,
        "current_model": "demo",
        "total_windows": 1,
    }


def _v2(root, **updates):
    payload = {
        "schema_version": 2,
        "workspace_fingerprint": workspace_fingerprint(root),
        "run_id": "rolling-demo-run",
        "family": "rolling",
        "action": "daily",
        "plan_fingerprint": "a" * 64,
        "execution_fingerprint": "b" * 64,
        "config_fingerprint": "c" * 64,
        "anchor_date": "2024-03-31",
        "target_keys": ["demo@rolling"],
        "window_keys": ["rolling:2024-01-01:2024-03-31:abcdef123456"],
        "attempt_id": None,
        "phase": "executing",
        "units": [],
    }
    payload.update(updates)
    return payload


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_missing_is_distinct_from_zero_byte_and_empty_mapping(tmp_path):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    assert inspect_rolling_state(path, root).classification == "missing"
    path.write_bytes(b"")
    assert inspect_rolling_state(path, root).classification == "corrupt"
    path.write_bytes(b"\xff")
    assert inspect_rolling_state(path, root).classification == "corrupt"
    path.write_text("{} trailing", encoding="utf-8")
    assert inspect_rolling_state(path, root).classification == "corrupt"
    path.write_text("{}", encoding="utf-8")
    assert inspect_rolling_state(path, root).classification == "unsupported_schema"


def test_valid_legacy_slide_and_cpcv_snapshots_are_typed(tmp_path):
    root = _root(tmp_path)
    for family, filename in (
            ("rolling", "rolling_state.json"),
            ("cpcv_rolling", "rolling_state_cpcv.json")):
        path = root / "data" / filename
        _write(path, _legacy(family))
        inspection = inspect_rolling_state(path, root)
        assert inspection.classification == "valid_legacy"
        assert inspection.reason_code == "rolling_state_legacy_valid"
        assert isinstance(inspection.snapshot, LegacyRollingStateSnapshot)
        assert inspection.family == family
        assert inspection.completed_windows == 1
        assert inspection.completed_units == 1


def test_legacy_completed_units_preserve_bare_to_canonical_mapping_without_evidence_claim(tmp_path):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    _write(path, _legacy())
    inspection = inspect_rolling_state(path, root)
    _index, units = inspection.snapshot.completed_windows[0]
    assert units == (("demo@rolling", "recorder-demo"),)
    assert inspection.snapshot.to_public_dict()["completion_authority"] == "legacy_unverified"
    assert inspection.consumption == "legacy_compatible"


@pytest.mark.parametrize("change", [
    {"completed_windows": {"01": {"demo": "recorder-demo"}}},
    {"completed_windows": {"0": {"demo": ""}}},
    {"current_window_idx": None, "current_model": "demo"},
    {"current_window_idx": 1, "current_model": "demo"},
    {"total_windows": 0},
    {"anchor_date": "2024-3-31"},
    {"rolling_config": []},
])
def test_legacy_index_pointer_total_and_record_shapes_fail_closed(tmp_path, change):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    payload = _legacy()
    payload.update(change)
    _write(path, payload)
    inspection = inspect_rolling_state(path, root)
    assert inspection.classification in ("ambiguous", "unsupported_schema")
    assert inspection.consumption == "blocked"


def test_v2_identity_envelope_round_trips_to_immutable_snapshot(tmp_path):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    payload = _v2(
        root,
        target_keys=["demo@rolling", "other@rolling"],
        window_keys=[
            "rolling:2024-01-01:2024-03-31:abcdef123456",
            "rolling:2024-04-01:2024-06-30:111111111111",
        ],
    )
    _write(path, payload)
    inspection = inspect_rolling_state(path, root)
    assert inspection.classification == "valid_versioned"
    assert inspection.reason_code == "rolling_state_versioned_valid"
    assert isinstance(inspection.snapshot, RollingStateV2Snapshot)
    assert inspection.snapshot.to_public_dict() == dict(payload, extensions={})
    assert inspection.snapshot.target_keys == tuple(payload["target_keys"])
    assert inspection.snapshot.window_keys == tuple(payload["window_keys"])
    assert not hasattr(inspection, "save")
    assert not hasattr(inspection, "migrate")


@pytest.mark.parametrize("raw,classification", [
    ('{"schema_version": 3}', "unsupported_schema"),
    ('{"schema_version": true}', "unsupported_schema"),
    ('{"schema_version": "2"}', "unsupported_schema"),
    ('{"schema_version": 2.0}', "unsupported_schema"),
    ('{"schema_version": 2e0}', "unsupported_schema"),
    ('{"schema_version": 2, "schema_version": 2}', "corrupt"),
    ('{"schema_version": 2, "value": NaN}', "corrupt"),
])
def test_v2_rejects_unknown_version_duplicate_json_keys_constants_and_unknown_fields(
    tmp_path, raw, classification,
):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    path.write_text(raw, encoding="utf-8")
    assert inspect_rolling_state(path, root).classification == classification
    payload = _v2(root, unexpected=True)
    _write(path, payload)
    assert inspect_rolling_state(path, root).classification == "unsupported_schema"
    invalid_fields = (
        {"plan_fingerprint": "A" * 64},
        {"anchor_date": "2024-3-31"},
        {"family": "static"},
        {"action": "repair_truncated"},
        {"target_keys": ["demo"]},
        {"window_keys": ["slide:2024-01-01:2024-03-31:abcdef123456"]},
        {"phase": "unknown"},
    )
    for change in invalid_fields:
        _write(path, _v2(root, **change))
        assert inspect_rolling_state(path, root).classification == "unsupported_schema"


def test_v2_rejects_duplicate_or_out_of_scope_unit_identity(tmp_path):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    unit = {
        "target_key": "demo@rolling",
        "window_key": "rolling:2024-01-01:2024-03-31:abcdef123456",
        "status": "running",
    }
    _write(path, _v2(root, units=[unit, unit]))
    assert inspect_rolling_state(path, root).classification == "ambiguous"
    outside = dict(unit, target_key="other@rolling")
    _write(path, _v2(root, units=[outside]))
    assert inspect_rolling_state(path, root).classification == "unsupported_schema"


def test_state_path_symlink_escape_is_foreign_without_reading_target(tmp_path):
    root = _root(tmp_path)
    outside = tmp_path / "private_state.json"
    outside.write_text("private sentinel", encoding="utf-8")
    path = root / "data" / "rolling_state.json"
    path.symlink_to(outside)
    before = outside.read_bytes()
    inspection = inspect_rolling_state(path, root)
    assert inspection.classification == "foreign"
    assert inspection.fingerprint is None
    assert outside.read_bytes() == before

    path.unlink()
    path.parent.rmdir()
    outside_data = tmp_path / "private_data"
    outside_data.mkdir()
    outside_nested = outside_data / "rolling_state.json"
    outside_nested.write_text("another private sentinel", encoding="utf-8")
    (root / "data").symlink_to(outside_data, target_is_directory=True)
    nested_before = outside_nested.read_bytes()
    nested = inspect_rolling_state(root / "data" / "rolling_state.json", root)
    assert nested.classification == "foreign"
    assert nested.path_kind == "parent_symlink"
    assert nested.fingerprint is None
    assert outside_nested.read_bytes() == nested_before


@pytest.mark.parametrize("field,value", [
    ("workspace_fingerprint", "d" * 64),
    ("family", "cpcv_rolling"),
    ("attempt_id", "foreign-attempt"),
])
def test_expectation_classifies_workspace_family_and_attempt_as_foreign(
    tmp_path, field, value,
):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    payload_updates = {"attempt_id": "attempt-a"} if field == "attempt_id" else {}
    _write(path, _v2(root, **payload_updates))
    expectation = RollingStateExpectation(**{field: value})
    inspection = inspect_rolling_state(path, root, expectation)
    assert inspection.classification == "foreign"
    assert inspection.reason_code == "rolling_state_foreign"
    assert field in inspection.checked


@pytest.mark.parametrize("field,value", [
    ("config_fingerprint", "d" * 64),
    ("target_keys", ("other@rolling",)),
    ("window_keys", ("rolling:2024-04-01:2024-06-30:111111111111",)),
    ("run_id", "other-run"),
    ("execution_fingerprint", "d" * 64),
])
def test_expectation_classifies_config_target_window_and_run_mismatch(
    tmp_path, field, value,
):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    _write(path, _v2(root))
    inspection = inspect_rolling_state(
        path, root, RollingStateExpectation(**{field: value}),
    )
    assert inspection.classification == "identity_mismatch"
    assert inspection.reason_code == "rolling_state_identity_mismatch"
    assert field in inspection.checked
    assert inspection.to_public_dict()["expectation_checks"]["attempt_id"] == "not_checked"


def test_legacy_empty_config_is_fingerprinted_compared_and_payload_is_fail_closed(
    tmp_path,
):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    payload = _legacy()
    _write(path, payload)

    matching = inspect_rolling_state(
        path, root,
        RollingStateExpectation(config_fingerprint=fingerprint_value({})),
    )
    assert matching.classification == "valid_legacy"
    assert "config_fingerprint" in matching.checked
    assert matching.legacy_payload() == payload

    mismatch = inspect_rolling_state(
        path, root,
        RollingStateExpectation(config_fingerprint=fingerprint_value({"x": 1})),
    )
    assert mismatch.classification == "identity_mismatch"
    assert "config_fingerprint" in mismatch.checked
    assert mismatch.legacy_payload() is None


def test_legacy_missing_config_does_not_claim_the_expectation_was_checked(tmp_path):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    payload = _legacy()
    del payload["rolling_config"]
    _write(path, payload)
    inspection = inspect_rolling_state(
        path, root,
        RollingStateExpectation(config_fingerprint=fingerprint_value({})),
    )
    assert inspection.classification == "valid_legacy"
    assert "config_fingerprint" not in inspection.checked
    assert inspection.to_public_dict()["expectation_checks"][
        "config_fingerprint"
    ] == "not_checked"


def test_completion_claim_without_evidence_is_not_reuse_authority(tmp_path):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    unit = {
        "target_key": "demo@rolling",
        "window_key": "rolling:2024-01-01:2024-03-31:abcdef123456",
        "status": "success",
        "record_id": "recorder-demo",
    }
    _write(path, _v2(root, phase="completed", units=[unit]))
    inspection = inspect_rolling_state(path, root)
    assert inspection.classification == "unverified_completion"
    assert inspection.reason_code == "rolling_state_completion_unverified"
    assert inspection.consumption == "blocked"


def test_classification_is_byte_stable_and_zero_write(tmp_path):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    _write(path, _legacy())
    before_bytes = path.read_bytes()
    before_entries = tuple(sorted(item.name for item in path.parent.iterdir()))
    first = inspect_rolling_state(path, root)
    second = inspect_rolling_state(path, root)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint == hashlib.sha256(before_bytes).hexdigest()
    assert path.read_bytes() == before_bytes
    assert tuple(sorted(item.name for item in path.parent.iterdir())) == before_entries


def _typed_v2(root, **changes):
    payload = _v2(root, **changes)
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    return parse_rolling_state_v2_bytes(raw)


def test_v2_serializer_round_trips_deterministic_bytes_and_extensions(tmp_path):
    root = _root(tmp_path)
    unit = {
        "target_key": "demo@rolling",
        "window_key": "rolling:2024-01-01:2024-03-31:abcdef123456",
        "status": "running",
        "extensions": {"operator_note": {"value": 0}},
    }
    snapshot = _typed_v2(
        root,
        units=[unit],
        extensions={"empty": {}, "enabled": False},
    )
    first = serialize_rolling_state_v2(snapshot)
    second = serialize_rolling_state_v2(parse_rolling_state_v2_bytes(first))
    assert first == second
    assert first.endswith(b"\n")
    reparsed = parse_rolling_state_v2_bytes(first)
    assert reparsed.extensions == {"empty": {}, "enabled": False}
    assert reparsed.units[0].extensions == {"operator_note": {"value": 0}}
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()

    absent = _typed_v2(root, units=[{
        "target_key": "demo@rolling",
        "window_key": "rolling:2024-01-01:2024-03-31:abcdef123456",
        "status": "running",
    }])
    present_empty = _typed_v2(root, units=[{
        "target_key": "demo@rolling",
        "window_key": "rolling:2024-01-01:2024-03-31:abcdef123456",
        "status": "running",
        "extensions": {},
    }])
    assert b'"extensions": {}' not in serialize_rolling_state_v2(absent).split(
        b'"units":', 1,
    )[1]
    assert serialize_rolling_state_v2(absent) != serialize_rolling_state_v2(
        present_empty,
    )


def test_v2_serializer_revalidates_direct_snapshot_members_and_family(tmp_path):
    root = _root(tmp_path)
    valid = _typed_v2(root)
    invalid_unit = RollingStateUnitClaim(
        "other@rolling",
        "rolling:2024-01-01:2024-03-31:abcdef123456",
        "running",
    )
    with pytest.raises(RollingIdentityError):
        serialize_rolling_state_v2(dataclasses.replace(valid, units=(invalid_unit,)))
    with pytest.raises(RollingIdentityError):
        serialize_rolling_state_v2(dataclasses.replace(
            valid,
            window_keys=(
                "cpcv_rolling:2024-01-01:2024-03-31:abcdef123456",
            ),
        ))


def test_v2_serializer_rejects_closest_invalid_representations(tmp_path):
    root = _root(tmp_path)
    valid = _typed_v2(
        root,
        target_keys=["demo@rolling", "other@rolling"],
        window_keys=[
            "rolling:2024-01-01:2024-03-31:abcdef123456",
            "rolling:2024-04-01:2024-06-30:111111111111",
        ],
    )
    out_of_order = (
        RollingStateUnitClaim(
            "other@rolling",
            "rolling:2024-04-01:2024-06-30:111111111111",
            "running",
        ),
        RollingStateUnitClaim(
            "demo@rolling",
            "rolling:2024-01-01:2024-03-31:abcdef123456",
            "running",
        ),
    )
    with pytest.raises(RollingIdentityError):
        serialize_rolling_state_v2(dataclasses.replace(valid, units=out_of_order))
    with pytest.raises(RollingIdentityError):
        serialize_rolling_state_v2(dataclasses.replace(
            valid, _extensions_json='[]',
        ))
    with pytest.raises(RollingIdentityError):
        serialize_rolling_state_v2(dataclasses.replace(
            valid, _extensions_json='{"same": 1, "same": 2}',
        ))
    for raw in (b'{"schema_version": true}', b'{"schema_version": 2.0}'):
        with pytest.raises(RollingIdentityError):
            parse_rolling_state_v2_bytes(raw)


def _migration(root, inspection, **changes):
    values = {
        "inspection": inspection,
        "workspace_identity": workspace_fingerprint(root),
        "run_id": "migration-demo-run",
        "family": "rolling",
        "action": "daily",
        "plan_fingerprint": "a" * 64,
        "execution_fingerprint": "b" * 64,
        "config_fingerprint": fingerprint_value({}),
        "anchor_date": "2024-03-31",
        "target_keys": ("demo@rolling", "other@rolling"),
        "window_keys": (
            "rolling:2024-01-01:2024-03-31:abcdef123456",
            "rolling:2024-04-01:2024-06-30:111111111111",
        ),
        "index_window_keys": {
            0: "rolling:2024-01-01:2024-03-31:abcdef123456",
            1: "rolling:2024-04-01:2024-06-30:111111111111",
        },
    }
    values.update(changes)
    return build_legacy_migration_proposal(**values)


def test_legacy_migration_proposal_is_deterministic_and_zero_write(tmp_path):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    payload = _legacy()
    _write(path, payload)
    inspection = inspect_rolling_state(path, root)
    before = path.read_bytes()
    entries = tuple(sorted(item.name for item in path.parent.iterdir()))
    first = _migration(root, inspection)
    second = _migration(root, inspection)
    assert first.status == "candidate"
    assert first.capability == "proposal_only"
    assert first.proposed_bytes == second.proposed_bytes
    assert first.proposed_fingerprint == second.proposed_fingerprint
    assert first.source_fingerprint == inspection.fingerprint
    assert path.read_bytes() == before
    assert tuple(sorted(item.name for item in path.parent.iterdir())) == entries
    assert not hasattr(first, "apply")


def test_migration_proposal_preserves_requested_order_and_marks_legacy_claims_unverified(
        tmp_path):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    payload = _legacy()
    payload["completed_windows"] = {
        "0": {"other": "recorder-other", "demo": "recorder-demo"},
    }
    payload["current_model"] = "demo"
    _write(path, payload)
    proposal = _migration(root, inspect_rolling_state(path, root))
    assert proposal.status == "candidate"
    snapshot = proposal.proposed_snapshot
    assert snapshot.target_keys == ("demo@rolling", "other@rolling")
    assert snapshot.window_keys == (
        "rolling:2024-01-01:2024-03-31:abcdef123456",
        "rolling:2024-04-01:2024-06-30:111111111111",
    )
    assert [unit.target_key for unit in snapshot.units] == [
        "demo@rolling", "other@rolling",
    ]
    assert all(unit.status == "success" for unit in snapshot.units)
    assert all(unit.evidence_id is None for unit in snapshot.units)
    assert all(unit.extensions["claim_authority"] == "legacy_unverified"
               for unit in snapshot.units)


@pytest.mark.parametrize("change", [
    {"index_window_keys": {
        0: "rolling:2024-01-01:2024-03-31:abcdef123456",
    }},
    {"target_keys": ("other@rolling",)},
    {"family": "cpcv_rolling"},
    {"anchor_date": "2024-04-01"},
])
def test_migration_proposal_blocks_missing_partial_foreign_or_out_of_scope_mapping(
        tmp_path, change):
    root = _root(tmp_path)
    path = root / "data" / "rolling_state.json"
    _write(path, _legacy())
    proposal = _migration(root, inspect_rolling_state(path, root), **change)
    assert proposal.status == "blocked"
    assert proposal.capability == "none"
    assert proposal.proposed_bytes is None
    assert proposal.proposed_snapshot is None

    missing_path = root / "data" / "missing.json"
    missing = inspect_rolling_state(missing_path, root)
    not_applicable = _migration(root, missing)
    assert not_applicable.status == "not_applicable"
    assert not_applicable.proposed_bytes is None
