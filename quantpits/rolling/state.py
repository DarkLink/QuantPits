"""Strict, single-snapshot, read-only Rolling state classification."""

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from quantpits.rolling.identity import (
    ROLLING_ACTIONS,
    ROLLING_FAMILIES,
    RollingTargetIdentity,
    family_for_training_method,
    normalize_iso_date,
    parse_rolling_window_key,
    training_method_for_family,
    workspace_fingerprint,
)
from quantpits.rolling.errors import RollingIdentityError
from quantpits.utils.workspace import fingerprint_value


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_V2_PHASES = ("prepared", "executing", "units_complete", "failed", "completed")
_UNIT_STATUSES = ("pending", "running", "success", "failed", "skipped", "completed")
_V2_FIELDS = {
    "schema_version", "workspace_fingerprint", "run_id", "family", "action",
    "plan_fingerprint", "execution_fingerprint", "config_fingerprint",
    "anchor_date", "target_keys", "window_keys", "attempt_id", "phase",
    "units", "extensions",
}
_UNIT_FIELDS = {
    "target_key", "window_key", "status", "record_id", "evidence_id",
    "extensions",
}
_EXPECTATION_FIELDS = (
    "workspace_fingerprint", "family", "attempt_id", "config_fingerprint",
    "target_keys", "window_keys", "run_id", "execution_fingerprint",
)


class _DuplicateKey(ValueError):
    pass


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON constant: %s" % value)


def _strict_json_bytes(data):
    text = data.decode("utf-8")
    return json.loads(
        text, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant,
    )


def _extensions_mapping(value):
    if value is None:
        return {}
    if not isinstance(value, str):
        raise ValueError("extensions representation must be canonical JSON text")
    parsed = _strict_json_bytes(value.encode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("extensions must decode to a mapping")
    return parsed


def _is_digest(value):
    return isinstance(value, str) and _DIGEST_RE.match(value) is not None


def _family_from_path(path):
    if path.name == "rolling_state.json":
        return "rolling"
    if path.name == "rolling_state_cpcv.json":
        return "cpcv_rolling"
    return None


@dataclass(frozen=True)
class RollingStateExpectation:
    workspace_fingerprint: str = None
    family: str = None
    config_fingerprint: str = None
    target_keys: tuple = None
    window_keys: tuple = None
    run_id: str = None
    execution_fingerprint: str = None
    attempt_id: str = None


@dataclass(frozen=True)
class LegacyRollingStateSnapshot:
    family: str
    training_method: str
    anchor_date: str
    completed_windows: tuple
    current_window_idx: int = None
    current_model: str = None
    total_windows: int = None
    rolling_config_fingerprint: str = None
    unknown_fields: tuple = ()
    _raw_payload_json: str = "{}"

    @property
    def completed_window_count(self):
        return len(self.completed_windows)

    @property
    def completed_unit_count(self):
        return sum(len(models) for _, models in self.completed_windows)

    def raw_payload(self):
        return json.loads(self._raw_payload_json)

    def to_public_dict(self):
        target_keys = []
        for _index, models in self.completed_windows:
            for target_key, _record_id in models:
                if target_key not in target_keys:
                    target_keys.append(target_key)
        return {
            "family": self.family,
            "training_method": self.training_method,
            "anchor_date": self.anchor_date,
            "completed_windows": self.completed_window_count,
            "completed_units": self.completed_unit_count,
            "completed_window_indices": [
                index for index, _models in self.completed_windows
            ],
            "completed_target_keys": target_keys,
            "current_window_idx": self.current_window_idx,
            "current_model": self.current_model,
            "current_target_key": (
                RollingTargetIdentity(self.current_model, self.family).target_key
                if self.current_model is not None else None
            ),
            "total_windows": self.total_windows,
            "completion_authority": "legacy_unverified",
            "unknown_fields": list(self.unknown_fields),
        }


@dataclass(frozen=True)
class RollingStateUnitClaim:
    target_key: str
    window_key: str
    status: str
    record_id: str = None
    evidence_id: str = None
    _extensions_json: str = None

    @property
    def extensions(self):
        return _extensions_mapping(self._extensions_json)

    def to_public_dict(self):
        payload = {
            "target_key": self.target_key,
            "window_key": self.window_key,
            "status": self.status,
            "record_id": self.record_id,
            "evidence_id": self.evidence_id,
        }
        if self._extensions_json is not None:
            payload["extensions"] = self.extensions
        return payload


@dataclass(frozen=True)
class RollingStateV2Snapshot:
    workspace_fingerprint: str
    run_id: str
    family: str
    action: str
    plan_fingerprint: str
    execution_fingerprint: str
    config_fingerprint: str
    anchor_date: str
    target_keys: tuple
    window_keys: tuple
    attempt_id: str
    phase: str
    units: tuple
    _extensions_json: str = "{}"

    @property
    def extensions(self):
        return _extensions_mapping(self._extensions_json)

    def to_public_dict(self):
        return {
            "schema_version": 2,
            "workspace_fingerprint": self.workspace_fingerprint,
            "run_id": self.run_id,
            "family": self.family,
            "action": self.action,
            "plan_fingerprint": self.plan_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "anchor_date": self.anchor_date,
            "target_keys": list(self.target_keys),
            "window_keys": list(self.window_keys),
            "attempt_id": self.attempt_id,
            "phase": self.phase,
            "units": [item.to_public_dict() for item in self.units],
            "extensions": self.extensions,
        }


@dataclass(frozen=True)
class RollingStateInspection:
    classification: str
    protocol: str
    reason_code: str
    consumption: str
    migration: str
    path: str
    path_kind: str
    fingerprint: str = None
    snapshot: object = None
    checked: tuple = ()
    warnings: tuple = ()
    blockers: tuple = ()

    @property
    def status(self):
        """Compatibility view for Phase 28 callers."""

        if self.classification == "unsupported_schema":
            return "unsupported"
        return self.classification

    @property
    def warning(self):
        values = self.blockers or self.warnings
        return values[0] if values else None

    @property
    def anchor(self):
        return getattr(self.snapshot, "anchor_date", None)

    @property
    def training_method(self):
        return getattr(self.snapshot, "training_method", None)

    @property
    def completed_windows(self):
        return getattr(self.snapshot, "completed_window_count", 0)

    @property
    def completed_units(self):
        return getattr(self.snapshot, "completed_unit_count", 0)

    @property
    def family(self):
        return getattr(self.snapshot, "family", None)

    @property
    def run_id(self):
        return getattr(self.snapshot, "run_id", None)

    @property
    def attempt_id(self):
        return getattr(self.snapshot, "attempt_id", None)

    def legacy_payload(self):
        if (self.classification == "valid_legacy"
                and isinstance(self.snapshot, LegacyRollingStateSnapshot)):
            return self.snapshot.raw_payload()
        return None

    def to_public_dict(self):
        return {
            "classification": self.classification,
            "protocol": self.protocol,
            "reason_code": self.reason_code,
            "consumption": self.consumption,
            "migration": self.migration,
            "path": self.path,
            "path_kind": self.path_kind,
            "fingerprint": self.fingerprint,
            "checked": list(self.checked),
            "expectation_checks": {
                field: ("checked" if field in self.checked else "not_checked")
                for field in _EXPECTATION_FIELDS
            },
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "snapshot": (
                self.snapshot.to_public_dict() if self.snapshot is not None else None
            ),
        }


def _inspection(classification, protocol, reason, consumption, migration,
                relative, path_kind, fingerprint=None, snapshot=None,
                checked=(), warnings=(), blockers=()):
    return RollingStateInspection(
        classification=classification,
        protocol=protocol,
        reason_code=reason,
        consumption=consumption,
        migration=migration,
        path=relative,
        path_kind=path_kind,
        fingerprint=fingerprint,
        snapshot=snapshot,
        checked=tuple(checked),
        warnings=tuple(warnings),
        blockers=tuple(blockers),
    )


def _unsupported(relative, path_kind, fingerprint, message, protocol="unknown"):
    return _inspection(
        "unsupported_schema", protocol, "rolling_state_schema_unsupported",
        "blocked", "blocked", relative, path_kind, fingerprint,
        blockers=(message,),
    )


def _parse_legacy(payload, family):
    if "completed_windows" not in payload:
        raise ValueError("legacy state requires completed_windows")
    completed = payload["completed_windows"]
    if not isinstance(completed, dict):
        raise ValueError("completed_windows must be a mapping")
    method = payload.get("training_method", training_method_for_family(family))
    if method not in ("slide", "cpcv"):
        raise ValueError("legacy training_method is unsupported")
    if family_for_training_method(method) != family:
        raise LookupError("legacy training_method conflicts with state family")
    anchor = payload.get("anchor_date")
    if anchor is not None:
        anchor = normalize_iso_date(anchor, "anchor_date")
    normalized_windows = []
    canonical_indices = set()
    for key, model_records in completed.items():
        if (not isinstance(key, str) or not key.isdigit()
                or str(int(key)) != key):
            raise KeyError("legacy window index is not canonical")
        index = int(key)
        if index in canonical_indices:
            raise KeyError("legacy window indices collide")
        canonical_indices.add(index)
        if not isinstance(model_records, dict):
            raise ValueError("legacy window value must be a model mapping")
        normalized_models = []
        for model_name, record_id in model_records.items():
            identity = RollingTargetIdentity(model_name=model_name, family=family)
            if (not isinstance(record_id, str) or not record_id
                    or record_id != record_id.strip()):
                raise ValueError("legacy recorder ID must be a non-empty string")
            normalized_models.append((identity.target_key, record_id))
        normalized_windows.append((index, tuple(normalized_models)))
    normalized_windows.sort(key=lambda item: item[0])
    total = payload.get("total_windows")
    if total is not None:
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("legacy total_windows must be a non-negative integer")
        if normalized_windows and total < normalized_windows[-1][0] + 1:
            raise ValueError("legacy total_windows is below the highest completed index")
    current_index = payload.get("current_window_idx")
    current_model = payload.get("current_model")
    if (current_index is None) != (current_model is None):
        raise ValueError("legacy current pointer must be wholly present or absent")
    if current_index is not None:
        if isinstance(current_index, bool) or not isinstance(current_index, int):
            raise ValueError("legacy current_window_idx must be an integer")
        current_target = RollingTargetIdentity(current_model, family).target_key
        pairs = {
            (window_index, target_key)
            for window_index, models in normalized_windows
            for target_key, _record_id in models
        }
        if (current_index, current_target) not in pairs:
            raise ValueError("legacy current pointer does not name a completed unit")
    rolling_config = payload.get("rolling_config")
    if rolling_config is not None and not isinstance(rolling_config, dict):
        raise ValueError("legacy rolling_config must be a mapping")
    known = {
        "started_at", "rolling_config", "anchor_date", "training_method",
        "completed_windows", "current_window_idx", "current_model",
        "total_windows",
    }
    return LegacyRollingStateSnapshot(
        family=family,
        training_method=method,
        anchor_date=anchor,
        completed_windows=tuple(normalized_windows),
        current_window_idx=current_index,
        current_model=current_model,
        total_windows=total,
        rolling_config_fingerprint=(
            fingerprint_value(rolling_config) if rolling_config is not None else None
        ),
        unknown_fields=tuple(sorted(set(payload) - known)),
        _raw_payload_json=json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ),
    )


def _parse_v2(payload):
    unknown = set(payload) - _V2_FIELDS
    if unknown:
        raise ValueError("unknown V2 top-level fields: %s" % ", ".join(sorted(unknown)))
    required = _V2_FIELDS - {"extensions"}
    missing = required - set(payload)
    if missing:
        raise ValueError("missing V2 fields: %s" % ", ".join(sorted(missing)))
    for field in (
            "workspace_fingerprint", "plan_fingerprint",
            "execution_fingerprint", "config_fingerprint"):
        if not _is_digest(payload[field]):
            raise ValueError("%s must be a lowercase SHA-256 digest" % field)
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id or run_id != run_id.strip():
        raise ValueError("run_id must be a non-empty trimmed string")
    family = payload["family"]
    if family not in ROLLING_FAMILIES:
        raise ValueError("V2 family is unsupported")
    action = payload["action"]
    if action not in ROLLING_ACTIONS:
        raise ValueError("V2 action is unsupported")
    anchor = normalize_iso_date(payload["anchor_date"], "anchor_date")
    targets = payload["target_keys"]
    windows = payload["window_keys"]
    if not isinstance(targets, list) or not isinstance(windows, list):
        raise ValueError("V2 target_keys/window_keys must be lists")
    for key in targets:
        if RollingTargetIdentity.parse(key).family != family:
            raise ValueError("V2 target family does not match envelope family")
    for key in windows:
        parse_rolling_window_key(key, expected_family=family)
    if len(targets) != len(set(targets)) or len(windows) != len(set(windows)):
        raise KeyError("V2 target/window identity is ambiguous")
    attempt = payload["attempt_id"]
    if attempt is not None and (not isinstance(attempt, str) or not attempt.strip()):
        raise ValueError("attempt_id must be null or a non-empty string")
    phase = payload["phase"]
    if phase not in _V2_PHASES:
        raise ValueError("V2 phase is unsupported")
    extensions = payload.get("extensions", {})
    if not isinstance(extensions, dict):
        raise ValueError("V2 extensions must be a mapping")
    raw_units = payload["units"]
    if not isinstance(raw_units, list):
        raise ValueError("V2 units must be a list")
    units = []
    identities = set()
    ordered_scope = [
        (target, window) for target in targets for window in windows
    ]
    scope = set(ordered_scope)
    last_position = -1
    for raw in raw_units:
        if not isinstance(raw, dict) or set(raw) - _UNIT_FIELDS:
            raise ValueError("V2 unit has an invalid shape")
        if not {"target_key", "window_key", "status"}.issubset(raw):
            raise ValueError("V2 unit is missing identity/status")
        identity = (raw["target_key"], raw["window_key"])
        if identity not in scope:
            raise ValueError("V2 unit identity is outside envelope scope")
        if identity in identities:
            raise KeyError("V2 unit identity is duplicated")
        identities.add(identity)
        position = ordered_scope.index(identity)
        if position <= last_position:
            raise KeyError("V2 unit identities are out of canonical order")
        last_position = position
        status_value = raw["status"]
        if status_value not in _UNIT_STATUSES:
            raise ValueError("V2 unit status is unsupported")
        record_id = raw.get("record_id")
        evidence_id = raw.get("evidence_id")
        for value, label in ((record_id, "record_id"), (evidence_id, "evidence_id")):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("V2 unit %s must be null or non-empty" % label)
        if "extensions" in raw and not isinstance(raw["extensions"], dict):
            raise ValueError("V2 unit extensions must be a mapping")
        unit_extensions = raw.get("extensions")
        units.append(RollingStateUnitClaim(
            target_key=identity[0], window_key=identity[1], status=status_value,
            record_id=record_id, evidence_id=evidence_id,
            _extensions_json=(
                json.dumps(
                    unit_extensions, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True,
                ) if unit_extensions is not None else None
            ),
        ))
    return RollingStateV2Snapshot(
        workspace_fingerprint=payload["workspace_fingerprint"], run_id=run_id,
        family=family, action=action,
        plan_fingerprint=payload["plan_fingerprint"],
        execution_fingerprint=payload["execution_fingerprint"],
        config_fingerprint=payload["config_fingerprint"],
        anchor_date=anchor, target_keys=tuple(targets), window_keys=tuple(windows),
        attempt_id=attempt, phase=phase, units=tuple(units),
        _extensions_json=json.dumps(
            extensions, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ),
    )


def parse_rolling_state_v2_bytes(data):
    """Parse strict State V2 bytes through the canonical schema owner."""

    if not isinstance(data, bytes):
        raise RollingIdentityError("Rolling State V2 payload must be bytes")
    try:
        payload = _strict_json_bytes(data)
        if not isinstance(payload, dict):
            raise ValueError("State V2 root must be a mapping")
        if type(payload.get("schema_version")) is not int:
            raise ValueError("State V2 schema_version must be an integer")
        if payload.get("schema_version") != 2:
            raise ValueError("State V2 schema_version is unsupported")
        return _parse_v2(payload)
    except (KeyError, TypeError, UnicodeError, ValueError, json.JSONDecodeError,
            RollingIdentityError) as exc:
        if isinstance(exc, RollingIdentityError):
            raise
        raise RollingIdentityError("invalid Rolling State V2: %s" % exc)


def serialize_rolling_state_v2(snapshot):
    """Return deterministic canonical bytes after aggregate revalidation."""

    if not isinstance(snapshot, RollingStateV2Snapshot):
        raise RollingIdentityError("State V2 serializer requires a typed snapshot")
    try:
        payload = snapshot.to_public_dict()
        validated = _parse_v2(payload)
        canonical = validated.to_public_dict()
    except (KeyError, TypeError, ValueError, RollingIdentityError) as exc:
        if isinstance(exc, RollingIdentityError):
            raise
        raise RollingIdentityError("invalid Rolling State V2 snapshot: %s" % exc)
    return (
        json.dumps(canonical, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class RollingStateMigrationProposal:
    status: str
    reason_code: str
    capability: str
    source_path: str
    source_fingerprint: str = None
    proposed_snapshot: object = None
    _proposed_bytes: bytes = None
    proposed_fingerprint: str = None
    warnings: tuple = ()
    blockers: tuple = ()

    def __post_init__(self):
        if self.status not in ("candidate", "blocked", "not_applicable"):
            raise RollingIdentityError("unsupported migration proposal status")
        if self.capability not in ("proposal_only", "none"):
            raise RollingIdentityError("unsupported migration proposal capability")
        if self.status == "candidate":
            if (self.source_path not in (
                    "data/rolling_state.json",
                    "data/rolling_state_cpcv.json",
                ) or not _is_digest(self.source_fingerprint)
                    or self.capability != "proposal_only"
                    or not isinstance(self.proposed_snapshot,
                                      RollingStateV2Snapshot)
                    or not isinstance(self._proposed_bytes, bytes)
                    or hashlib.sha256(self._proposed_bytes).hexdigest()
                    != self.proposed_fingerprint):
                raise RollingIdentityError("candidate migration proposal is incomplete")
            parsed = parse_rolling_state_v2_bytes(self._proposed_bytes)
            if parsed.to_public_dict() != self.proposed_snapshot.to_public_dict():
                raise RollingIdentityError("migration proposal postimage facts disagree")
            if self.reason_code != "rolling_state_migration_candidate":
                raise RollingIdentityError("migration proposal reason contradicts status")
        elif (self.proposed_snapshot is not None or self._proposed_bytes is not None
              or self.proposed_fingerprint is not None):
            raise RollingIdentityError("blocked migration proposal exposes a postimage")
        elif self.capability != "none":
            raise RollingIdentityError("non-candidate migration cannot grant capability")

    @property
    def proposed_bytes(self):
        return self._proposed_bytes if self.status == "candidate" else None

    def to_public_dict(self):
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "capability": self.capability,
            "source_path": self.source_path,
            "source_fingerprint": self.source_fingerprint,
            "proposed_fingerprint": self.proposed_fingerprint,
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
        }


def build_legacy_migration_proposal(
        inspection, workspace_identity, run_id, family, action,
        plan_fingerprint, execution_fingerprint, config_fingerprint,
        anchor_date, target_keys, window_keys, index_window_keys,
        attempt_id="legacy-migration-proposal"):
    """Build a deterministic, read-only legacy-to-V2 postimage audit."""

    source_path = getattr(inspection, "path", "rolling_state.json")
    source_fingerprint = getattr(inspection, "fingerprint", None)

    def blocked(reason, message, status="blocked"):
        return RollingStateMigrationProposal(
            status=status,
            reason_code=reason,
            capability="none",
            source_path=source_path,
            source_fingerprint=source_fingerprint,
            blockers=(message,),
        )

    if inspection.classification == "missing":
        return blocked(
            "rolling_state_migration_not_applicable",
            "missing state has no legacy bytes to migrate",
            status="not_applicable",
        )
    if (inspection.classification != "valid_legacy"
            or not isinstance(inspection.snapshot, LegacyRollingStateSnapshot)):
        return blocked(
            "rolling_state_migration_blocked",
            "only valid legacy state can produce a migration proposal",
        )
    snapshot = inspection.snapshot
    try:
        if family != snapshot.family:
            raise ValueError("legacy family does not match explicit family")
        if snapshot.anchor_date is not None and snapshot.anchor_date != anchor_date:
            raise ValueError("legacy anchor does not match explicit anchor")
        if not _is_digest(workspace_identity):
            raise ValueError("workspace identity must be a lowercase SHA-256")
        targets = tuple(target_keys)
        windows = tuple(window_keys)
        for target in targets:
            if RollingTargetIdentity.parse(target).family != family:
                raise ValueError("migration target family is foreign")
        for window in windows:
            parse_rolling_window_key(window, expected_family=family)
        if len(targets) != len(set(targets)) or len(windows) != len(set(windows)):
            raise ValueError("migration scope contains duplicate identities")
        if not isinstance(index_window_keys, dict):
            raise ValueError("legacy index mapping must be a mapping")
        if set(index_window_keys) != set(range(len(windows))):
            raise ValueError("legacy index mapping must cover the explicit window scope")
        if any(index_window_keys[index] != windows[index]
               for index in range(len(windows))):
            raise ValueError("legacy index mapping does not preserve window order")
        requested_scope = [
            (target, window) for target in targets for window in windows
        ]
        legacy_claims = {}
        for index, models in snapshot.completed_windows:
            if index not in index_window_keys:
                raise ValueError("legacy completed index is outside explicit windows")
            window_key = index_window_keys[index]
            for target_key, record_id in models:
                identity = (target_key, window_key)
                if identity not in requested_scope:
                    raise ValueError("legacy completed target is outside explicit scope")
                legacy_claims[identity] = record_id
        units = []
        for identity in requested_scope:
            if identity not in legacy_claims:
                continue
            units.append(RollingStateUnitClaim(
                target_key=identity[0],
                window_key=identity[1],
                status="success",
                record_id=legacy_claims[identity],
                evidence_id=None,
                _extensions_json=json.dumps(
                    {"claim_authority": "legacy_unverified"},
                    sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                ),
            ))
        proposed = RollingStateV2Snapshot(
            workspace_fingerprint=workspace_identity,
            run_id=run_id,
            family=family,
            action=action,
            plan_fingerprint=plan_fingerprint,
            execution_fingerprint=execution_fingerprint,
            config_fingerprint=config_fingerprint,
            anchor_date=anchor_date,
            target_keys=targets,
            window_keys=windows,
            attempt_id=attempt_id,
            phase="executing",
            units=tuple(units),
            _extensions_json=json.dumps(
                {"migration": "legacy_proposal_only"},
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            ),
        )
        proposed_bytes = serialize_rolling_state_v2(proposed)
    except (KeyError, TypeError, ValueError, RollingIdentityError) as exc:
        return blocked("rolling_state_migration_blocked", str(exc))
    return RollingStateMigrationProposal(
        status="candidate",
        reason_code="rolling_state_migration_candidate",
        capability="proposal_only",
        source_path=source_path,
        source_fingerprint=source_fingerprint,
        proposed_snapshot=parse_rolling_state_v2_bytes(proposed_bytes),
        _proposed_bytes=proposed_bytes,
        proposed_fingerprint=hashlib.sha256(proposed_bytes).hexdigest(),
        warnings=(
            "legacy completion claims remain unverified and cannot be applied",
        ),
    )


def _compare_expectation(snapshot, expectation):
    if expectation is None:
        return None, (), ()
    checked = []
    foreign = []
    mismatch = []
    for field in _EXPECTATION_FIELDS:
        expected = getattr(expectation, field)
        if expected is None:
            continue
        checked.append(field)
        actual = getattr(snapshot, field, None)
        if field in ("target_keys", "window_keys"):
            actual = tuple(actual) if actual is not None else None
            expected = tuple(expected)
        if actual != expected:
            message = "%s does not match the selected Rolling identity" % field
            if field in ("workspace_fingerprint", "family", "attempt_id"):
                foreign.append(message)
            else:
                mismatch.append(message)
    if foreign:
        return "foreign", tuple(checked), tuple(foreign)
    if mismatch:
        return "identity_mismatch", tuple(checked), tuple(mismatch)
    return None, tuple(checked), ()


def inspect_rolling_state(path, workspace_root, expectation=None):
    """Inspect exactly one state byte snapshot without locks or mutation."""

    path = Path(path).expanduser()
    supplied_root = Path(workspace_root).expanduser().absolute()
    root = supplied_root.resolve()
    if not path.is_absolute():
        path = supplied_root / path
    path = path.absolute()
    try:
        relative = path.relative_to(supplied_root).as_posix()
    except ValueError:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            return _inspection(
                "foreign", "unknown", "rolling_state_foreign", "blocked", "blocked",
                path.name, "outside_workspace",
                blockers=("state path is outside workspace",),
            )
    try:
        path.parent.resolve(strict=True).relative_to(root)
    except ValueError:
        return _inspection(
            "foreign", "unknown", "rolling_state_foreign", "blocked", "blocked",
            relative, "parent_symlink",
            blockers=("state parent resolves outside workspace",),
        )
    except OSError:
        pass
    try:
        metadata = os.lstat(str(path))
    except FileNotFoundError:
        return _inspection(
            "missing", "none", "rolling_state_missing", "absent",
            "not_applicable", relative, "missing",
        )
    except OSError as exc:
        return _inspection(
            "corrupt", "unknown", "rolling_state_corrupt", "blocked", "blocked",
            relative, "unreadable", blockers=(
                "state metadata cannot be read: %s" % exc.__class__.__name__,
            ),
        )
    try:
        physical = path.resolve(strict=True)
        physical.relative_to(root)
    except (OSError, ValueError):
        return _inspection(
            "foreign", "unknown", "rolling_state_foreign", "blocked", "blocked",
            relative, "symlink" if stat.S_ISLNK(metadata.st_mode) else "physical_escape",
            blockers=("state path resolves outside the workspace",),
        )
    if stat.S_ISLNK(metadata.st_mode):
        path_kind = "symlink"
    elif stat.S_ISREG(metadata.st_mode):
        path_kind = "file"
    elif stat.S_ISDIR(metadata.st_mode):
        return _inspection(
            "corrupt", "unknown", "rolling_state_corrupt", "blocked", "blocked",
            relative, "directory", blockers=("state path is a directory",),
        )
    else:
        return _inspection(
            "corrupt", "unknown", "rolling_state_corrupt", "blocked", "blocked",
            relative, "special", blockers=("state path is not a regular file",),
        )
    try:
        with path.open("rb") as handle:
            data = handle.read()
    except OSError as exc:
        return _inspection(
            "corrupt", "unknown", "rolling_state_corrupt", "blocked", "blocked",
            relative, path_kind, blockers=(
                "state bytes cannot be read: %s" % exc.__class__.__name__,
            ),
        )
    return inspect_rolling_state_bytes(
        data,
        relative_path=relative,
        workspace_root=root,
        expectation=expectation,
        path_kind=path_kind,
    )


def inspect_rolling_state_bytes(data, relative_path, workspace_root,
                                expectation=None, path_kind="file"):
    """Classify one already-observed byte snapshot without another read."""

    relative = Path(relative_path).as_posix()
    if data is None:
        return _inspection(
            "missing", "none", "rolling_state_missing", "absent",
            "not_applicable", relative, "missing",
        )
    if not isinstance(data, bytes):
        raise RollingIdentityError("Rolling state observation must be bytes")
    root = Path(workspace_root).expanduser().resolve()
    path = Path(relative)
    fingerprint = hashlib.sha256(data).hexdigest()
    if not data:
        return _inspection(
            "corrupt", "unknown", "rolling_state_corrupt", "blocked", "blocked",
            relative, path_kind, fingerprint,
            blockers=("existing state file is empty",),
        )
    try:
        payload = _strict_json_bytes(data)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return _inspection(
            "corrupt", "unknown", "rolling_state_corrupt", "blocked", "blocked",
            relative, path_kind, fingerprint,
            blockers=("state bytes are not strict JSON: %s" % exc.__class__.__name__,),
        )
    if not isinstance(payload, dict):
        return _unsupported(
            relative, path_kind, fingerprint, "state root must be a mapping",
        )
    if "schema_version" not in payload:
        family = _family_from_path(path)
        if expectation is not None and expectation.family is not None:
            if family is not None and family != expectation.family:
                return _inspection(
                    "foreign", "legacy_unversioned", "rolling_state_foreign",
                    "blocked", "blocked", relative, path_kind, fingerprint,
                    blockers=("state filename family conflicts with expectation",),
                )
            family = expectation.family
        if family not in ROLLING_FAMILIES:
            return _unsupported(
                relative, path_kind, fingerprint,
                "legacy state family cannot be determined",
                protocol="legacy_unversioned",
            )
        try:
            snapshot = _parse_legacy(payload, family)
        except KeyError as exc:
            return _inspection(
                "ambiguous", "legacy_unversioned", "rolling_state_ambiguous",
                "blocked", "blocked", relative, path_kind, fingerprint,
                blockers=(str(exc),),
            )
        except LookupError as exc:
            return _inspection(
                "foreign", "legacy_unversioned", "rolling_state_foreign",
                "blocked", "blocked", relative, path_kind, fingerprint,
                blockers=(str(exc),),
            )
        except (TypeError, ValueError, RollingIdentityError) as exc:
            return _unsupported(
                relative, path_kind, fingerprint, str(exc),
                protocol="legacy_unversioned",
            )
        checked = []
        blockers = []
        if (expectation is not None
                and expectation.config_fingerprint is not None
                and snapshot.rolling_config_fingerprint is not None):
            checked.append("config_fingerprint")
            if snapshot.rolling_config_fingerprint != expectation.config_fingerprint:
                blockers.append("legacy rolling_config does not match expectation")
        if blockers:
            return _inspection(
                "identity_mismatch", "legacy_unversioned",
                "rolling_state_identity_mismatch", "blocked", "blocked",
                relative, path_kind, fingerprint, snapshot, checked=checked,
                blockers=blockers,
            )
        migration = (
            "candidate" if snapshot.completed_window_count == 0
            else "deferred_requires_windows"
        )
        warnings = ["legacy completion claims are not immutable evidence"]
        warnings.extend(
            "unknown legacy field retained: %s" % item
            for item in snapshot.unknown_fields
        )
        return _inspection(
            "valid_legacy", "legacy_unversioned", "rolling_state_legacy_valid",
            "legacy_compatible", migration, relative, path_kind, fingerprint,
            snapshot, checked=checked, warnings=warnings,
        )
    version = payload.get("schema_version")
    if type(version) is not int or version != 2:
        return _unsupported(
            relative, path_kind, fingerprint,
            "state schema_version is unsupported",
        )
    try:
        snapshot = _parse_v2(payload)
    except KeyError as exc:
        return _inspection(
            "ambiguous", "rolling_state_v2", "rolling_state_ambiguous",
            "blocked", "blocked", relative, path_kind, fingerprint,
            blockers=(str(exc),),
        )
    except (TypeError, ValueError, RollingIdentityError) as exc:
        return _unsupported(
            relative, path_kind, fingerprint, str(exc),
            protocol="rolling_state_v2",
        )
    path_family = _family_from_path(path)
    if path_family is not None and path_family != snapshot.family:
        return _inspection(
            "foreign", "rolling_state_v2", "rolling_state_foreign", "blocked",
            "blocked", relative, path_kind, fingerprint, snapshot,
            blockers=("V2 family conflicts with selected state filename",),
        )
    expected = expectation or RollingStateExpectation()
    if expected.workspace_fingerprint is None:
        expected = RollingStateExpectation(
            workspace_fingerprint=workspace_fingerprint(root),
            family=expected.family,
            config_fingerprint=expected.config_fingerprint,
            target_keys=expected.target_keys,
            window_keys=expected.window_keys,
            run_id=expected.run_id,
            execution_fingerprint=expected.execution_fingerprint,
            attempt_id=expected.attempt_id,
        )
    mismatch_class, checked, mismatch_blockers = _compare_expectation(snapshot, expected)
    if mismatch_class is not None:
        reason = (
            "rolling_state_foreign" if mismatch_class == "foreign"
            else "rolling_state_identity_mismatch"
        )
        return _inspection(
            mismatch_class, "rolling_state_v2", reason, "blocked", "blocked",
            relative, path_kind, fingerprint, snapshot, checked=checked,
            blockers=mismatch_blockers,
        )
    completion_claim = (
        snapshot.phase in ("units_complete", "completed")
        or any(item.status in ("success", "completed") for item in snapshot.units)
    )
    if completion_claim:
        return _inspection(
            "unverified_completion", "rolling_state_v2",
            "rolling_state_completion_unverified", "blocked",
            "not_applicable", relative, path_kind, fingerprint, snapshot,
            checked=checked,
            blockers=("state completion claim has no immutable evidence authority",),
        )
    return _inspection(
        "valid_versioned", "rolling_state_v2", "rolling_state_versioned_valid",
        "future_repository", "not_applicable", relative, path_kind, fingerprint,
        snapshot, checked=checked,
        warnings=(
            "V2 CAS repository is available; legacy execution integration remains blocked",
        ),
    )
