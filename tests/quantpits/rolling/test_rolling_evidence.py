import builtins
import hashlib
import io
import os
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quantpits.rolling import (
    RollingArtifactExpectation,
    RollingArtifactObservation,
    RollingEvidenceContractError,
    RollingEvidenceSetInspection,
    RollingPredictionCoverage,
    RollingRunIdentity,
    RollingTargetIdentity,
    RollingUnitEvidenceInspection,
    RollingUnitEvidenceRequest,
    RollingWindowIdentity,
    inspect_rolling_evidence,
    workspace_fingerprint,
)
from quantpits.rolling.evidence import RollingOrphanObservation
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value


DIGEST = "a" * 64


class MaliciousPrediction:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.system, ("touch %s" % self.marker,)


class FakeEvidenceBackend:
    def __init__(self, identity, candidates=(), *, drift=False, fail_inventory=False):
        self.identity = dict(identity)
        self.candidates = tuple(candidates)
        self.drift = drift
        self.fail_inventory = fail_inventory
        self.inventory_calls = 0
        self.mutation_calls = []

    def tracking_identity(self):
        return dict(self.identity)

    def inventory(self, requests):
        self.inventory_calls += 1
        if self.fail_inventory:
            raise RuntimeError("metadata unavailable")
        token = fingerprint_value({"inventory": ("drift-a" if self.inventory_calls % 2 else "drift-b") if self.drift else "stable"})
        return {"fingerprint": token, "candidates": self.candidates}

    def create(self, *args):
        self.mutation_calls.append(("create", args))
        raise AssertionError("read-only inspector called create")

    def write(self, *args):
        self.mutation_calls.append(("write", args))
        raise AssertionError("read-only inspector called write")

    def download(self, *args):
        self.mutation_calls.append(("download", args))
        raise AssertionError("read-only inspector called download")


def prediction_bytes(sessions=("2026-01-05", "2026-01-06"), *, duplicate=False, nonfinite=False):
    tuples = []
    values = []
    for offset, session in enumerate(sessions):
        tuples.append((pd.Timestamp(session), "asset-%s" % offset))
        values.append(float(offset + 1))
    if duplicate:
        tuples.append(tuples[0])
        values.append(3.0)
    if nonfinite:
        values[-1] = float("nan")
    frame = pd.DataFrame(
        {"score": values},
        index=pd.MultiIndex.from_tuples(tuples, names=("datetime", "instrument")),
    )
    buffer = io.BytesIO()
    frame.to_pickle(buffer)
    return buffer.getvalue()


def make_window(family="rolling"):
    if family == "rolling":
        return RollingWindowIdentity(
            family="rolling", train_start="2025-01-01", train_end="2025-10-31",
            valid_start="2025-11-01", valid_end="2025-12-31",
            test_start="2026-01-05", test_end="2026-01-06",
            effective_config_fingerprint=DIGEST,
        )
    from quantpits.rolling import RollingFoldIdentity
    fold = RollingFoldIdentity(
        train_segments=(("2025-01-01", "2025-05-31"),),
        valid_start="2025-06-01", valid_end="2025-10-31",
    )
    return RollingWindowIdentity(
        family="cpcv_rolling", train_start="2025-01-01", train_end="2025-10-31",
        test_start="2026-01-05", test_end="2026-01-06",
        effective_config_fingerprint=DIGEST, folds=(fold,),
    )


def make_request(root, *, family="rolling", model="alpha", protocol="execution_bound_v1", data=None, supporting=False):
    root = Path(root).resolve()
    window = make_window(family)
    target = RollingTargetIdentity(model, family).target_key
    run = RollingRunIdentity(
        workspace_fingerprint=workspace_fingerprint(root), family=family,
        action="merge", plan_fingerprint="b" * 64,
        config_fingerprint="c" * 64, anchor_date="2026-01-06",
        target_keys=(target,), window_keys=(window.window_key,),
        runtime_params_fingerprint="d" * 64, attempt_id="attempt-1",
    )
    data = prediction_bytes() if data is None else data
    artifacts = [RollingArtifactExpectation(
        "pred.pkl", "prediction", len(data), hashlib.sha256(data).hexdigest(),
    )]
    if supporting:
        artifacts.append(RollingArtifactExpectation(
            "model.pkl", "supporting", 5, hashlib.sha256(b"model").hexdigest(),
        ))
    request = RollingUnitEvidenceRequest(
        run, target, window, protocol, target, "merge", "rolling-exp", "exp-1",
        "recorder-1", tuple(artifacts), ("2026-01-05", "2026-01-06"),
    )
    return request, data


def make_candidate(request, artifact_root, *, overrides=None):
    payload = {
        "workspace_fingerprint": request.run_identity.workspace_fingerprint,
        "backend_fingerprint": "e" * 64,
        "experiment_name": request.experiment_name,
        "experiment_id": request.experiment_id,
        "recorder_id": request.recorder_id,
        "run_fingerprint": request.run_identity.fingerprint,
        "attempt_id": request.run_identity.attempt_id,
        "plan_fingerprint": request.run_identity.plan_fingerprint,
        "config_fingerprint": request.run_identity.config_fingerprint,
        "target_key": request.target_key,
        "window_key": request.window_key,
        "source_protocol": request.source_protocol,
        "source_publication_key": request.source_publication_key,
        "source_operation": request.source_operation,
        "source_manifest_fingerprint": request.source_manifest_fingerprint,
        "artifact_root_uri": Path(artifact_root).resolve().as_uri(),
    }
    payload.update(overrides or {})
    return payload


def make_valid_case(tmp_path, *, family="rolling", protocol="execution_bound_v1", data=None, supporting=False):
    root = (tmp_path / "workspace").resolve()
    artifact_root = root / "mlruns" / "exp-1" / "recorder-1" / "artifacts"
    artifact_root.mkdir(parents=True)
    request, data = make_request(root, family=family, protocol=protocol, data=data, supporting=supporting)
    (artifact_root / "pred.pkl").write_bytes(data)
    if supporting:
        (artifact_root / "model.pkl").write_bytes(b"model")
    candidate = make_candidate(request, artifact_root)
    identity = {
        "workspace_fingerprint": request.run_identity.workspace_fingerprint,
        "backend_fingerprint": "e" * 64,
        "present": True,
        "contained": True,
    }
    return WorkspaceContext.from_root(root), request, candidate, FakeEvidenceBackend(identity, (candidate,))


def inspect_case(tmp_path, **kwargs):
    context, request, candidate, backend = make_valid_case(tmp_path, **kwargs)
    return request, candidate, backend, inspect_rolling_evidence(context, (request,), backend)


def missing_result(request):
    return RollingUnitEvidenceInspection(
        request.unit_key, "missing", "rolling_evidence_missing",
        request.source_protocol, request.source_manifest_fingerprint,
        ("candidate_cardinality",), blockers=("missing",),
    )


def test_request_rejects_closest_invalid_representations(tmp_path):
    request, _ = make_request(tmp_path.resolve())
    for field, value in (
        ("source_publication_key", ""), ("experiment_name", None),
        ("experiment_id", 1), ("recorder_id", " recorder "),
        ("source_protocol", "EXECUTION_BOUND_V1"),
    ):
        with pytest.raises(RollingEvidenceContractError):
            replace(request, **{field: value})
    with pytest.raises(RollingEvidenceContractError):
        RollingArtifactExpectation("pred.pkl", "prediction", True, "a" * 64)


def test_artifact_expectation_rejects_unsafe_keys_and_non_strict_counts():
    for key in ("", ".", "..", "../pred.pkl", "/pred.pkl", "a\\pred.pkl", "a/../pred.pkl", "x\x00p"):
        with pytest.raises(RollingEvidenceContractError):
            RollingArtifactExpectation(key, "prediction", 1, "a" * 64)
    for size in (True, -1, 1.0, "1"):
        with pytest.raises(RollingEvidenceContractError):
            RollingArtifactExpectation("pred.pkl", "prediction", size, "a" * 64)
    for digest in ("A" * 64, "a" * 63, "a" * 65, None):
        with pytest.raises(RollingEvidenceContractError):
            RollingArtifactExpectation("pred.pkl", "prediction", 1, digest)


def test_request_rejects_duplicate_or_out_of_scope_units_and_sessions(tmp_path):
    request, _ = make_request(tmp_path.resolve())
    with pytest.raises(RollingEvidenceContractError):
        replace(request, expected_prediction_sessions=("2026-01-05", "2026-01-05"))
    with pytest.raises(RollingEvidenceContractError):
        replace(request, expected_prediction_sessions=("2026-01-06", "2026-01-05"))
    with pytest.raises(RollingEvidenceContractError):
        inspect_rolling_evidence(WorkspaceContext.from_root(tmp_path), (request, request), object())
    with pytest.raises(RollingEvidenceContractError):
        replace(request, target_key="outside@rolling")


def test_request_requires_exact_window_identity_and_attempt_for_bound_source(tmp_path):
    request, _ = make_request(tmp_path.resolve())
    with pytest.raises(RollingEvidenceContractError):
        replace(request, run_identity=replace(request.run_identity, attempt_id=None))
    other = replace(request.window_identity, effective_config_fingerprint="f" * 64)
    with pytest.raises(RollingEvidenceContractError):
        replace(request, window_identity=other)


def test_non_unit_actions_cannot_claim_bound_per_window_evidence(tmp_path):
    request, _ = make_request(tmp_path.resolve())
    for action in ("predict_only", "backtest_only", "clear_state", "show_state"):
        with pytest.raises(RollingEvidenceContractError):
            replace(request, run_identity=replace(request.run_identity, action=action))


def test_exact_source_artifacts_and_prediction_coverage_are_valid(tmp_path):
    request, _, _, evidence = inspect_case(tmp_path)
    unit = evidence.unit_results[0]
    assert unit.classification == "valid"
    assert unit.reason_code == "rolling_evidence_valid"
    assert unit.prediction_coverage.expected_sessions == ("2026-01-05", "2026-01-06")
    assert unit.prediction_coverage.observed_sessions == unit.prediction_coverage.expected_sessions
    assert {"prediction_schema", "prediction_session_coverage", "artifact_byte_fingerprint"}.issubset(unit.checked)
    assert unit.capabilities == ("render", "immutable_summary", "reuse_proposal", "recovery_proposal")
    assert evidence.n_requested == evidence.n_valid == 1
    series = pd.Series(
        (1.0, 2.0), name="score",
        index=pd.MultiIndex.from_tuples(
            ((pd.Timestamp("2026-01-05"), "asset-a"), (pd.Timestamp("2026-01-06"), "asset-b")),
            names=("datetime", "instrument"),
        ),
    )
    buffer = io.BytesIO()
    series.to_pickle(buffer)
    context, series_request, _, backend = make_valid_case(
        tmp_path / "series", data=buffer.getvalue(),
    )
    series_result = inspect_rolling_evidence(context, (series_request,), backend)
    assert series_result.unit_results[0].classification == "valid"


def test_evidence_fingerprint_is_deterministic_and_path_private(tmp_path):
    context, request, _, backend = make_valid_case(tmp_path)
    first = inspect_rolling_evidence(context, (request,), backend)
    second = inspect_rolling_evidence(context, (request,), backend)
    assert first.unit_results[0].evidence_fingerprint == second.unit_results[0].evidence_fingerprint
    rendered = repr(first.unit_results[0].to_public_dict())
    assert str(context.root) not in rendered
    assert "artifact_root_uri" not in rendered
    assert not hasattr(first.unit_results[0], "raw_bytes")
    assert not hasattr(first.unit_results[0], "backend")


def test_slide_and_cpcv_share_rules_without_inferring_wrapper_capability(tmp_path):
    for family in ("rolling", "cpcv_rolling"):
        context, request, _, backend = make_valid_case(tmp_path / family, family=family, supporting=True)
        result = inspect_rolling_evidence(context, (request,), backend).unit_results[0]
        assert result.classification == "valid"
        assert len(result.artifact_observations) == 2
        assert "execute" not in result.capabilities
        assert "predict" not in result.capabilities


def test_missing_candidate_is_terminal_without_member_loss(tmp_path):
    context, request, _, backend = make_valid_case(tmp_path)
    backend.candidates = ()
    evidence = inspect_rolling_evidence(context, (request,), backend)
    assert evidence.requested_unit_keys == (request.unit_key,)
    assert evidence.unit_results[0].classification == "missing"
    backend.fail_inventory = True
    unavailable = inspect_rolling_evidence(context, (request,), backend)
    assert unavailable.requested_unit_keys == (request.unit_key,)
    assert unavailable.unit_results[0].classification == "not_comparable"


def test_duplicate_candidates_fail_closed_without_selecting_first(tmp_path):
    context, request, candidate, backend = make_valid_case(tmp_path)
    backend.candidates = (candidate, dict(candidate))
    evidence = inspect_rolling_evidence(context, (request,), backend)
    unit = evidence.unit_results[0]
    assert unit.classification == "duplicate"
    assert unit.candidate_count == 2
    assert evidence.n_candidates == 2
    assert unit.capabilities == ("render",)
    assert not unit.artifact_observations


def test_foreign_backend_or_artifact_escape_denies_capability(tmp_path):
    context, request, candidate, backend = make_valid_case(tmp_path)
    backend.identity["foreign"] = True
    assert inspect_rolling_evidence(context, (request,), backend).unit_results[0].classification == "foreign"
    outside = tmp_path / "outside"
    outside.mkdir()
    candidate["artifact_root_uri"] = outside.resolve().as_uri()
    backend.identity.pop("foreign")
    unit = inspect_rolling_evidence(context, (request,), backend).unit_results[0]
    assert unit.classification == "foreign"
    assert "reuse_proposal" not in unit.capabilities
    outside_artifacts = outside / "artifacts"
    outside_artifacts.mkdir()
    link = context.root / "contained-looking"
    link.symlink_to(outside_artifacts, target_is_directory=True)
    candidate["artifact_root_uri"] = link.as_uri()
    unit = inspect_rolling_evidence(context, (request,), backend).unit_results[0]
    assert unit.classification == "foreign"


def test_source_recorder_or_unit_identity_mismatch_denies_capability(tmp_path):
    for field, value in (("recorder_id", "foreign-recorder"), ("attempt_id", "other-attempt"), ("source_manifest_fingerprint", "f" * 64)):
        context, request, candidate, backend = make_valid_case(tmp_path / field)
        candidate[field] = value
        unit = inspect_rolling_evidence(context, (request,), backend).unit_results[0]
        assert unit.classification == "identity_mismatch"
        assert unit.capabilities == ("render",)
    uri_shape = "file:" + "///" + "redacted/path"
    windows_shape = "Z:" + "\\" + "redacted\\path"
    for index, value in enumerate((True, 7, object(), uri_shape, windows_shape)):
        context, request, candidate, backend = make_valid_case(tmp_path / ("malformed-%s" % index))
        candidate["recorder_id"] = value
        evidence = inspect_rolling_evidence(context, (request,), backend)
        assert len(evidence.unit_results) == 1
        assert evidence.unit_results[0].classification == "identity_mismatch"
        rendered = repr(evidence.to_public_dict())
        assert uri_shape not in rendered
        assert windows_shape not in rendered


def test_missing_required_artifact_is_partial(tmp_path):
    context, request, _, backend = make_valid_case(tmp_path)
    (context.root / "mlruns" / "exp-1" / "recorder-1" / "artifacts" / "pred.pkl").unlink()
    unit = inspect_rolling_evidence(context, (request,), backend).unit_results[0]
    assert unit.classification == "partial"
    assert any(item.status == "missing" for item in unit.artifact_observations)
    assert "artifact_node_kind" not in unit.checked
    assert "artifact_byte_fingerprint" not in unit.checked


@pytest.mark.parametrize("mode", ("digest", "size", "decode", "duplicate", "nonfinite", "malicious"))
def test_digest_size_decode_index_and_nonfinite_failures_are_corrupt(tmp_path, mode):
    marker = tmp_path / "pickle-side-effect"
    data = (
        pickle.dumps(MaliciousPrediction(marker))
        if mode == "malicious"
        else prediction_bytes(duplicate=mode == "duplicate", nonfinite=mode == "nonfinite")
    )
    context, request, candidate, backend = make_valid_case(tmp_path / mode, data=data)
    artifact = context.root / "mlruns" / "exp-1" / "recorder-1" / "artifacts" / "pred.pkl"
    if mode == "digest":
        artifact.write_bytes(data[:-1] + bytes((data[-1] ^ 1,)))
    elif mode == "size":
        request = replace(request, artifacts=(replace(request.artifacts[0], size_bytes=len(data) + 1),))
        backend.candidates = (make_candidate(request, artifact.parent),)
    elif mode == "decode":
        bad = b"not-a-pickle"
        artifact.write_bytes(bad)
        request = replace(request, artifacts=(RollingArtifactExpectation("pred.pkl", "prediction", len(bad), hashlib.sha256(bad).hexdigest()),))
        backend.candidates = (make_candidate(request, artifact.parent),)
    unit = inspect_rolling_evidence(context, (request,), backend).unit_results[0]
    assert unit.classification == "corrupt"
    assert unit.capabilities == ("render",)
    assert not marker.exists()
    if mode in ("decode", "malicious"):
        assert not {
            "prediction_schema", "prediction_index_unique",
            "prediction_scores_finite", "prediction_session_coverage",
        }.intersection(unit.checked)
    elif mode == "duplicate":
        assert "prediction_index_unique" in unit.checked
        assert "prediction_scores_finite" not in unit.checked
        assert "prediction_session_coverage" not in unit.checked
    elif mode == "nonfinite":
        assert "prediction_scores_finite" in unit.checked
        assert "prediction_session_coverage" not in unit.checked


def test_artifact_symlink_or_special_node_never_becomes_valid(tmp_path, monkeypatch):
    context, request, _, backend = make_valid_case(tmp_path / "symlink")
    artifact = context.root / "mlruns" / "exp-1" / "recorder-1" / "artifacts" / "pred.pkl"
    original = artifact.read_bytes()
    artifact.unlink()
    target = artifact.parent / "real.pkl"
    target.write_bytes(original)
    artifact.symlink_to(target.name)
    assert inspect_rolling_evidence(context, (request,), backend).unit_results[0].classification == "corrupt"
    context2, request2, _, backend2 = make_valid_case(tmp_path / "directory")
    artifact2 = context2.root / "mlruns" / "exp-1" / "recorder-1" / "artifacts" / "pred.pkl"
    artifact2.unlink()
    artifact2.mkdir()
    assert inspect_rolling_evidence(context2, (request2,), backend2).unit_results[0].classification == "corrupt"
    context3, request3, _, backend3 = make_valid_case(tmp_path / "ancestor-race")
    artifact3 = context3.root / "mlruns" / "exp-1" / "recorder-1" / "artifacts" / "pred.pkl"
    artifact_root3 = artifact3.parent
    outside = tmp_path / "race-outside"
    outside.mkdir()
    outside_file = outside / "pred.pkl"
    outside_file.write_bytes(artifact3.read_bytes())
    outside_identity = (outside_file.stat().st_dev, outside_file.stat().st_ino)
    real_open = os.open
    real_read = os.read
    raced = {"done": False, "outside_read": False}
    def racing_open(path, flags, *args, **kwargs):
        if path == "artifacts" and kwargs.get("dir_fd") is not None and not raced["done"]:
            raced["done"] = True
            displaced = artifact_root3.with_name("artifacts-displaced")
            artifact_root3.rename(displaced)
            artifact_root3.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)
    def observing_read(fd, size):
        node = os.fstat(fd)
        if (node.st_dev, node.st_ino) == outside_identity:
            raced["outside_read"] = True
        return real_read(fd, size)
    monkeypatch.setattr(os, "open", racing_open)
    monkeypatch.setattr(os, "read", observing_read)
    raced_unit = inspect_rolling_evidence(context3, (request3,), backend3).unit_results[0]
    assert raced["done"] is True
    assert raced["outside_read"] is False
    assert raced_unit.classification in ("foreign", "corrupt", "drifted")


@pytest.mark.parametrize("sessions,detail", (
    (("2026-01-05",), "prediction_tail_missing"),
    (("2026-01-06",), "prediction_head_missing"),
))
def test_prediction_tail_head_and_internal_gap_are_coverage_short(tmp_path, sessions, detail):
    data = prediction_bytes(sessions)
    context, request, _, backend = make_valid_case(tmp_path / detail, data=data)
    unit = inspect_rolling_evidence(context, (request,), backend).unit_results[0]
    assert unit.classification == "coverage_short"
    assert detail in unit.blockers
    # A three-session request proves that a present head/tail still cannot hide an internal gap.
    if detail == "prediction_tail_missing":
        window = replace(request.window_identity, test_end="2026-01-07")
        run = replace(request.run_identity, window_keys=(window.window_key,))
        gap_data = prediction_bytes(("2026-01-05", "2026-01-07"))
        expectation = RollingArtifactExpectation("pred.pkl", "prediction", len(gap_data), hashlib.sha256(gap_data).hexdigest())
        gap_request = replace(request, run_identity=run, window_identity=window, artifacts=(expectation,), expected_prediction_sessions=("2026-01-05", "2026-01-06", "2026-01-07"))
        path = context.root / "mlruns" / "exp-1" / "recorder-1" / "artifacts" / "pred.pkl"
        path.write_bytes(gap_data)
        backend.candidates = (make_candidate(gap_request, path.parent),)
        gap = inspect_rolling_evidence(context, (gap_request,), backend).unit_results[0]
        assert gap.classification == "coverage_short"
        assert "prediction_internal_gap" in gap.blockers


def test_extra_or_out_of_window_sessions_are_identity_mismatch(tmp_path):
    data = prediction_bytes(("2026-01-05", "2026-01-06", "2026-01-07"))
    context, request, _, backend = make_valid_case(tmp_path, data=data)
    unit = inspect_rolling_evidence(context, (request,), backend).unit_results[0]
    assert unit.classification == "identity_mismatch"


def test_missing_calendar_or_unsupported_payload_is_not_comparable(tmp_path):
    request, _ = make_request(tmp_path.resolve())
    with pytest.raises(RollingEvidenceContractError):
        replace(request, expected_prediction_sessions=())
    frame = pd.DataFrame({"a": [1], "b": [2]})
    buffer = io.BytesIO()
    frame.to_pickle(buffer)
    context, request, _, backend = make_valid_case(tmp_path / "schema", data=buffer.getvalue())
    assert inspect_rolling_evidence(context, (request,), backend).unit_results[0].classification == "not_comparable"


def test_legacy_recorder_remains_unverified_even_when_artifacts_exist(tmp_path):
    context, request, _, backend = make_valid_case(tmp_path, protocol="legacy_unverified")
    unit = inspect_rolling_evidence(context, (request,), backend).unit_results[0]
    assert unit.classification == "legacy_unverified"
    assert "source_identity" not in unit.checked
    assert unit.capabilities == ("render",)


def test_orphan_candidate_is_reported_without_expanding_requested_results(tmp_path):
    context, request, candidate, backend = make_valid_case(tmp_path)
    orphan = dict(
        candidate, target_key="orphan@rolling",
        window_key="rolling:2026-01-05:2026-01-06:123456789abc",
        recorder_id="orphan", source_manifest_fingerprint="f" * 64,
    )
    backend.candidates = (candidate, orphan, dict(orphan, recorder_id="orphan-duplicate"))
    evidence = inspect_rolling_evidence(context, (request,), backend)
    assert evidence.requested_unit_keys == (request.unit_key,)
    assert len(evidence.unit_results) == 1
    assert tuple(item.unit_key for item in evidence.orphan_observations) == ((orphan["target_key"], orphan["window_key"]),)
    assert evidence.orphan_observations[0].candidate_count == 2
    assert evidence.n_candidates == 3


def test_inventory_or_public_path_drift_invalidates_the_observation(tmp_path, monkeypatch):
    context, request, _, backend = make_valid_case(tmp_path)
    backend.drift = True
    evidence = inspect_rolling_evidence(context, (request,), backend)
    assert evidence.status == "observation_drifted"
    assert evidence.unit_results[0].classification == "drifted"
    assert evidence.unit_results[0].capabilities == ("render",)
    context_identity, request_identity, _, backend_identity = make_valid_case(tmp_path / "identity-drift")
    identity_calls = {"count": 0}
    stable_identity = dict(backend_identity.identity)
    def changing_identity():
        identity_calls["count"] += 1
        value = dict(stable_identity)
        if identity_calls["count"] > 1:
            value["backend_fingerprint"] = "f" * 64
        return value
    backend_identity.tracking_identity = changing_identity
    identity_drift = inspect_rolling_evidence(context_identity, (request_identity,), backend_identity)
    assert identity_drift.status == "observation_drifted"
    assert identity_drift.unit_results[0].classification == "drifted"
    context_root, request_root, _, backend_root = make_valid_case(tmp_path / "root-drift")
    original_inventory = backend_root.inventory
    root_calls = {"count": 0}
    def root_swapping_inventory(requests):
        root_calls["count"] += 1
        if root_calls["count"] == 1:
            displaced = context_root.root.with_name(context_root.root.name + "-displaced")
            context_root.root.rename(displaced)
            shutil.copytree(displaced, context_root.root)
        return original_inventory(requests)
    backend_root.inventory = root_swapping_inventory
    root_drift = inspect_rolling_evidence(context_root, (request_root,), backend_root)
    assert root_drift.status == "observation_drifted"
    assert root_drift.unit_results[0].classification == "drifted"
    context2, request2, _, backend2 = make_valid_case(tmp_path / "path-drift")
    artifact = context2.root / "mlruns" / "exp-1" / "recorder-1" / "artifacts" / "pred.pkl"
    real_open = os.open
    calls = {"artifact": 0}
    def swapping_open(path, flags, *args, **kwargs):
        if path == "pred.pkl" and kwargs.get("dir_fd") is not None:
            calls["artifact"] += 1
            if calls["artifact"] == 2:
                displaced = artifact.with_suffix(".old")
                artifact.rename(displaced)
                artifact.write_bytes(displaced.read_bytes())
        return real_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", swapping_open)
    path_drift = inspect_rolling_evidence(context2, (request2,), backend2)
    assert path_drift.unit_results[0].classification == "drifted"
    assert path_drift.status == "incomplete"


def test_zero_one_many_requested_units_have_exact_terminal_cardinality(tmp_path):
    context, request, candidate, backend = make_valid_case(tmp_path)
    with pytest.raises(RollingEvidenceContractError):
        inspect_rolling_evidence(context, (), backend)
    one = inspect_rolling_evidence(context, (request,), backend)
    assert one.n_requested == 1
    assert one.n_candidates == 1
    second = replace(request, target_key="beta@rolling", run_identity=replace(request.run_identity, target_keys=(request.target_key, "beta@rolling")), recorder_id="recorder-2")
    first = replace(request, run_identity=second.run_identity)
    candidate1 = make_candidate(first, context.root / "mlruns" / "exp-1" / "recorder-1" / "artifacts")
    candidate1["experiment_id"] = True
    backend.candidates = (candidate1,)
    many = inspect_rolling_evidence(context, (first, second), backend)
    assert many.requested_unit_keys == (first.unit_key, second.unit_key)
    assert len(many.unit_results) == 2
    assert many.unit_results[0].classification == "identity_mismatch"
    assert many.unit_results[1].classification == "missing"
    assert many.n_candidates == 1


def test_aggregate_revalidates_foreign_duplicate_missing_and_reordered_members(tmp_path):
    request, _, _, evidence = inspect_case(tmp_path)
    result = evidence.unit_results[0]
    with pytest.raises(RollingEvidenceContractError):
        replace(evidence, unit_results=())
    with pytest.raises(RollingEvidenceContractError):
        replace(evidence, unit_results=(result, result), requested_unit_keys=(request.unit_key, request.unit_key))
    with pytest.raises(RollingEvidenceContractError):
        foreign = replace(result, unit_key=("foreign@rolling", request.window_key))
        replace(evidence, unit_results=(foreign,))
    run = replace(request.run_identity, target_keys=(request.target_key, "beta@rolling"))
    first = replace(request, run_identity=run)
    second = replace(first, target_key="beta@rolling", recorder_id="recorder-2")
    members = (missing_result(first), missing_result(second))
    token = fingerprint_value("aggregate-token")
    aggregate = RollingEvidenceSetInspection(
        (first.unit_key, second.unit_key), members, (), token, token,
        "none_valid", "rolling_evidence_set_none_valid",
    )
    with pytest.raises(RollingEvidenceContractError):
        replace(aggregate, unit_results=tuple(reversed(members)))


def test_counts_are_recomputed_from_terminal_members(tmp_path):
    _, _, _, evidence = inspect_case(tmp_path)
    assert evidence.n_requested == 1
    assert evidence.n_valid == 1
    assert evidence.n_blocked == 0
    assert evidence.n_candidates == 1
    assert evidence.to_public_dict()["n_candidates"] == 1
    assert not hasattr(evidence, "n_valid_input")
    with pytest.raises(TypeError):
        RollingEvidenceSetInspection(**dict(evidence.__dict__, n_valid=99))
    with pytest.raises(TypeError):
        RollingEvidenceSetInspection(**dict(evidence.__dict__, n_candidates=99))


def test_fake_backend_self_probe_detects_missing_duplicate_foreign_corrupt_and_drift(tmp_path):
    context, request, candidate, backend = make_valid_case(tmp_path)
    probes = []
    backend.candidates = ()
    probes.append(inspect_rolling_evidence(context, (request,), backend).unit_results[0].classification)
    backend.candidates = (candidate, candidate)
    probes.append(inspect_rolling_evidence(context, (request,), backend).unit_results[0].classification)
    backend.candidates = (dict(candidate, artifact_root_uri=(tmp_path / "outside").resolve().as_uri()),)
    probes.append(inspect_rolling_evidence(context, (request,), backend).unit_results[0].classification)
    backend.candidates = (candidate,)
    artifact = context.root / "mlruns" / "exp-1" / "recorder-1" / "artifacts" / "pred.pkl"
    artifact.write_bytes(b"corrupt")
    probes.append(inspect_rolling_evidence(context, (request,), backend).unit_results[0].classification)
    backend.drift = True
    probes.append(inspect_rolling_evidence(context, (request,), backend).unit_results[0].classification)
    assert probes == ["missing", "duplicate", "foreign", "corrupt", "drifted"]
    short_context, short_request, _, short_backend = make_valid_case(
        tmp_path / "coverage", data=prediction_bytes(("2026-01-05",)),
    )
    assert inspect_rolling_evidence(short_context, (short_request,), short_backend).unit_results[0].classification == "coverage_short"
    assert backend.mutation_calls == short_backend.mutation_calls == []


def test_inspection_is_zero_write_and_does_not_initialize_missing_backend(tmp_path, monkeypatch):
    context, request, _, backend = make_valid_case(tmp_path)
    controlled_temp = tmp_path / "controlled-temp"
    controlled_cache = tmp_path / "controlled-cache"
    controlled_temp.mkdir()
    controlled_cache.mkdir()
    monkeypatch.setenv("TMPDIR", str(controlled_temp))
    monkeypatch.setenv("TMP", str(controlled_temp))
    monkeypatch.setenv("TEMP", str(controlled_temp))
    monkeypatch.setenv("XDG_CACHE_HOME", str(controlled_cache))
    def snapshot_tree(root):
        result = {}
        for path in root.rglob("*"):
            key = path.relative_to(root).as_posix()
            if path.is_symlink():
                result[key] = ("symlink", os.readlink(path))
            elif path.is_file():
                result[key] = ("file", path.read_bytes())
            else:
                result[key] = ("directory", None)
        return result
    write_attempts = []
    real_os_open = os.open
    def observing_os_open(path, flags, *args, **kwargs):
        write_mask = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if flags & write_mask:
            write_attempts.append(("os.open", os.fspath(path), flags))
        return real_os_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", observing_os_open)
    real_builtin_open = builtins.open
    real_io_open = io.open
    def observing_stream_open(original, source, mode="r", *args, **kwargs):
        if any(marker in mode for marker in ("w", "a", "x", "+")):
            write_attempts.append(("stream.open", os.fspath(source), mode))
        return original(source, mode, *args, **kwargs)
    monkeypatch.setattr(
        builtins, "open",
        lambda source, mode="r", *args, **kwargs: observing_stream_open(
            real_builtin_open, source, mode, *args, **kwargs
        ),
    )
    monkeypatch.setattr(
        io, "open",
        lambda source, mode="r", *args, **kwargs: observing_stream_open(
            real_io_open, source, mode, *args, **kwargs
        ),
    )
    for name in ("mkdir", "makedirs", "rename", "replace", "unlink", "remove", "rmdir"):
        original = getattr(os, name)
        def observing_mutation(*args, _name=name, _original=original, **kwargs):
            write_attempts.append((_name, repr(args), None))
            return _original(*args, **kwargs)
        monkeypatch.setattr(os, name, observing_mutation)
    before = snapshot_tree(tmp_path)
    before_cwd = os.getcwd()
    before_env = dict(os.environ)
    inspect_rolling_evidence(context, (request,), backend)
    after = snapshot_tree(tmp_path)
    assert before == after
    assert os.getcwd() == before_cwd
    assert dict(os.environ) == before_env
    assert write_attempts == []
    assert backend.mutation_calls == []
    missing = FakeEvidenceBackend({})
    result = inspect_rolling_evidence(context, (request,), missing)
    assert result.unit_results[0].classification == "not_comparable"
    assert "tracking_identity" not in result.unit_results[0].checked
    assert missing.mutation_calls == []
    class FailingIdentityBackend(FakeEvidenceBackend):
        def tracking_identity(self):
            raise RuntimeError("identity unavailable")
    failed_identity = FailingIdentityBackend(backend.identity)
    failed_result = inspect_rolling_evidence(context, (request,), failed_identity)
    assert "tracking_identity" not in failed_result.unit_results[0].checked
    class ForbiddenMutationBackend(FakeEvidenceBackend):
        def inventory(self, requests):
            self.write("forbidden-probe")
    mutation_probe = ForbiddenMutationBackend(backend.identity)
    inspect_rolling_evidence(context, (request,), mutation_probe)
    assert mutation_probe.mutation_calls == [
        ("write", ("forbidden-probe",)),
        ("write", ("forbidden-probe",)),
    ]
    class InterruptedBackend(FakeEvidenceBackend):
        def inventory(self, requests):
            raise KeyboardInterrupt()
    interrupted = InterruptedBackend(backend.identity)
    with pytest.raises(KeyboardInterrupt):
        inspect_rolling_evidence(context, (request,), interrupted)
    probe = tmp_path / "observer-self-probe"
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(fd)
    assert any(item[0] == "os.open" for item in write_attempts)


def test_import_does_not_load_legacy_env_qlib_or_mlflow_or_change_cwd():
    import subprocess
    import sys
    script = (
        "import os,sys; before=os.getcwd(); import quantpits.rolling.evidence; "
        "assert os.getcwd()==before; "
        "assert 'qlib' not in sys.modules; assert 'mlflow' not in sys.modules; "
        "assert 'quantpits.utils.env' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_state_receipt_current_records_and_registry_cannot_substitute_source_evidence(tmp_path):
    context, request, _, backend = make_valid_case(tmp_path)
    for substitute in ({"record_id": "recorder-1"}, object(), request.run_identity):
        with pytest.raises(RollingEvidenceContractError):
            inspect_rolling_evidence(context, (substitute,), backend)
