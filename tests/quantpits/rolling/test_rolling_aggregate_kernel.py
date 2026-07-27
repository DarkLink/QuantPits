from quantpits.rolling import (
    materialize_rolling_aggregate_candidates,
    rolling_aggregate_result_json,
)

from tests.quantpits.rolling.aggregate_support import (
    FakeCandidateBackend,
    aggregate_case,
)


def test_candidate_partition_values_equal_exact_sources(tmp_path):
    _context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend()
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "success"
    assert result.target_results[0].candidate.content_fingerprint


def test_candidate_inventory_partition_is_disjoint_and_count_conserving(tmp_path):
    _context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend()
    backend.candidates["historical"] = {
        "candidate_key": "c" * 64,
        "target_key": aggregate.target_keys[0],
        "scope_fingerprint": "d" * 64,
        "aggregate_attempt_id": "historical-attempt",
    }
    backend.candidates["malformed"] = {"candidate_key": None}
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    inventory = backend.inventory(aggregate)
    assert inventory["raw_count"] == len(backend.candidates) == 3
    assert result.to_public_dict()["n_materialized"] == 1
    assert result.to_public_dict()["candidate_inventory"] == {
        "raw_inventory_count": 3,
        "n_requested_owned": 1,
        "n_orphan_owned": 1,
        "n_unassigned": 1,
    }


def test_candidate_reuse_requires_one_exact_reobserved_candidate(tmp_path):
    _context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend()
    first = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    second = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert first.target_results[0].status == "materialized_success"
    assert second.target_results[0].status == "reused_success"
    assert len(backend.calls) == 1


def test_candidate_manifest_and_artifact_are_independently_reobserved(tmp_path):
    _context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend()
    first = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    candidate = backend.candidates[aggregate.candidate_keys[0]]
    candidate["content_fingerprint"] = "0" * 64
    second = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert first.status == "success"
    assert second.status == "blocked"


def test_candidate_materialization_never_changes_publication_or_state(tmp_path):
    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    protected = {
        path: path.read_bytes() if path.exists() else None
        for path in (
            repository.state_path,
            context.data_dir / "latest_train_records.json",
        )
    }
    materialize_rolling_aggregate_candidates(
        aggregate, repository, source, FakeCandidateBackend(),
    )
    assert protected == {
        path: path.read_bytes() if path.exists() else None
        for path in protected
    }


def test_ordinary_target_failure_preserves_later_identity_and_execution(tmp_path):
    _context, _scope, repository, source, aggregate = aggregate_case(
        tmp_path, n_targets=2,
    )
    backend = FakeCandidateBackend()
    backend.controls[0] = RuntimeError("injected")
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert tuple(item.target_key for item in result.target_results) == aggregate.target_keys
    assert result.target_results[0].status == "failed"
    assert result.target_results[1].status == "materialized_success"


def test_aggregate_result_renderings_share_one_invocation_truth(tmp_path):
    import json
    from quantpits.rolling import render_rolling_aggregate_result

    _context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, FakeCandidateBackend(),
    )
    payload = json.loads(rolling_aggregate_result_json(result))
    assert payload["fingerprint"] == result.fingerprint
    assert result.fingerprint in render_rolling_aggregate_result(result)


def test_phase35_domain_api_is_not_wired_to_legacy_rolling_cli():
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    text = (root / "quantpits/scripts/rolling_train.py").read_text()
    assert "rolling.aggregate" not in text
    assert "materialize_rolling_aggregate_candidates" not in text
