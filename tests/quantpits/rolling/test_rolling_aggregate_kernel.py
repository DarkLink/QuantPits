from quantpits.rolling import (
    materialize_rolling_aggregate_candidates,
    rolling_aggregate_result_json,
)

from tests.quantpits.rolling.aggregate_support import (
    FakeCandidateBackend,
    aggregate_case,
)


def test_candidate_partition_values_equal_exact_sources(tmp_path):
    import io
    import pandas as pd

    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend(context)
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "success"
    assert result.target_results[0].candidate.content_fingerprint
    candidate = backend.candidates[aggregate.candidate_keys[0]]
    requests = source.requests_for_state(
        aggregate.execution_scope,
        aggregate.state_repository_view.inspection.snapshot,
    )
    expected = pd.concat([
        pd.read_pickle(io.BytesIO(source.prediction_bytes(request)))
        for request in requests
    ])
    actual = pd.read_pickle(io.BytesIO(candidate["prediction_bytes"]))
    pd.testing.assert_frame_equal(actual, expected)
    manifest = candidate["manifest"]
    assert manifest["source_row_counts"] == [2, 2]
    assert manifest["row_count"] == len(actual) == 4
    assert manifest["expected_sessions"] == [
        session
        for sessions in manifest["source_sessions"]
        for session in sessions
    ]
    assert manifest["checked_predicates"][-3:] == [
        "candidate_index_exactness",
        "candidate_value_exactness",
        "candidate_content_exactness",
    ]


def test_candidate_inventory_partition_is_disjoint_and_count_conserving(tmp_path):
    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend(context)
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
    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend(context)
    first = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    second = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert first.target_results[0].status == "materialized_success"
    assert second.target_results[0].status == "reused_success"
    assert len(backend.calls) == 1


def test_foreign_candidate_backend_workspace_is_blocked_before_write(
    tmp_path,
):
    context, _scope, repository, source, aggregate = aggregate_case(
        tmp_path / "source",
    )
    foreign_context, *_unused = aggregate_case(tmp_path / "candidate")
    backend = FakeCandidateBackend(foreign_context)
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "blocked"
    assert backend.calls == []
    assert backend.candidates == {}
    assert "publication_input" not in result.capabilities
    assert context.root != foreign_context.root


def test_stale_or_noncanonical_state_blocks_before_candidate_write(
    tmp_path,
):
    context, _scope, repository, source, aggregate = aggregate_case(
        tmp_path,
    )
    backend = FakeCandidateBackend(context)
    repository.state_path.write_text("{}", encoding="utf-8")
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "blocked"
    assert backend.calls == []
    assert "publication_input" not in result.capabilities


def test_terminal_duplicate_denies_publication_input(tmp_path):
    context, _scope, repository, source, aggregate = aggregate_case(
        tmp_path,
    )
    backend = FakeCandidateBackend(context)

    def inject_duplicate(
        _scope, callback, create_if_missing=False,
    ):
        candidate = backend.candidates[aggregate.candidate_keys[0]]
        backend.candidates["terminal-duplicate"] = dict(candidate)
        return callback()

    backend.with_candidate_lock = inject_duplicate
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert result.status == "indeterminate"
    assert result.inventory_counts == ()
    assert result.target_results[0].candidate is None
    assert "publication_input" not in result.capabilities


def test_candidate_is_rechecked_after_final_source_state_recheck(
    tmp_path,
):
    context, _scope, repository, source, aggregate = aggregate_case(
        tmp_path,
    )
    backend = FakeCandidateBackend(context)
    original_inventory = backend.inventory
    calls = []

    def race_on_final_inventory(scope):
        calls.append(len(calls) + 1)
        if len(calls) == 4:
            candidate = backend.candidates[
                aggregate.candidate_keys[0]
            ]
            backend.candidates["late-terminal-duplicate"] = dict(
                candidate,
            )
        return original_inventory(scope)

    backend.inventory = race_on_final_inventory
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert len(calls) == 4
    assert result.status == "indeterminate"
    assert "publication_input" not in result.capabilities


def test_candidate_manifest_and_artifact_are_independently_reobserved(tmp_path):
    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend(context)
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


def test_candidate_reuse_rejects_self_consistent_source_provenance_rewrite(
    tmp_path,
):
    from quantpits.rolling.aggregate import (
        _candidate_manifest_contract_fingerprint,
    )

    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    backend = FakeCandidateBackend(context)
    first = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    candidate = backend.candidates[aggregate.candidate_keys[0]]
    candidate["manifest"]["source_recorder_ids"][0] = "foreign-recorder"
    candidate["manifest_contract_fingerprint"] = (
        _candidate_manifest_contract_fingerprint(candidate["manifest"])
    )
    second = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, backend,
    )
    assert first.status == "success"
    assert second.status == "blocked"
    assert second.target_results[0].candidate is None
    assert "publication_input" not in second.capabilities


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
        aggregate, repository, source, FakeCandidateBackend(context),
    )
    assert protected == {
        path: path.read_bytes() if path.exists() else None
        for path in protected
    }


def test_ordinary_target_failure_preserves_later_identity_and_execution(tmp_path):
    context, _scope, repository, source, aggregate = aggregate_case(
        tmp_path, n_targets=2,
    )
    backend = FakeCandidateBackend(context)
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

    context, _scope, repository, source, aggregate = aggregate_case(tmp_path)
    result = materialize_rolling_aggregate_candidates(
        aggregate, repository, source, FakeCandidateBackend(context),
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
