import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

import quantpits.research.decision_surface as module
from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes
from quantpits.research.decision_surface import (
    COMPONENT_NAMES,
    CURATED_CODE_PATHS,
    DecisionSurfaceContractError,
    DecisionSurfaceInputError,
    DecisionSurfaceComponent,
    ProductionDecisionSurfaceResult,
    observe_production_decision_surface,
)


def _components(change=None, incomparable=None):
    rows = []
    for index, name in enumerate(COMPONENT_NAMES):
        reference = {"position": index, "value": "stable"}
        current = copy.deepcopy(reference)
        if name == change:
            current["value"] = "changed"
        if name == incomparable:
            current = None
        rows.append(module._component(name, reference, current, "SYNTHETIC_INCOMPARABLE"))
    return tuple(rows)


def _result(change=None, incomparable=None):
    return module._make_result(
        _components(change, incomparable), "2026-08-14", "2026-08-21",
    )


def test_exact_six_equal_rows_are_the_only_same_segment_capability():
    result = _result()
    summary = result.to_safe_summary_dict()
    assert result.status == "SAME_CHAMPION_SEGMENT"
    assert summary["component_count"] == 6
    assert [row["name"] for row in summary["components"]] == list(COMPONENT_NAMES)
    assert summary["same_champion_segment"] is True
    assert summary["may_continue_to_retention_gates"] is True
    assert summary["decision_surface_id"].startswith("surface.")
    assert all(summary[field] is False for field in (
        "intent_capability", "epoch_started", "prospective_claim",
        "promotion_capability", "did_write",
    ))


@pytest.mark.parametrize("name", COMPONENT_NAMES)
def test_each_comparable_component_change_is_version_break(name):
    summary = _result(change=name).to_safe_summary_dict()
    assert summary["status"] == "VERSION_BREAK"
    changed = [row for row in summary["components"] if row["name"] == name][0]
    assert changed["comparison"] == "DIFFERENT"
    assert changed["reference_digest"] != changed["current_digest"]
    assert summary["decision_surface_id"] is None
    assert summary["same_champion_segment"] is False


@pytest.mark.parametrize("name", COMPONENT_NAMES)
def test_each_incomparable_component_overrides_version_break_and_denies_capability(name):
    other = COMPONENT_NAMES[0] if name != COMPONENT_NAMES[0] else COMPONENT_NAMES[1]
    rows = list(_components(change=other, incomparable=name))
    result = module._make_result(rows, "2026-08-14", "2026-08-21")
    summary = result.to_safe_summary_dict()
    assert summary["status"] == "INCOMPARABLE"
    blocked = [row for row in summary["components"] if row["name"] == name][0]
    assert blocked["current_digest"] is None
    assert summary["decision_surface_id"] is None
    assert summary["may_continue_to_retention_gates"] is False


def test_result_public_replay_replace_and_count_forgery_cannot_grant_authority():
    result = _result()
    payload = result.to_safe_summary_dict()
    with pytest.raises(DecisionSurfaceContractError, match="observer-owned"):
        ProductionDecisionSurfaceResult(**payload)
    with pytest.raises(DecisionSurfaceContractError, match="observer-owned"):
        DecisionSurfaceComponent(**payload["components"][0])
    result._components = result._components[:-1]
    with pytest.raises(DecisionSurfaceContractError):
        result.to_safe_summary_dict()


def test_component_failure_preserves_exact_six_rows_and_later_identity():
    rows = []
    for index, name in enumerate(COMPONENT_NAMES):
        if index == 2:
            def factory():
                raise OSError("private raw error")
        else:
            factory = lambda value=index: {"position": value}
        rows.append(module._observe_component(name, {"position": index}, factory))
    result = module._make_result(rows, "2026-08-14", "2026-08-21")
    assert result.status == "INCOMPARABLE"
    assert len(result.components) == 6
    assert result.components[2].reason_code == "ENSEMBLE_SEMANTICS_OBSERVATION_FAILED"
    assert result.components[3].comparison == "EQUAL"


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_process_control_propagates_from_component_observation(exc):
    def factory():
        raise exc
    with pytest.raises(type(exc)):
        module._observe_component(COMPONENT_NAMES[0], {}, factory)


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=str(repo), check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.decode().strip()


def _synthetic_engine(tmp_path):
    repo = tmp_path / "engine"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "synthetic@example.invalid")
    _git(repo, "config", "user.name", "Synthetic")
    for index, logical in enumerate(CURATED_CODE_PATHS):
        path = repo / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("member-%d\n" % index)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("position", range(len(CURATED_CODE_PATHS)))
def test_each_curated_code_member_change_is_observed_as_version_break(tmp_path, position):
    repo, first = _synthetic_engine(tmp_path)
    target = repo / CURATED_CODE_PATHS[position]
    target.write_text(target.read_text() + "changed\n")
    _git(repo, "add", CURATED_CODE_PATHS[position])
    _git(repo, "commit", "-m", "change one")
    second = _git(repo, "rev-parse", "HEAD")
    reference = module._git_blob_projection(repo, first)
    current = module._git_blob_projection(repo, second)
    row = module._component(COMPONENT_NAMES[0], reference, current, "")
    assert row.comparison == "DIFFERENT"
    changed = [
        item["path"] for item, other in zip(reference["members"], current["members"])
        if item != other
    ]
    assert changed == [CURATED_CODE_PATHS[position]]


def test_unrelated_git_change_does_not_change_curated_code_surface(tmp_path):
    repo, first = _synthetic_engine(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "note.md").write_text("unrelated\n")
    _git(repo, "add", "docs/note.md")
    _git(repo, "commit", "-m", "unrelated")
    second = _git(repo, "rev-parse", "HEAD")
    assert module._git_blob_projection(repo, first) == module._git_blob_projection(repo, second)


def _lineage(*, child_suffix="A", universe_digest="1" * 64):
    members = ["MODEL_%d" % index for index in range(4)]
    models = []
    artifacts = []
    for index, key in enumerate(members):
        source = "SOURCE_%d" % index
        models.append({
            "resolved_key": key, "recorder_id": "PRED_%d_%s" % (index, child_suffix),
            "source_recorder_id": source, "family": "TREE" if index == 0 else "SEQUENCE",
            "mode": "STATIC", "training_cutoff": "2026-08-01",
            "operation": "predict_only",
            "status": "ready",
        })
        member_digest = TypedDigest.raw(("artifact-%d" % index).encode()).to_dict()
        inventory = [{"path": "model-%d.bin" % index, "digest": member_digest}]
        artifacts.append({
            "position": index, "role": "source_training", "recorder_id": source,
            "experiment_name": "TRAINING", "artifact_locator": "source/%d" % index,
            "members": [{
                "path": "model-%d.bin" % index, "status": "observed",
                "digest": member_digest, "preservation_status": "embedded", "detail": "",
            }],
            "artifact_tree_digest": TypedDigest.canonical(inventory, "file_inventory").to_dict(),
        })
        prediction_digest = TypedDigest.raw(("prediction-%d" % index).encode()).to_dict()
        prediction_inventory = [{
            "path": "prediction-%d.pkl" % index, "digest": prediction_digest,
        }]
        artifacts.append({
            "position": index, "recorder_id": "PRED_%d_%s" % (index, child_suffix),
            "source_recorder_id": source, "artifact_locator": "pred/%d" % index,
            "members": [{
                "path": "prediction-%d.pkl" % index, "status": "observed",
                "digest": prediction_digest, "preservation_status": "embedded", "detail": "",
            }],
            "artifact_tree_digest": TypedDigest.canonical(
                prediction_inventory, "file_inventory",
            ).to_dict(),
        })
    ensemble_digest = TypedDigest.raw(b"ensemble").to_dict()
    ensemble_inventory = [{"path": "pred.pkl", "digest": ensemble_digest}]
    artifacts.append({
        "position": "ensemble", "recorder_id": "ENSEMBLE",
        "artifact_locator": "ensemble", "members": [{
            "path": "pred.pkl", "status": "observed", "digest": ensemble_digest,
            "preservation_status": "embedded", "detail": "",
        }],
        "artifact_tree_digest": TypedDigest.canonical(
            ensemble_inventory, "file_inventory",
        ).to_dict(),
    })
    return {
        "model_and_ensemble_lineage": {
            "combo": {
                "method": "equal", "resolved_members": members,
                "ensemble_recorder_id": "ENSEMBLE",
            },
            "source_models": models, "source_artifacts": artifacts,
        },
        "data_identity": {"qlib_materialization_identity": {
            "status": "observed", "universe_name": "csi300",
            "universe_digest": {"value": universe_digest},
        }},
    }


def test_dynamic_child_and_weekly_universe_digest_do_not_break_stable_components():
    first = _lineage(child_suffix="A", universe_digest="1" * 64)
    second = _lineage(child_suffix="B", universe_digest="2" * 64)
    assert module._source_projection(first) == module._source_projection(second)
    assert module._prediction_projection(first) == module._prediction_projection(second)
    assert module._market_projection(first) == module._market_projection(second)


def test_market_policy_change_is_version_break_and_dirty_curated_path_is_incomparable():
    first = _lineage()
    second = _lineage()
    second["data_identity"]["qlib_materialization_identity"]["universe_name"] = "csi500"
    row = module._component(
        COMPONENT_NAMES[4], module._market_projection(first),
        module._market_projection(second), "",
    )
    assert row.comparison == "DIFFERENT"
    with pytest.raises(module._ComponentIncomparable, match="DIRTY"):
        module._engine_commit({
            "engine_identity": {
                "commit": "a" * 40, "status_inventory": [CURATED_CODE_PATHS[0]],
            },
        })


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign_parent"])
def test_invalid_source_partition_or_prediction_parent_is_incomparable(mutation):
    manifest = _lineage()
    lineage = manifest["model_and_ensemble_lineage"]
    if mutation == "missing":
        lineage["source_artifacts"] = [
            row for row in lineage["source_artifacts"]
            if not (row.get("role") == "source_training" and row.get("position") == 3)
        ]
    elif mutation == "duplicate":
        training = [row for row in lineage["source_artifacts"] if row.get("role") == "source_training"]
        training[1]["position"] = 0
    else:
        lineage["source_models"][2]["source_recorder_id"] = "FOREIGN"
    with pytest.raises(module._ComponentIncomparable):
        module._source_projection(manifest)


def _embedded_object(cycle, value, *, raw=False):
    data = value if raw else canonical_json_bytes(value)
    digest = TypedDigest.raw(data).to_dict()
    path = cycle / "objects" / digest["value"][:2] / digest["value"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_bytes(data)
    path.chmod(0o600)
    return {
        "path": "synthetic", "status": "observed", "digest": digest,
        "preservation_status": "embedded", "detail": "",
    }


def _intent_fixture(tmp_path, *, topk=22, requested=None):
    from quantpits.config_contracts.normalizers import normalize_strategy_config
    from quantpits.utils.workspace import fingerprint_value
    import yaml

    cycle = tmp_path / "cycle"
    cycle.mkdir(mode=0o700, parents=True)
    (cycle / "objects").mkdir(mode=0o700)
    config = {
        "strategy": {"name": "topk_dropout", "params": {
            "topk": topk, "n_drop": 4, "buy_suggestion_factor": 2,
            "sell_out_of_universe": True,
        }},
        "backtest": {},
    }
    config_bytes = yaml.safe_dump(config, sort_keys=True).encode()
    config_observation = _embedded_object(cycle, config_bytes, raw=True)
    records_bytes = json.dumps({
        "default_combo": "CHAMPION_4",
        "combos": {"CHAMPION_4": "DYNAMIC"},
    }, indent=2).encode()
    records_observation = _embedded_object(cycle, records_bytes, raw=True)
    raw_manifest = {
        "schema_version": 1, "command": "order_gen", "status": "success",
        "run_id": "M4", "args": [],
        "config_fingerprints": {
            "strategy_config": fingerprint_value(normalize_strategy_config(config)),
        },
        "records": {"source": {
            "mode": "ensemble", "requested_name": requested,
            "resolved_name": "CHAMPION_4", "record_id": "DYNAMIC",
            "experiment_name": "Ensemble_Fusion",
        }},
    }
    run_observation = _embedded_object(cycle, raw_manifest)
    manifest = {
        "run_evidence": [{"class": "order", **run_observation}],
        "referenced_evidence": [{
            "evidence_class": "order", "collection": "inputs", "kind": "config",
            "locator": "config/strategy_config.yaml", "locator_type": "workspace_file",
            "observation": config_observation,
        }, {
            "evidence_class": "order", "collection": "inputs", "kind": "record",
            "locator": "config/ensemble_records.json", "locator_type": "workspace_file",
            "observation": records_observation,
        }],
    }
    return cycle, manifest


def _ensemble_fixture(tmp_path, *, method="equal", normalization="rank", reverse=False):
    cycle = tmp_path / "cycle"
    cycle.mkdir(mode=0o700, parents=True)
    (cycle / "objects").mkdir(mode=0o700)
    members = ["MODEL_%d" % index for index in range(4)]
    if reverse:
        members.reverse()
    args = ["--from-config"]
    if normalization != "rank":
        args.extend(["--norm-method", normalization])
    raw = {
        "schema_version": 1, "command": "ensemble_fusion", "status": "success",
        "run_id": "M3", "args": args,
        "records": {"combos": [{
            "name": "CHAMPION_4", "method": method, "is_default": True,
            "models": members, "resolved_models": members,
        }]},
    }
    observation = _embedded_object(cycle, raw)
    manifest = {
        "run_evidence": [{"class": "ensemble", **observation}],
        "model_and_ensemble_lineage": {
            "combo": {"method": method, "resolved_members": members},
        },
    }
    return cycle, manifest


def test_combo_member_order_method_normalization_and_default_are_strict(tmp_path):
    cycle, manifest = _ensemble_fixture(tmp_path / "same")
    reference = module._ensemble_projection(cycle, manifest)
    changed_cycle, changed_manifest = _ensemble_fixture(tmp_path / "order", reverse=True)
    changed = module._ensemble_projection(changed_cycle, changed_manifest)
    assert module._component(COMPONENT_NAMES[2], reference, changed, "").comparison == "DIFFERENT"
    bad_cycle, bad_manifest = _ensemble_fixture(tmp_path / "method", method="weighted")
    with pytest.raises(module._ComponentIncomparable):
        module._ensemble_projection(bad_cycle, bad_manifest)
    bad_cycle, bad_manifest = _ensemble_fixture(tmp_path / "normalization", normalization="zscore")
    with pytest.raises(module._ComponentIncomparable):
        module._ensemble_projection(bad_cycle, bad_manifest)
    ambiguous_cycle, ambiguous_manifest = _ensemble_fixture(tmp_path / "ambiguous")
    raw = module._run_manifest(ambiguous_cycle, ambiguous_manifest, "ensemble")
    raw["records"]["combos"].append(copy.deepcopy(raw["records"]["combos"][0]))
    raw["records"]["combos"][1]["name"] = "OTHER"
    observation = _embedded_object(ambiguous_cycle, raw)
    ambiguous_manifest["run_evidence"] = [{"class": "ensemble", **observation}]
    with pytest.raises(module._ComponentIncomparable, match="AMBIGUOUS"):
        module._ensemble_projection(ambiguous_cycle, ambiguous_manifest)


def test_sealed_run_manifest_accepts_strict_noncanonical_bytes_and_rejects_duplicates(
    tmp_path,
):
    cycle, manifest = _ensemble_fixture(tmp_path)
    raw = module._run_manifest(cycle, manifest, "ensemble")
    noncanonical = json.dumps(raw, indent=2).encode()
    observation = _embedded_object(cycle, noncanonical, raw=True)
    manifest["run_evidence"] = [{"class": "ensemble", **observation}]
    assert module._ensemble_projection(cycle, manifest)["method"] == "equal"

    duplicate = noncanonical.replace(
        b'"schema_version": 1,',
        b'"schema_version": 1, "schema_version": 1,',
        1,
    )
    observation = _embedded_object(cycle, duplicate, raw=True)
    manifest["run_evidence"] = [{"class": "ensemble", **observation}]
    with pytest.raises(module._ComponentIncomparable, match="JSON_INVALID"):
        module._ensemble_projection(cycle, manifest)


def test_intent_projection_uses_sealed_config_and_actual_ensemble_source(tmp_path):
    cycle, manifest = _intent_fixture(tmp_path)
    projection = module._intent_projection(cycle, manifest)
    assert projection["topk"] == 22
    assert projection["resolved_combo"] == "CHAMPION_4"
    assert projection["requested_combo_relation"] == "DEFAULT"
    assert projection["production_primitive_id"] == "TOPK_DROPOUT_ORDER_GENERATOR_CURRENT_RULE_V1"


def test_strategy_projection_change_is_comparable_and_fallback_is_incomparable(tmp_path):
    first_cycle, first_manifest = _intent_fixture(tmp_path / "first")
    second_cycle, second_manifest = _intent_fixture(tmp_path / "second", topk=23)
    first = module._intent_projection(first_cycle, first_manifest)
    second = module._intent_projection(second_cycle, second_manifest)
    assert module._component(COMPONENT_NAMES[5], first, second, "").comparison == "DIFFERENT"
    fallback_cycle, fallback_manifest = _intent_fixture(
        tmp_path / "fallback", requested="MISSING_REQUEST",
    )
    with pytest.raises(module._ComponentIncomparable, match="FALLBACK"):
        module._intent_projection(fallback_cycle, fallback_manifest)


def test_strict_schema_rejects_bool_as_integer_digest_size():
    value = TypedDigest.canonical({"x": 1}).to_dict()
    value["size_bytes"] = True
    with pytest.raises(module._ComponentIncomparable):
        module._typed_digest(value, "test", "canonical_json")


def _sealed_cycle(tmp_path, *, status="sealed_complete", problems=()):
    root = tmp_path / "workspace"
    cycle_id = "2026-08-21"
    cycle = root / "data" / "evidence" / "v1" / "cycles" / cycle_id
    cycle.mkdir(parents=True, mode=0o700)
    cycle.chmod(0o700)
    core = {
        "schema_version": 1,
        "cycle_identity": {"cycle_id": cycle_id},
        "engine_identity": {}, "workspace_identity": {}, "data_identity": {},
        "run_evidence": [], "model_and_ensemble_lineage": {}, "ranking": {},
        "portfolio_state": {}, "decision_state": {}, "referenced_evidence": [],
        "preservation": {"embedded_object_count": 0, "named_file_count": 0},
        "problems": [dict(row) for row in problems],
    }
    replay_core = {key: value for key, value in core.items() if key != "workspace_identity"}
    core["request_content_digest"] = TypedDigest.canonical(replay_core).to_dict()
    core["capture_time"] = "2026-08-21T00:00:00+00:00"
    core["status"] = status
    manifest_data = canonical_json_bytes(core)
    seal = {
        "schema_version": 1, "cycle_id": cycle_id, "status": status,
        "manifest_digest": TypedDigest.raw(manifest_data).to_dict(),
        "artifact_root_digest": TypedDigest.canonical({
            "objects": [], "named_files": {},
        }).to_dict(),
        "object_digests": [], "named_file_digests": {},
    }
    for name, data in (
        ("manifest.json", manifest_data), ("seal.json", canonical_json_bytes(seal)),
    ):
        (cycle / name).write_bytes(data)
        (cycle / name).chmod(0o600)
    return root, cycle_id, cycle


def test_cycle_admission_allows_only_scoped_deep_analysis_partial(tmp_path):
    allowed = {
        "code": "deep_analysis_missing", "evidence_class": "deep_analysis",
        "detail": "scoped", "blocks_complete": True,
    }
    root, cycle_id, _cycle = _sealed_cycle(
        tmp_path / "allowed", status="sealed_partial", problems=(allowed,),
    )
    assert module._cycle_authority(root, cycle_id)[1]["status"] == "sealed_partial"
    blocked = dict(allowed)
    blocked.update({"code": "ranking_unavailable", "evidence_class": "ranking"})
    root, cycle_id, _cycle = _sealed_cycle(
        tmp_path / "blocked", status="sealed_partial", problems=(blocked,),
    )
    with pytest.raises(module._ComponentIncomparable, match="NOT_SCOPED"):
        module._cycle_authority(root, cycle_id)


@pytest.mark.parametrize("mutation", ["extra", "bool_schema", "request_digest"])
def test_cycle_admission_rejects_extra_inventory_and_resealed_semantic_drift(
    tmp_path, mutation,
):
    root, cycle_id, cycle = _sealed_cycle(tmp_path)
    if mutation == "extra":
        (cycle / "foreign").write_bytes(b"")
        (cycle / "foreign").chmod(0o600)
    else:
        manifest = json.loads((cycle / "manifest.json").read_bytes())
        if mutation == "bool_schema":
            manifest["schema_version"] = True
        else:
            manifest["ranking"] = {"status": "changed"}
        data = canonical_json_bytes(manifest)
        (cycle / "manifest.json").write_bytes(data)
        seal = json.loads((cycle / "seal.json").read_bytes())
        seal["manifest_digest"] = TypedDigest.raw(data).to_dict()
        (cycle / "seal.json").write_bytes(canonical_json_bytes(seal))
    with pytest.raises(module._ComponentIncomparable):
        module._cycle_authority(root, cycle_id)


def test_observer_request_rejects_nonphysical_paths_before_any_authority_call(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(DecisionSurfaceInputError):
        observe_production_decision_surface(
            alias, root, root, "2026-08-21", root / "missing",
            root, root, root, "bootstrap.synthetic",
        )


class _SyntheticGuard:
    supported = True

    def __init__(self, *_args, **_kwargs):
        self.closed = False

    def mutated(self):
        return False

    def close(self):
        self.closed = True


class _SyntheticReceipt:
    request_digest = {}
    manifest_digest = {}


class _SyntheticEvidence:
    status = "ADOPTED"
    definition_evidence_complete = True
    evidence_cycle_id = "2026-08-14"
    evidence_receipt = _SyntheticReceipt()


class _SyntheticLeaf:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return copy.deepcopy(self.value)


class _SyntheticDefinition:
    definition_set_id = "shadow.synthetic"

    def __init__(self):
        self.compiled_definitions = type("Compiled", (), {
            "champion": _SyntheticLeaf({
                "fusion_definition": "PER_MODEL_CROSS_SECTIONAL_PERCENTILE_RANK_EQUAL_MEAN_V1",
                "source_members": [],
            }),
            "protocol": _SyntheticLeaf({"intent_definition": {
                "schema_version": 1, "strategy_name": "topk_dropout",
                "topk": 22, "n_drop": 4, "buy_suggestion_factor": 2,
                "sell_out_of_universe": True, "production_buy_lot_size": 100,
                "cashflow_mode": "ZERO_EXTERNAL_CASHFLOW",
                "planning_price_rule": "CASH_CLOSE_WITH_0_9_1_1_ESTIMATES_V1",
                "buy_selection_rule": "FIRST_N_GENERATED_SUGGESTIONS_V1",
                "production_primitive_id": "TOPK_DROPOUT_ORDER_GENERATOR_CURRENT_RULE_V1",
            }}),
        })()

    def to_store_request(self):
        return type("Request", (), {"request_digest": {}})()


def _observer_layout(tmp_path):
    research = tmp_path / "research-workspace"
    production = tmp_path / "production-workspace"
    engine = tmp_path / "engine"
    for root in (research, production, engine):
        root.mkdir()
    (engine / ".git").mkdir()
    shadow = research / "research" / "shadow_v1"
    activation_dir = shadow / "activations"
    definitions = shadow / "definitions"
    evidence = shadow / "definition_evidence"
    bootstraps = shadow / "bootstraps"
    for path in (activation_dir, definitions, evidence, bootstraps):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)
    activation = activation_dir / "shadow.synthetic.json"
    activation.write_bytes(canonical_json_bytes({
        "evidence_cycle_id": "2026-08-14", "effective_cycle": "2026-08-15",
    }))
    activation.chmod(0o600)
    return research, production, engine, activation, definitions, evidence, bootstraps


@pytest.mark.parametrize(
    "outcome", ["same", "version", "incomparable", "reference_intent_incomparable"],
)
@pytest.mark.parametrize("reference_source", ["research", "production"])
def test_observer_is_strictly_zero_write_and_returns_exact_six_rows(
    tmp_path, monkeypatch, outcome, reference_source,
):
    layout = _observer_layout(tmp_path)
    research, production, engine, activation, definitions, evidence, bootstraps = layout
    reference_root = production if reference_source == "production" else research
    reference_path = reference_root / "reference-cycle"
    current_path = production / "current-cycle"
    reference_path.mkdir()
    current_path.mkdir()
    monkeypatch.setattr(module, "SourceMutationObserver", _SyntheticGuard)
    import quantpits.research.forward_definition_evidence as evidence_module
    import quantpits.research.forward_observation as observation_module
    monkeypatch.setattr(
        evidence_module, "adopt_forward_definition_evidence", lambda *_: _SyntheticEvidence(),
    )
    monkeypatch.setattr(
        observation_module, "observe_frozen_shadow_forward_definition_candidate",
        lambda *_: _SyntheticDefinition(),
    )
    monkeypatch.setattr(
        module, "_bootstrap_authority",
        lambda *_: ("2026-08-14", {
            "phase37a_seal_digest": {"sealed": True},
            "phase37a_manifest_digest": {"manifest": True},
        }),
    )
    calls = []
    def fresh_evidence(prod, res, *args):
        assert prod == production and res == research
        calls.append("fresh_evidence")
        return _SyntheticEvidence()
    def fresh_candidate(prod, res, *args):
        assert prod == production and res == research
        calls.append("fresh_candidate")
        return _SyntheticDefinition()
    monkeypatch.setattr(evidence_module, "adopt_fresh_champion_segment_definition_evidence", fresh_evidence)
    monkeypatch.setattr(observation_module, "observe_fresh_champion_segment_candidate", fresh_candidate)
    reference_manifest = {"kind": "reference"}
    current_manifest = {"kind": "current"}
    monkeypatch.setattr(
        module, "_cycle_authority",
        lambda root, cycle: (
            reference_path, reference_manifest, {}
        ) if cycle == "2026-08-14" and root == reference_root else (current_path, current_manifest, {}),
    )
    activation_bytes = activation.read_bytes()
    def read_regular(path, **_kwargs):
        if path == activation:
            return activation_bytes, ()
        return (b"seal" if path.name == "seal.json" else b"manifest"), ()
    monkeypatch.setattr(module, "_read_regular", read_regular)
    before = module._selected_fingerprint((tmp_path,))
    source_receipt = {
        "phase37a_seal_digest": module._digest(b"seal", "raw_bytes"),
        "phase37a_manifest_digest": module._digest(b"manifest", "raw_bytes"),
    }
    monkeypatch.setattr(
        module, "_bootstrap_authority", lambda *_: ("2026-08-14", source_receipt),
    )
    monkeypatch.setattr(module, "_source_matches_definition", lambda *_: True)
    monkeypatch.setattr(module, "_intent_matches_definition", lambda *_: True)
    monkeypatch.setattr(module, "_engine_commit", lambda manifest: manifest["kind"] * 40)
    monkeypatch.setattr(module, "_git_blob_projection", lambda _engine, _commit: {"code": 1})
    monkeypatch.setattr(module, "_source_projection", lambda manifest: {
        "source": 2 if outcome == "version" and manifest["kind"] == "current" else 1,
    })
    monkeypatch.setattr(module, "_ensemble_projection", lambda *_: {
        "protocol": "PER_MODEL_CROSS_SECTIONAL_PERCENTILE_RANK_EQUAL_MEAN_V1",
        "ensemble": 1,
    })
    monkeypatch.setattr(module, "_prediction_projection", lambda _manifest: {"parents": 1})
    def market(manifest):
        if outcome == "incomparable" and manifest["kind"] == "current":
            raise module._ComponentIncomparable("MARKET_POLICY_INCOMPARABLE")
        return {"market": 1}
    monkeypatch.setattr(module, "_market_projection", market)
    def intent(path, _manifest):
        if outcome == "reference_intent_incomparable" and path == reference_path:
            raise module._ComponentIncomparable("STRATEGY_CONFIG_FALLBACK_REQUIRED")
        return {
            **module._definition_intent_projection(_SyntheticDefinition()),
            "resolved_combo": "CHAMPION_4",
        }
    monkeypatch.setattr(module, "_intent_projection", intent)
    result = observe_production_decision_surface(
        research, production, engine, "2026-08-21", activation,
        definitions, evidence, bootstraps, "bootstrap.synthetic",
        reference_source=reference_source,
    )
    assert calls == (["fresh_evidence", "fresh_candidate"] if reference_source == "production" else [])
    expected = {
        "same": "SAME_CHAMPION_SEGMENT", "version": "VERSION_BREAK",
        "incomparable": "INCOMPARABLE",
        "reference_intent_incomparable": "INCOMPARABLE",
    }[outcome]
    assert result.status == expected
    assert len(result.components) == 6
    if outcome == "reference_intent_incomparable":
        assert result.components[-1].reason_code == "STRATEGY_CONFIG_FALLBACK_REQUIRED"
    assert module._selected_fingerprint((tmp_path,)) == before


def test_safe_summary_contains_no_cycle_ids_paths_or_private_values():
    encoded = canonical_json_bytes(_result().to_safe_summary_dict())
    assert b"2026-08-14" not in encoded
    assert b"2026-08-21" not in encoded
    assert b"/tmp" not in encoded
    assert b"recorder" not in encoded.lower()


def test_reference_source_default_preserves_legacy():
    import inspect
    assert inspect.signature(observe_production_decision_surface).parameters["reference_source"].default == "research"


@pytest.mark.parametrize("source_state", ["missing", "corrupt"])
def test_reference_source_does_not_fallback(tmp_path, monkeypatch, source_state):
    import shutil
    from tests.quantpits.research.test_forward_observation import _fresh_split_bundle
    production, research, activation, cycle, _ = _fresh_split_bundle(tmp_path)
    shadow = research / "research/shadow_v1"
    definitions, evidence, bootstraps = (shadow / name for name in ("definitions", "definition_evidence", "bootstraps"))
    for path in (definitions, evidence, bootstraps):
        path.mkdir(mode=0o700)
    copy = research / "data/evidence/v1/cycles" / cycle.name
    copy.parent.mkdir(parents=True)
    shutil.copytree(str(cycle), str(copy))
    if source_state == "missing":
        shutil.rmtree(str(cycle))
    else:
        (cycle / "seal.json").write_bytes(b"{}\n")
    engine, _ = _synthetic_engine(tmp_path)
    with pytest.raises(module.DecisionSurfaceInputError):
        observe_production_decision_surface(research, production, engine, "2026-09-04", activation,
            definitions, evidence, bootstraps, "bootstrap.synthetic", reference_source="production")
    assert copy.is_dir()
    assert cycle.exists() is (source_state == "corrupt")


@pytest.mark.parametrize("reference_source", ["research", "production"])
def test_failed_evidence_does_not_observe_candidate(tmp_path, monkeypatch, reference_source):
    from types import SimpleNamespace
    import quantpits.research.forward_definition_evidence as evidence_module
    import quantpits.research.forward_observation as observation_module
    research, production, engine, activation, definitions, evidence, bootstraps = _observer_layout(tmp_path)
    monkeypatch.setattr(module, "SourceMutationObserver", _SyntheticGuard)
    name = "adopt_forward_definition_evidence" if reference_source == "research" else "adopt_fresh_champion_segment_definition_evidence"
    observer = "observe_frozen_shadow_forward_definition_candidate" if reference_source == "research" else "observe_fresh_champion_segment_candidate"
    monkeypatch.setattr(evidence_module, name, lambda *args: SimpleNamespace(status="CONFLICT", definition_evidence_complete=False))
    monkeypatch.setattr(observation_module, observer, lambda *args: pytest.fail("failed adoption must stop before candidate observation"))
    with pytest.raises(module.DecisionSurfaceInputError):
        observe_production_decision_surface(research, production, engine, "2026-09-04", activation,
            definitions, evidence, bootstraps, "bootstrap.synthetic", reference_source=reference_source)
