from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from quantpits.evidence.contracts import TypedDigest, canonical_json_bytes
import quantpits.research.forward_portfolio_source as source_module
from quantpits.research.forward_portfolio_source import (
    ForwardPortfolioSourceContractError,
    ForwardPortfolioSourceInputError,
    ForwardPortfolioSourceObservation,
    observe_forward_portfolio_source,
)


CYCLE = "2026-08-28"


def _write(path, data, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def _problem(code="deep_analysis_missing", evidence_class="deep_analysis", blocks=True):
    return {
        "code": code,
        "evidence_class": evidence_class,
        "detail": "synthetic unrelated evidence debt",
        "blocks_complete": blocks,
    }


def _portfolio(**changes):
    value = {
        "current_cash": "-1.25",
        "current_holding": [
            {"instrument": "SH600001", "value": "100", "amount": "800.5"},
            {"instrument": "SZ000002", "value": "200", "amount": "1500"},
        ],
    }
    value.update(changes)
    return value


def _manifest(portfolio_data, source_digest, ranking_digest, problems):
    semantic = TypedDigest.canonical(
        json.loads(portfolio_data.decode("utf-8")), "semantic_config",
    )
    status = "sealed_partial" if any(item["blocks_complete"] for item in problems) else "sealed_complete"
    manifest = {
        "schema_version": 1,
        "cycle_identity": {
            "cycle_id": CYCLE, "research_epoch_id": "SYNTHETIC", "evidence_as_of": CYCLE,
        },
        "engine_identity": {},
        "workspace_identity": {},
        "data_identity": {},
        "run_evidence": [],
        "model_and_ensemble_lineage": {},
        "ranking": {"status": "complete", "ranking_digest": ranking_digest.to_dict()},
        "portfolio_state": {
            "path": "config/prod_config.json",
            "status": "observed",
            "digest": source_digest.to_dict(),
            "preservation_status": "embedded",
            "detail": "",
            "canonical_digest": semantic.to_dict(),
            "holding_count": len(json.loads(portfolio_data)["current_holding"]),
        },
        "decision_state": {"status": "not_recorded", "as_of_capture": True},
        "referenced_evidence": [],
        "preservation": {"embedded_object_count": 1, "named_file_count": 2},
        "problems": problems,
        "capture_time": "2026-08-30T00:00:00+00:00",
        "status": status,
        "request_content_digest": {},
    }
    replay = {
        key: value for key, value in manifest.items()
        if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
    }
    manifest["request_content_digest"] = TypedDigest.canonical(replay).to_dict()
    return manifest


def _write_bundle(root, *, problems=(), portfolio=None):
    root.mkdir()
    cycle = root / "data" / "evidence" / "v1" / "cycles" / CYCLE
    portfolio_data = canonical_json_bytes(_portfolio() if portfolio is None else portfolio)
    ranking_data = b"instrument,score,rank\nSH600001,1,1\n"
    source_data = canonical_json_bytes({"private_source": "synthetic"})
    source_digest = TypedDigest.raw(source_data)
    ranking_digest = TypedDigest.raw(ranking_data)
    portfolio_digest = TypedDigest.raw(portfolio_data)
    manifest = _manifest(portfolio_data, source_digest, ranking_digest, list(problems))
    manifest_data = canonical_json_bytes(manifest)
    named = {
        "ranking.csv": ranking_digest.to_dict(),
        "portfolio_state.json": portfolio_digest.to_dict(),
    }
    seal = {
        "schema_version": 1,
        "cycle_id": CYCLE,
        "status": manifest["status"],
        "manifest_digest": TypedDigest.raw(manifest_data).to_dict(),
        "artifact_root_digest": TypedDigest.canonical({
            "objects": [source_digest.value], "named_files": named,
        }).to_dict(),
        "object_digests": [source_digest.value],
        "named_file_digests": named,
    }
    _write(cycle / "objects" / source_digest.value[:2] / source_digest.value, source_data)
    _write(cycle / "ranking.csv", ranking_data)
    _write(cycle / "portfolio_state.json", portfolio_data)
    _write(cycle / "manifest.json", manifest_data)
    _write(cycle / "seal.json", canonical_json_bytes(seal))
    (cycle / "objects" / source_digest.value[:2]).chmod(0o700)
    (cycle / "objects").chmod(0o700)
    cycle.chmod(0o700)
    return cycle


def _load(path):
    return json.loads(path.read_bytes())


def _rewrite_manifest_and_seal(cycle, manifest, *, refresh_request=True):
    if refresh_request:
        replay = {
            key: value for key, value in manifest.items()
            if key not in {"capture_time", "status", "request_content_digest", "workspace_identity"}
        }
        manifest["request_content_digest"] = TypedDigest.canonical(replay).to_dict()
    manifest_data = canonical_json_bytes(manifest)
    seal = _load(cycle / "seal.json")
    seal["status"] = manifest["status"]
    seal["manifest_digest"] = TypedDigest.raw(manifest_data).to_dict()
    _write(cycle / "manifest.json", manifest_data)
    _write(cycle / "seal.json", canonical_json_bytes(seal))


def _snapshot(root):
    result = []
    for path in sorted(root.rglob("*")):
        info = os.lstat(str(path))
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes() if path.is_file() else None
        result.append((relative, info.st_mode, info.st_ino, data))
    return tuple(result)


@pytest.mark.parametrize(
    "problems,expected",
    [
        ((), "VERIFIED"),
        ((_problem("deep_analysis_missing"),), "VERIFIED_WITH_UNRELATED_CYCLE_DEBT"),
        ((_problem("deep_analysis_incomplete"),), "VERIFIED_WITH_UNRELATED_CYCLE_DEBT"),
        ((_problem("deep_analysis_missing"), _problem("deep_analysis_incomplete")),
         "VERIFIED_WITH_UNRELATED_CYCLE_DEBT"),
    ],
)
def test_exact_complete_and_allowlisted_partial_sources_grant_member_capability(tmp_path, problems, expected):
    root = tmp_path / "workspace"
    _write_bundle(root, problems=problems)
    before = _snapshot(root)
    result = observe_forward_portfolio_source(root, CYCLE)
    assert result.status == expected
    assert result.portfolio_member_verified is True
    assert result.bootstrap_source_capability is True
    assert result.did_write is False
    assert result.verified_portfolio() == _portfolio()
    assert len(result.problem_inventory()) == len(problems)
    assert _snapshot(root) == before
    safe = result.to_safe_dict()
    rendered = json.dumps(safe, sort_keys=True)
    for secret in ("-1.25", "SH600001", "SZ000002", "800.5", str(root)):
        assert secret not in rendered
    assert safe["prospective_claim"] is False
    assert safe["promotion_capability"] is False


def test_empty_portfolio_is_a_valid_exact_member_shape(tmp_path):
    root = tmp_path / "workspace"
    empty = {"current_cash": "0", "current_holding": []}
    _write_bundle(root, portfolio=empty)
    result = observe_forward_portfolio_source(root, CYCLE)
    assert result.status == "VERIFIED"
    assert result.portfolio_holding_count == 0
    assert result.verified_portfolio() == empty


def test_stable_missing_portfolio_member_is_blocked_and_preserves_cycle_problem_facts(tmp_path):
    root = tmp_path / "workspace"
    cycle = _write_bundle(root, problems=(_problem(),))
    (cycle / "portfolio_state.json").unlink()
    result = observe_forward_portfolio_source(root, CYCLE)
    assert result.status == "BLOCKED"
    assert result.source_cycle_status == "sealed_partial"
    assert result.problem_count == 1
    assert result.problem_inventory()[0]["code"] == "deep_analysis_missing"
    assert not result.bootstrap_source_capability


@pytest.mark.parametrize(
    "problems",
    [
        (_problem("deep_analysis_missing", "other"),),
        (_problem("unknown", "deep_analysis"),),
        (_problem("deep_analysis_missing", blocks=False),),
        (_problem("deep_analysis_missing"), _problem("portfolio_invalid", "portfolio")),
    ],
)
def test_non_allowlisted_or_mixed_problem_inventory_is_blocked(tmp_path, problems):
    root = tmp_path / "workspace"
    _write_bundle(root, problems=problems)
    result = observe_forward_portfolio_source(root, CYCLE)
    assert result.status == "BLOCKED"
    assert not result.portfolio_member_verified
    assert not result.bootstrap_source_capability
    with pytest.raises(ForwardPortfolioSourceContractError):
        result.verified_portfolio()


def test_duplicate_problem_identity_is_blocked_even_after_a_coordinated_reseal(tmp_path):
    root = tmp_path / "workspace"
    problem = _problem()
    _write_bundle(root, problems=(problem, problem))
    result = observe_forward_portfolio_source(root, CYCLE)
    assert (result.status, result.reason_code) == ("BLOCKED", "PROBLEM_INVENTORY_DUPLICATE")


def test_bool_as_integer_schema_and_non_bool_problem_flag_are_blocked(tmp_path):
    schema_root = tmp_path / "schema"
    schema_cycle = _write_bundle(schema_root)
    manifest = _load(schema_cycle / "manifest.json")
    manifest["schema_version"] = True
    _rewrite_manifest_and_seal(schema_cycle, manifest)
    assert observe_forward_portfolio_source(schema_root, CYCLE).status == "BLOCKED"

    problem_root = tmp_path / "problem"
    problem_cycle = _write_bundle(problem_root)
    manifest = _load(problem_cycle / "manifest.json")
    manifest["problems"] = [_problem()]
    manifest["problems"][0]["blocks_complete"] = 1
    manifest["status"] = "sealed_partial"
    _rewrite_manifest_and_seal(problem_cycle, manifest)
    assert observe_forward_portfolio_source(problem_root, CYCLE).status == "BLOCKED"


@pytest.mark.parametrize(
    "portfolio",
    [
        {"current_cash": "0", "current_holding": [
            {"instrument": "SH600001", "value": "1.5", "amount": "1"},
        ]},
        {"current_cash": "0", "current_holding": [
            {"instrument": "SH600001", "value": "0", "amount": "1"},
        ]},
        {"current_cash": "0", "current_holding": [
            {"instrument": "SH600001", "value": "100.0", "amount": "1"},
        ]},
        {"current_cash": "-0", "current_holding": []},
        {"current_cash": "0", "current_holding": [
            {"instrument": "US600001", "value": "100", "amount": "1"},
        ]},
        {"current_cash": "0", "current_holding": [
            {"instrument": "SH600001", "value": "100", "amount": "-1"},
        ]},
        {"current_cash": "0", "current_holding": [
            {"instrument": "SH600001", "value": "100", "amount": "1", "extra": "x"},
        ]},
    ],
)
def test_closest_invalid_portfolio_shapes_are_blocked_without_capability(tmp_path, portfolio):
    root = tmp_path / "workspace"
    _write_bundle(root, portfolio=portfolio)
    result = observe_forward_portfolio_source(root, CYCLE)
    assert result.status == "BLOCKED"
    assert not result.bootstrap_source_capability


def test_duplicate_portfolio_instruments_are_blocked(tmp_path):
    root = tmp_path / "workspace"
    portfolio = _portfolio(current_holding=[
        {"instrument": "SH600001", "value": "100", "amount": "1"},
        {"instrument": "SH600001", "value": "200", "amount": "2"},
    ])
    _write_bundle(root, portfolio=portfolio)
    assert observe_forward_portfolio_source(root, CYCLE).reason_code == "PORTFOLIO_INSTRUMENT_DUPLICATE"


def test_manifest_seal_request_named_and_semantic_digest_drifts_are_blocked(tmp_path):
    roots = [tmp_path / name for name in ("manifest", "request", "named", "semantic")]
    cycles = [_write_bundle(root) for root in roots]

    manifest = _load(cycles[0] / "manifest.json")
    manifest["capture_time"] = "2026-08-31T00:00:00+00:00"
    _write(cycles[0] / "manifest.json", canonical_json_bytes(manifest))

    manifest = _load(cycles[1] / "manifest.json")
    manifest["request_content_digest"]["value"] = "0" * 64
    _rewrite_manifest_and_seal(cycles[1], manifest, refresh_request=False)

    seal = _load(cycles[2] / "seal.json")
    seal["named_file_digests"]["portfolio_state.json"]["value"] = "0" * 64
    seal["artifact_root_digest"] = TypedDigest.canonical({
        "objects": seal["object_digests"], "named_files": seal["named_file_digests"],
    }).to_dict()
    _write(cycles[2] / "seal.json", canonical_json_bytes(seal))

    manifest = _load(cycles[3] / "manifest.json")
    manifest["portfolio_state"]["canonical_digest"]["value"] = "0" * 64
    _rewrite_manifest_and_seal(cycles[3], manifest)

    for root in roots:
        result = observe_forward_portfolio_source(root, CYCLE)
        assert result.status == "BLOCKED"
        assert not result.bootstrap_source_capability


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "special"])
def test_noncanonical_portfolio_public_nodes_are_blocked(tmp_path, kind):
    root = tmp_path / "workspace"
    cycle = _write_bundle(root)
    target = cycle / "portfolio_state.json"
    data = target.read_bytes()
    target.unlink()
    if kind == "symlink":
        backing = tmp_path / "backing.json"
        backing.write_bytes(data)
        target.symlink_to(backing)
    elif kind == "hardlink":
        backing = tmp_path / "backing.json"
        backing.write_bytes(data)
        os.link(str(backing), str(target))
    else:
        os.mkfifo(str(target), 0o600)
    result = observe_forward_portfolio_source(root, CYCLE)
    assert result.status == "BLOCKED"
    assert not result.bootstrap_source_capability


def test_extra_unsealed_inventory_is_blocked(tmp_path):
    root = tmp_path / "workspace"
    cycle = _write_bundle(root)
    _write(cycle / "foreign.json", b"{}")
    result = observe_forward_portfolio_source(root, CYCLE)
    assert (result.status, result.reason_code) == ("BLOCKED", "SOURCE_TOP_INVENTORY_INVALID")


def test_transient_cycle_move_away_and_back_is_uncertain(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    cycle = _write_bundle(root)
    original = source_module._read_regular
    moved = {"done": False}

    def moving(root_path, path, **kwargs):
        result = original(root_path, path, **kwargs)
        if path.name == "seal.json" and not moved["done"]:
            displaced = cycle.with_name("displaced")
            cycle.rename(displaced)
            displaced.rename(cycle)
            moved["done"] = True
        return result

    monkeypatch.setattr(source_module, "_read_regular", moving)
    result = observe_forward_portfolio_source(root, CYCLE)
    assert (result.status, result.reason_code) == ("UNCERTAIN", "SOURCE_MUTATION_OBSERVED")
    assert not result.bootstrap_source_capability


@pytest.mark.parametrize("target_kind", ["workspace", "ancestor", "member", "member_replace"])
def test_transient_workspace_and_member_namespace_changes_are_uncertain(
    monkeypatch, tmp_path, target_kind,
):
    root = tmp_path / "workspace"
    cycle = _write_bundle(root)
    member = cycle / "portfolio_state.json"
    original = source_module._read_regular
    moved = {"done": False}

    def moving(root_path, path, **kwargs):
        result = original(root_path, path, **kwargs)
        if path.name == "seal.json" and not moved["done"]:
            if target_kind == "workspace":
                displaced = root.with_name("workspace-displaced")
                root.rename(displaced)
                displaced.rename(root)
            elif target_kind == "ancestor":
                ancestor = root / "data"
                displaced = root / "data-displaced"
                ancestor.rename(displaced)
                displaced.rename(ancestor)
            elif target_kind == "member":
                displaced = member.with_name("portfolio-displaced.json")
                member.rename(displaced)
                displaced.rename(member)
            else:
                data = member.read_bytes()
                member.unlink()
                _write(member, data)
            moved["done"] = True
        return result

    monkeypatch.setattr(source_module, "_read_regular", moving)
    result = observe_forward_portfolio_source(root, CYCLE)
    assert (result.status, result.reason_code) == ("UNCERTAIN", "SOURCE_MUTATION_OBSERVED")
    assert not result.bootstrap_source_capability


def test_deliberate_write_during_call_is_detected_by_the_zero_write_guard(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    cycle = _write_bundle(root)
    original = source_module._observe_bundle

    def writing(*args, **kwargs):
        result = original(*args, **kwargs)
        _write(cycle / "observer-probe", b"probe")
        return result

    monkeypatch.setattr(source_module, "_observe_bundle", writing)
    result = observe_forward_portfolio_source(root, CYCLE)
    assert (result.status, result.reason_code) == ("UNCERTAIN", "SOURCE_MUTATION_OBSERVED")
    assert result.did_write is False


def test_ordinary_observer_failure_fails_closed_and_process_control_propagates(monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    _write_bundle(root)
    monkeypatch.setattr(source_module, "_observe_bundle", lambda *_args: (_ for _ in ()).throw(OSError("boom")))
    result = observe_forward_portfolio_source(root, CYCLE)
    assert result.status == "UNCERTAIN"
    assert not result.bootstrap_source_capability

    for error in (KeyboardInterrupt(), SystemExit(), GeneratorExit()):
        monkeypatch.setattr(
            source_module, "_observe_bundle",
            lambda *_args, error=error: (_ for _ in ()).throw(error),
        )
        with pytest.raises(type(error)):
            observe_forward_portfolio_source(root, CYCLE)


def test_public_construction_replace_and_mutation_cannot_mint_or_retain_authority(tmp_path):
    with pytest.raises(ForwardPortfolioSourceContractError):
        ForwardPortfolioSourceObservation()
    root = tmp_path / "workspace"
    _write_bundle(root)
    result = observe_forward_portfolio_source(root, CYCLE)
    with pytest.raises(ForwardPortfolioSourceContractError):
        replace(result, status="BLOCKED")
    object.__setattr__(result, "status", "BLOCKED")
    with pytest.raises(ForwardPortfolioSourceContractError, match="authority"):
        result.to_safe_dict()
    with pytest.raises(ForwardPortfolioSourceContractError, match="authority"):
        result.verified_portfolio()


def test_result_constructor_revalidates_foreign_aggregate_members(tmp_path):
    root = tmp_path / "workspace"
    _write_bundle(root, problems=(_problem(),))
    result = observe_forward_portfolio_source(root, CYCLE)
    values = dict(vars(result))
    foreign = [_problem("portfolio_invalid", "portfolio")]
    foreign_bytes = canonical_json_bytes(foreign)
    values["_problem_inventory_bytes"] = foreign_bytes
    values["problem_inventory_digest"] = TypedDigest.canonical(foreign).to_dict()
    with pytest.raises(ForwardPortfolioSourceContractError, match="not admissible"):
        ForwardPortfolioSourceObservation._create(source_module._AUTHORITY, **values)

    values = dict(vars(result))
    changed = canonical_json_bytes({"current_cash": "0", "current_holding": []})
    values["_portfolio_bytes"] = changed
    values["portfolio_raw_digest"] = TypedDigest.raw(changed).to_dict()
    values["portfolio_semantic_digest"] = TypedDigest.canonical(
        json.loads(changed), "semantic_config",
    ).to_dict()
    values["portfolio_holding_count"] = 0
    # The copied original observation binding cannot authorize different member bytes.
    with pytest.raises(ForwardPortfolioSourceContractError, match="identity binding"):
        ForwardPortfolioSourceObservation._create(source_module._AUTHORITY, **values)


@pytest.mark.parametrize("cycle", [None, True, 1, "", "2026-8-28", "2026-02-30"])
def test_cycle_representation_is_strict(tmp_path, cycle):
    root = tmp_path / "workspace"
    root.mkdir()
    with pytest.raises(ForwardPortfolioSourceContractError):
        observe_forward_portfolio_source(root, cycle)


def test_workspace_path_must_be_absolute_physical_and_existing(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    with pytest.raises(ForwardPortfolioSourceContractError):
        observe_forward_portfolio_source(Path("relative"), CYCLE)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ForwardPortfolioSourceInputError):
        observe_forward_portfolio_source(alias, CYCLE)
    missing = tmp_path / "missing"
    with pytest.raises(ForwardPortfolioSourceInputError):
        observe_forward_portfolio_source(missing, CYCLE)


def test_symlinked_source_ancestor_never_grants_capability(tmp_path):
    physical_root = tmp_path / "physical"
    _write_bundle(physical_root)
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "data").symlink_to(physical_root / "data", target_is_directory=True)
    result = observe_forward_portfolio_source(root, CYCLE)
    assert result.status in {"BLOCKED", "UNCERTAIN"}
    assert not result.bootstrap_source_capability


def test_missing_cycle_is_a_stable_blocked_observation(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    result = observe_forward_portfolio_source(root, CYCLE)
    assert result.status == "BLOCKED"
    assert result.did_write is False


def test_lazy_public_exports_resolve_without_eager_optional_dependencies():
    import quantpits.research as research

    assert research.ForwardPortfolioSourceObservation is ForwardPortfolioSourceObservation
    assert research.observe_forward_portfolio_source is observe_forward_portfolio_source
