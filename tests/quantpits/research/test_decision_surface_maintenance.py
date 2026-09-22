"""Reviewed real source variants; temporary Git and file-backed model ancestry."""
import copy
import shutil
from pathlib import Path

import pytest

from quantpits.research import decision_surface as s
from tests.quantpits.research.test_decision_surface import _git, _observer_layout, _SyntheticDefinition, _SyntheticEvidence, _SyntheticLeaf
from tests.quantpits.research.test_model_continuity import copied_models


def reviewed_train_sources():
    current = (Path(s.__file__).parents[1] / "utils/train_utils.py").read_text()
    old = current
    blocks = [
        "        from quantpits.training.model_identity import prediction_origin_tags\n"
        "        origin_tags = prediction_origin_tags(\n"
        "            source_exp, source_id,\n"
        "            lambda exp, rid: R.get_recorder(experiment_name=exp, recorder_id=rid).list_tags(),\n"
        "            model_name,\n"
        "        )\n",
        "                        **origin_tags,\n",
        "        # Preserve direct parent audit tags and independently traced training origin.\n"
        "        from quantpits.training.model_identity import prediction_origin_tags\n"
        "        origin_tags = prediction_origin_tags(\n"
        "            source_experiment, source_record_id,\n"
        "            lambda exp, rid: R.get_recorder(experiment_name=exp, recorder_id=rid).list_tags(),\n"
        "            model_name,\n"
        "        )\n",
        "                **origin_tags,\n",
    ]
    for block in blocks:
        assert old.count(block) == 1
        old = old.replace(block, "")
    assert s._digest(old.encode(), "raw_bytes")["value"] == "64cc427c57795ff45dc1629ef06016c8d67d471780ed42f45c746f64a1026e19"
    assert s._digest(current.encode(), "raw_bytes")["value"] == "36a4ca3c314faf77636b59467ef42b16386dd3cba70082b73ab88f1032bc203b"
    return old, current


def reviewed_engine(engine):
    # Reconstruct the actual old bytes, independently checked against the
    # reviewed blob identity. This also works in shallow CI checkouts.
    source = Path(s.__file__).parents[2]
    old, new = reviewed_train_sources()
    _git(engine, "init", "-q")
    _git(engine, "config", "user.email", "test@example.invalid")
    _git(engine, "config", "user.name", "Test")
    for logical in s.CURATED_CODE_PATHS:
        path = engine / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source / logical), str(path))
    train = engine / "quantpits/utils/train_utils.py"
    train.write_text(old)
    _git(engine, "add", ".")
    _git(engine, "commit", "-qm", "old reviewed bytes")
    reference = _git(engine, "rev-parse", "HEAD")
    train.write_text(new)
    helper = engine / s.ORIGIN_TAGS_HELPER
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source / s.ORIGIN_TAGS_HELPER), str(helper))
    _git(engine, "add", ".")
    _git(engine, "commit", "-qm", "reviewed annotations")
    current = _git(engine, "rev-parse", "HEAD")
    assert s._origin_tags_pair(s._digest(s._git_blob_projection(engine, reference)),
                               s._digest(s._git_blob_projection(engine, current)))
    return reference, current


@pytest.mark.parametrize("variant", ["maintenance", "exact", "economic", "helper", "parent", "market", "multi", "third", "helper_exact", "reverse"])
def test_real_surface_source_and_engine(tmp_path, monkeypatch, copied_models, variant):
    layout = _observer_layout(tmp_path)
    research, production, engine, activation, definitions, evidence, bootstraps = layout
    old, new = reviewed_engine(engine)
    source_root, manifests = copied_models
    shutil.copytree(str(source_root / "mlruns"), str(production / "mlruns"))
    reference, current = copy.deepcopy(manifests)
    if variant in ("exact", "helper_exact"):
        old = new
    if variant in ("economic", "third"):
        path = engine / "quantpits/utils/train_utils.py"
        if variant == "economic":
            path.write_text(path.read_text().replace("fm.predict(dataset=dataset)", "fm.predict(dataset=dataset) * 2"))
        else:
            path.write_text(path.read_text() + "\n# unreviewed third implementation\n")
        _git(engine, "add", ".")
        _git(engine, "commit", "-qm", "unreviewed")
        new = _git(engine, "rev-parse", "HEAD")
    if variant in ("helper", "helper_exact"):
        path = engine / s.ORIGIN_TAGS_HELPER
        path.write_text(path.read_text() + "\n# changed dependency\n")
        _git(engine, "add", ".")
        _git(engine, "commit", "-qm", "dependency change")
        new = _git(engine, "rev-parse", "HEAD")
    if variant == "parent":
        (production / "mlruns/1/source_0_b/tags/source_record_id").write_text("missing")
    if variant == "reverse":
        old, new = new, old
    reference["engine_identity"] = {"commit": old}
    current["engine_identity"] = {"commit": new}
    reference_path = production / "reference"
    current_path = production / "current"
    for path in (reference_path, current_path):
        path.mkdir()
        for name in ("seal.json", "manifest.json"):
            (path / name).write_bytes(b"{}")
            (path / name).chmod(0o600)
    receipt = {"phase37a_seal_digest": s._digest(b"{}", "raw_bytes"),
               "phase37a_manifest_digest": s._digest(b"{}", "raw_bytes")}
    import quantpits.research.forward_definition_evidence as ev
    import quantpits.research.forward_observation as obs
    monkeypatch.setattr(ev, "adopt_fresh_champion_segment_definition_evidence", lambda *a: _SyntheticEvidence())
    candidate = _SyntheticDefinition()
    champion = candidate.compiled_definitions.champion.to_dict()
    champion["source_members"] = [{"position": row["position"], "source_id": row["source_id"],
        "model_artifact_digest": row["artifact_inventory_digest"]}
        for row in s._source_projection(reference)["members"]]
    candidate.compiled_definitions.champion = _SyntheticLeaf(champion)
    monkeypatch.setattr(obs, "observe_fresh_champion_segment_candidate", lambda *a: candidate)
    monkeypatch.setattr(s, "_bootstrap_authority", lambda *a: ("2026-08-14", receipt))
    monkeypatch.setattr(s, "_cycle_authority", lambda root, day, **kw: (reference_path, reference, {})
                        if day == "2026-08-14" else (current_path, current, {}))
    monkeypatch.setattr(s, "_intent_matches_definition", lambda *a: True)
    monkeypatch.setattr(s, "_ensemble_projection", lambda *a: {
        "protocol": "PER_MODEL_CROSS_SECTIONAL_PERCENTILE_RANK_EQUAL_MEAN_V1"})
    def market(manifest):
        if manifest is current and variant in ("market", "multi"):
            if variant == "multi":
                raise s._ComponentIncomparable("MARKET_UNAVAILABLE")
            return {"market": "changed"}
        return {"market": "stable"}
    monkeypatch.setattr(s, "_market_projection", market)
    monkeypatch.setattr(s, "_intent_projection", lambda path, manifest: {"policy": 2 if
                        variant == "multi" and manifest is current else 1})
    result = s.observe_production_decision_surface(research, production, engine, "2026-08-21",
        activation, definitions, evidence, bootstraps, "bootstrap.synthetic", reference_source="production")
    expected = {"maintenance": "SAME_CHAMPION_SEGMENT", "exact": "SAME_CHAMPION_SEGMENT",
                "economic": "VERSION_BREAK", "third": "VERSION_BREAK", "market": "VERSION_BREAK",
                "helper_exact": "INCOMPARABLE", "reverse": "VERSION_BREAK",
                "helper": "INCOMPARABLE", "parent": "INCOMPARABLE", "multi": "INCOMPARABLE"}
    assert result.status == expected[variant], result.to_safe_summary_dict()
    if variant in ("maintenance", "exact"):
        from quantpits.research.forward_intent_preparation import _engine
        assert _engine(engine, current)[0] == new
    if variant == "maintenance":
        row = result.components[0]
        assert row.comparison == "COMPATIBLE" and row.reference_digest != row.current_digest
        assert row.reason_code == s.ORIGIN_TAGS_RULE
        assert result.to_safe_summary_dict()["schema_version"] == 2
        admission = s.maintenance_admission(result)
        s.validate_maintenance_admission(admission)
        admission["rule_id"] = "caller-approved"
        with pytest.raises(s.DecisionSurfaceContractError):
            s.validate_maintenance_admission(admission)
    if variant == "multi":
        assert result.components[0].comparison == "DIFFERENT"
        assert result.components[4].comparison == "INCOMPARABLE"
        assert result.components[5].comparison == "DIFFERENT"
