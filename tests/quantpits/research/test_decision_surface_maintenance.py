"""Controlled origin-annotation variants; real Git and model ancestry."""
import ast
import copy
import shutil
from pathlib import Path

import pytest

from quantpits.research import decision_surface as s
from tests.quantpits.research.test_decision_surface import _git, _observer_layout, _SyntheticDefinition, _SyntheticEvidence, _SyntheticLeaf
from tests.quantpits.research.test_model_continuity import copied_models


def prediction_origin_source_variants(current=None):
    """Remove only origin annotations from the current prediction functions.

    These are controlled before/after inputs, not historical whole-file bytes.
    Unrelated changes to training must remain present in both versions.
    """
    if current is None:
        current = (Path(s.__file__).parents[1] / "utils/train_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(current)
    removed_lines = set()
    for name in ("predict_single_model", "predict_cpcv_model"):
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        nodes = list(ast.walk(function))
        imports = [node for node in nodes if isinstance(node, ast.ImportFrom)
                   and node.module == "quantpits.training.model_identity"
                   and [alias.name for alias in node.names] == ["prediction_origin_tags"]]
        assignments = [node for node in nodes if isinstance(node, ast.Assign)
                       and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id == "origin_tags"]
        expansions = [node for node in nodes if isinstance(node, ast.keyword)
                      and node.arg is None and isinstance(node.value, ast.Name)
                      and node.value.id == "origin_tags"]
        assert len(imports) == len(assignments) == len(expansions) == 1, name
        # keyword nodes have no source positions on Python 3.8; their Name
        # values do, and each expansion occupies its own line in these calls.
        for node in imports + assignments + [keyword.value for keyword in expansions]:
            removed_lines.update(range(node.lineno, node.end_lineno + 1))
    old = "".join(line for number, line in enumerate(current.splitlines(keepends=True), 1)
                  if number not in removed_lines)
    ast.parse(old)
    return old, current


def test_prediction_origin_variants_preserve_unrelated_training_change():
    source = (Path(s.__file__).parents[1] / "utils/train_utils.py").read_text(encoding="utf-8")
    training = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "train_single_model")
    lines = source.splitlines(keepends=True)
    lines.insert(training.body[0].end_lineno, "    unrelated_training_revision = 2\n")
    edited = "".join(lines)
    before, after = prediction_origin_source_variants(edited)
    assert after == edited
    assert "unrelated_training_revision = 2" in before
    training_bodies = [next(node for node in ast.parse(value).body
                           if isinstance(node, ast.FunctionDef) and node.name == "train_single_model")
                       for value in (before, after)]
    assert ast.dump(training_bodies[0]) == ast.dump(training_bodies[1])


def origin_annotation_engine(engine, monkeypatch=None):
    # Temporary sources work without historical Git objects in shallow CI.
    source = Path(s.__file__).parents[2]
    old, new = prediction_origin_source_variants()
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
    _git(engine, "commit", "-qm", "without origin annotations")
    reference = _git(engine, "rev-parse", "HEAD")
    train.write_text(new)
    helper = engine / s.ORIGIN_TAGS_HELPER
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(source / s.ORIGIN_TAGS_HELPER), str(helper))
    _git(engine, "add", ".")
    _git(engine, "commit", "-qm", "with origin annotations")
    current = _git(engine, "rev-parse", "HEAD")
    if monkeypatch is not None:
        # Isolate the legacy rule's reviewed inputs from the evolving checkout.
        # Capture this pair once, before tests mutate code or dependencies;
        # retain the real comparison, dependency checks and admission validator.
        reference_digest = s._digest(s._git_blob_projection(engine, reference))
        current_digest = s._digest(s._git_blob_projection(engine, current))
        dependency_digest = s._digest(helper.read_bytes(), "raw_bytes")
        monkeypatch.setattr(s, "_ORIGIN_TAGS_OLD", reference_digest["value"])
        monkeypatch.setattr(s, "_ORIGIN_TAGS_NEW", current_digest["value"])
        monkeypatch.setattr(s, "_ORIGIN_TAGS_DEPENDENCY", dependency_digest["value"])
        monkeypatch.setattr(s, "_ORIGIN_TAGS_DEPENDENCY_SIZE", dependency_digest["size_bytes"])
        assert s._origin_tags_pair(reference_digest, current_digest)
    return reference, current


@pytest.mark.parametrize("variant", ["maintenance", "exact", "economic", "helper", "parent", "market", "multi", "third", "helper_exact", "reverse"])
def test_real_surface_source_and_engine(tmp_path, monkeypatch, copied_models, variant):
    layout = _observer_layout(tmp_path)
    research, production, engine, activation, definitions, evidence, bootstraps = layout
    old, new = origin_annotation_engine(engine, monkeypatch)
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
