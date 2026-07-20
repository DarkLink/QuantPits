import ast
from pathlib import Path

from quantpits.model_capabilities.catalog import (
    AUTHORITATIVE_CATALOG,
    declared_repository_models,
    repository_wrapper_inventory,
)
from quantpits.model_capabilities.inspector import ModelCapabilityInspector
from quantpits.model_capabilities.probes import ImportObservation


def _not_importable(_module, _class_name):
    return ImportObservation(False, False, False, False, False, False, True, "dependency_missing")


def test_repository_wrapper_inventory_equals_declared_inventory():
    filesystem = repository_wrapper_inventory()
    declared = tuple(module for module, _class_name in declared_repository_models())
    assert len(filesystem) == 43
    assert len(declared) == 43
    assert filesystem == declared
    assert "quantpits.utils.model_wrappers.custom.pytorch_add" in filesystem
    assert any(item.model_class == "ADD" for item in AUTHORITATIVE_CATALOG)
    assert {item.wrapper_kind for item in AUTHORITATIVE_CATALOG} == {
        "custom", "loss_history", "external_passthrough",
    }
    assert any(
        item.model_module == "qlib.contrib.model.linear" and item.model_class == "LinearModel"
        for item in AUTHORITATIVE_CATALOG
    )
    for module, class_name in declared_repository_models():
        relative = Path(*module.split("."))
        source = Path(__file__).resolve().parents[3] / relative.with_suffix(".py")
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        assert class_name in {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def test_raw_declaration_assignment_is_single_and_conservative():
    empty = ModelCapabilityInspector._with_probes(_not_importable).inspect(())
    assert empty.n_declarations == 0
    assert empty.results == ()
    assert empty.unassigned == ()

    first = AUTHORITATIVE_CATALOG[0]
    raw = (first, {"required": True}, AUTHORITATIVE_CATALOG[-1])
    matrix = ModelCapabilityInspector._with_probes(_not_importable).inspect(raw)
    assert matrix.n_declarations == 3
    assert matrix.n_unassigned_declarations == 1
    assert len(matrix.results) == 2
    assert [item.raw_position for item in matrix.results] == [0, 2]
    assert matrix.results[0].identity == first_identity(first)
    assert matrix.results[1].identity == first_identity(AUTHORITATIVE_CATALOG[-1])
    assert matrix.unassigned[0].position == 1
    assert {item.raw_fingerprint for item in matrix.unassigned}.issubset(set(matrix.raw_fingerprints))


def first_identity(declaration):
    from quantpits.model_capabilities.contracts import ModelCapabilityIdentity
    return ModelCapabilityIdentity.from_declaration(declaration)


def test_catalog_rows_are_atomic_and_cover_action_family_vocabularies():
    identities = [item.to_public_dict() for item in AUTHORITATIVE_CATALOG]
    assert {item["action"] for item in identities} == {"train", "incremental", "predict_only", "resume"}
    assert {item["execution_family"] for item in identities} == {"static", "cpcv", "rolling", "cpcv_rolling"}
    assert len(identities) == 704
    assert len(identities) == len({
        tuple(sorted((key, str(value)) for key, value in item.items() if key not in ("required_predicates", "required")))
        for item in identities
    })


def test_catalog_inventory_drift_remains_unassigned(monkeypatch):
    import quantpits.model_capabilities.inspector as inspector_module

    inventory = repository_wrapper_inventory() + ("quantpits.utils.model_wrappers.custom.pytorch_unlisted",)
    monkeypatch.setattr(inspector_module, "repository_wrapper_inventory", lambda: inventory)
    matrix = ModelCapabilityInspector._with_probes(_not_importable).inspect_catalog()
    assert matrix.n_declarations == len(AUTHORITATIVE_CATALOG) + 1
    assert matrix.n_unassigned_declarations == 1
    assert matrix.unassigned[0].reason == "invalid_raw_declaration"
    assert matrix.status == "inventory_invalid"
