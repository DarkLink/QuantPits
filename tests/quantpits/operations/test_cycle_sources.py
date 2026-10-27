"""Per-cycle source selection preserves default requests and frozen sources."""

import copy
import json

import pytest

from quantpits.operations.config import CycleError, cycle_request, load_project, selected_fit_records
from quantpits.operations.cycle import _preflight, build_steps, execute, load_state, preview
from quantpits.operations.native import NativeSteps, save_json
from quantpits.utils.workspace import fingerprint_file, fingerprint_value
from tests.quantpits.operations.scenario import scenario, write_json


@pytest.fixture
def project(tmp_path):
    return load_project(scenario(tmp_path, with_broker=False))


def request(project, inputs=None, cycle="sources"):
    return cycle_request(project, cycle=cycle, anchor="2024-05-17", value_through="2024-05-17",
                         trade_date="2024-05-21", since="2024-05-10", inputs=inputs)


def predict(steps, signal="candidate"):
    return next(s for s in steps if s["id"] == signal + "-predict")


def source(project, name="chosen.json", recorder="different-fit", signal="candidate"):
    ctx = project.members[project.signals[signal]["producer"]].ctx
    path = ctx.root / "data" / name
    write_json(path, {"experiment_name": "fit", "models": {"small@static": recorder}})
    return path


def test_absent_override_keeps_old_shape_and_explicit_empty_is_a_new_request(project):
    old = request(project)
    assert old["inputs"] == {}
    assert all("fit_records" not in s for s in build_steps(project, old))
    assert preview(project, old)["signals"] == project.signals
    explicit = request(project, {"fit_records": {}})
    assert explicit["inputs"] == {"fit_records": {}}
    assert fingerprint_value(explicit) != fingerprint_value(old)
    assert predict(build_steps(project, old))["fit_sha256"] == predict(build_steps(project, explicit))["fit_sha256"]


@pytest.mark.parametrize("absolute", [False, True])
def test_partial_override_uses_producer_and_keeps_other_signal_default(project, absolute):
    path = source(project)
    chosen = str(path) if absolute else "data/chosen.json"
    selected = request(project, {"fit_records": {"candidate": chosen}})
    before = project.path.read_bytes(), project.identity
    steps = build_steps(project, selected)
    assert selected_fit_records(project, selected, "candidate") == path
    assert predict(steps)["fit_sha256"] == fingerprint_file(path)
    assert predict(steps)["fit_records"] == chosen
    assert predict(steps, "shared")["fit_sha256"] == fingerprint_file(project.members["live"].ctx.root / "fit.json")
    assert "fit_records" not in predict(steps, "shared")
    plan = preview(project, selected)
    assert plan["signals"]["candidate"]["fit_records"] == chosen
    assert plan["signals"]["shared"] == project.signals["shared"]
    assert (project.path.read_bytes(), project.identity) == before


@pytest.mark.parametrize("overrides", [None, [], "file", {"unknown": "fit.json"},
                                      {"candidate": None}, {"candidate": 3},
                                      {"candidate": ""}, {"candidate": "  "},
                                      {"candidate": "bad\x00file"}, {"candidate": "../../elsewhere.json"}])
def test_invalid_overrides_fail_without_fallback(project, overrides):
    with pytest.raises(CycleError):
        request(project, {"fit_records": overrides})


def test_source_symlink_cannot_read_another_workspace(project):
    producer = project.members["paper_b"].ctx.root
    other = project.members["live"].ctx.root / "fit.json"
    (producer / "alias.json").symlink_to(other)
    with pytest.raises(CycleError, match="escapes"):
        request(project, {"fit_records": {"candidate": "alias.json"}})


def test_bound_content_change_is_rejected_before_freeze(project):
    path = source(project)
    selected = request(project, {"fit_records": {"candidate": "data/chosen.json"}})
    step = predict(build_steps(project, selected))
    source(project, recorder="replacement")
    with pytest.raises(CycleError, match="changed since request capture"):
        _preflight(project, selected, step)
    with pytest.raises(CycleError, match="changed since request capture"):
        NativeSteps(project, selected, None, project.cycle_dir("sources")).execute(step)
    assert not (project.signal_dir("sources", "candidate") / "fit_records.json").exists()


def test_missing_source_waits_without_reading_default(project):
    selected = request(project, {"fit_records": {"candidate": "data/later.json"}})
    step = predict(build_steps(project, selected))
    assert step["fit_sha256"] is None
    assert "later.json" in _preflight(project, selected, step)
    assert project.members["paper_b"].ctx.path("fit.json").exists()


@pytest.mark.parametrize("change", ["delete", "replace", "external_link"])
def test_frozen_source_is_used_without_reselecting_original(project, change):
    path = source(project)
    selected = request(project, {"fit_records": {"candidate": "data/chosen.json"}})
    step = predict(build_steps(project, selected))
    frozen = project.signal_dir("sources", "candidate") / "fit_records.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(path.read_bytes())
    if change == "delete":
        path.unlink()
    elif change == "replace":
        source(project, recorder="later")
    else:
        path.unlink()
        path.symlink_to(project.members["live"].ctx.root / "fit.json")
    state = {"schema_version": 1, "project": str(project.path), "project_identity": project.identity,
             "request": selected, "request_identity": fingerprint_value(selected), "steps": [step],
             "steps_identity": fingerprint_value([step]), "results": {step["id"]: {"status": "pending"}}}
    save_json(project.cycle_dir("sources") / "progress.json", state)
    assert load_state(project, "sources") == state
    assert _preflight(project, selected, step) is None
    class Capture:
        def run(self, ctx, module, arguments, log):
            assert arguments[arguments.index("--source-records") + 1] == str(frozen)
            assert json.loads(frozen.read_text())["models"]["small@static"] == "different-fit"
            return 0
    assert NativeSteps(project, selected, Capture(), project.cycle_dir("sources")).execute(step) == 0


def test_fixed_copy_tampering_is_rejected(project):
    path = source(project)
    selected = request(project, {"fit_records": {"candidate": "data/chosen.json"}})
    step = predict(build_steps(project, selected))
    frozen = project.signal_dir("sources", "candidate") / "fit_records.json"
    write_json(frozen, {"experiment_name": "fit", "models": {"small@static": "tampered"}})
    with pytest.raises(CycleError, match="changed since request capture"):
        _preflight(project, selected, step)
    with pytest.raises(CycleError, match="fixed fit source differs"):
        NativeSteps(project, selected, None, project.cycle_dir("sources")).execute(step)


def test_same_cycle_different_override_is_rejected(project):
    path = source(project)
    selected = request(project, {"fit_records": {"candidate": "data/chosen.json"}})
    steps = build_steps(project, selected)
    save_json(project.cycle_dir("sources") / "progress.json", {
        "schema_version": 1, "project": str(project.path), "project_identity": project.identity,
        "request": selected, "request_identity": fingerprint_value(selected), "steps": steps,
        "steps_identity": fingerprint_value(steps), "results": {s["id"]: {"status": "pending"} for s in steps}})
    changed = copy.deepcopy(selected)
    changed["inputs"]["fit_records"]["candidate"] = "fit.json"
    with pytest.raises(CycleError, match="cycle request changed"):
        execute(project, request=changed)


def test_saved_display_binding_cannot_disagree_with_requested_source(project):
    source(project)
    selected = request(project, {"fit_records": {"candidate": "data/chosen.json"}})
    steps = build_steps(project, selected)
    predict(steps)["fit_records"] = "fit.json"
    save_json(project.cycle_dir("sources") / "progress.json", {
        "schema_version": 1, "project": str(project.path), "project_identity": project.identity,
        "request": selected, "request_identity": fingerprint_value(selected), "steps": steps,
        "steps_identity": fingerprint_value(steps), "results": {s["id"]: {"status": "pending"} for s in steps}})
    with pytest.raises(CycleError, match="saved source binding differs"):
        load_state(project, "sources")
