"""Per-request order candidate selection leaves default cycle identities alone."""
import copy
import json
from pathlib import Path
from dataclasses import replace

import pytest

from quantpits.operations.config import CycleError, cycle_request, load_project
from quantpits.operations.cycle import build_steps, load_state, preview
from quantpits.operations.native import NativeSteps, save_json
from quantpits.utils.workspace import fingerprint_value
from tests.quantpits.operations.scenario import scenario


def request(project, inputs=None):
    return cycle_request(project, cycle="new_week", anchor="2024-05-17", value_through="2024-05-17",
                         trade_date="2024-05-21", since="2024-05-10", inputs=inputs)


def normalized_baseline(project):
    req = request(project)
    prefix = "cycle-" + fingerprint_value({"project": str(project.path), "request": req})[:16]
    return json.loads(json.dumps(dict(request=req, steps=build_steps(project, req)))
                      .replace(prefix, "<run-prefix>").replace(str(project.path.parent.parent), "<root>"))


def test_default_request_steps_and_run_ids_match_prechange_baseline(tmp_path):
    project = load_project(scenario(tmp_path))
    baseline = json.loads((Path(__file__).parent / "fixtures/cycle_candidates_legacy_baseline.json").read_text())
    assert normalized_baseline(project) == baseline
    assert all("candidate_output" not in step for step in build_steps(project, request(project)))


@pytest.fixture
def project(tmp_path):
    return load_project(scenario(tmp_path))


def order_step(project, selected):
    return next(s for s in build_steps(project, selected) if s["id"] == "live-new")


def test_explicit_selection_and_empty_list_are_distinct_without_default_fields(project):
    default = request(project)
    empty = request(project, {"order_candidates": []})
    assert default["inputs"] == {} and empty["inputs"] == {"order_candidates": []}
    assert fingerprint_value(default) != fingerprint_value(empty)
    assert all("candidate_output" not in s for s in build_steps(project, empty))
    selected = request(project, {"order_candidates": ["live"]})
    steps = build_steps(project, selected)
    bound = [s for s in steps if "candidate_output" in s]
    assert len(bound) == 1 and bound[0]["id"] == "live-new"
    assert bound[0]["candidate_output"] == "output/cycles/project/new_week/orders/order_candidates.json"
    assert [s["id"] for s in steps] == [s["id"] for s in build_steps(project, default)]
    assert next(s for s in preview(project, selected)["steps"] if s["id"] == "live-new")["candidate_output"] == bound[0]["candidate_output"]


@pytest.mark.parametrize("selected", [None, "live", {}, True, 3, ["live","live"], ["unknown"],
                                      ["paper_a"], ["live",None], [["live"]]])
def test_invalid_selection_is_not_silently_reduced(project, selected):
    with pytest.raises(CycleError, match="order_candidates"):
        request(project, {"order_candidates": selected})


@pytest.mark.parametrize("extra", [{"hold":["live"]}, {"skip":["live-new"]}])
def test_selected_member_cannot_hold_or_skip_its_order(project, extra):
    with pytest.raises(CycleError, match="remove the selection"):
        request(project, dict(order_candidates=["live"], **extra))
    # The old choices remain legal without candidate selection.
    build_steps(project, request(project, extra))


def save_state(project, req, steps):
    value = dict(schema_version=1, project=str(project.path), project_identity=project.identity,
        request=req, request_identity=fingerprint_value(req), steps=steps, steps_identity=fingerprint_value(steps),
        results={s["id"]: {"status":"pending"} for s in steps})
    save_json(project.cycle_dir(req["cycle"]) / "progress.json", value)
    return value


@pytest.mark.parametrize("change", ["removed", "path", "paper", "kind", "member", "signal", "unselected_none"])
def test_step_hash_cannot_hide_mismatched_candidate_binding(project, change):
    req = request(project, {"order_candidates":["live"]} if change != "unselected_none" else None)
    steps = build_steps(project, req)
    step = next(s for s in steps if s["id"] == "live-new")
    if change == "removed": del step["candidate_output"]
    elif change == "path": step["candidate_output"] = "output/other.json"
    elif change == "paper":
        next(s for s in steps if s["id"] == "paper_a-new")["candidate_output"] = step.pop("candidate_output")
    elif change == "kind": step["kind"] = "hold"
    elif change == "member": step["member"] = "paper_a"
    elif change == "signal": step["signal"] = "candidate"
    elif change == "unselected_none": step["candidate_output"] = None
    save_state(project,req,steps)  # recompute steps_identity intentionally
    with pytest.raises(CycleError,match="candidate"):
        load_state(project,req["cycle"])


def test_execute_passes_selection_in_original_order_invocation(project):
    req = request(project,{"order_candidates":["live"]})
    step = order_step(project,req)
    calls = []
    class Capture:
        def run(self,ctx,module,args,log):
            calls.append((ctx,module,args))
            return 0
    native = NativeSteps(project,req,Capture(),project.cycle_dir(req["cycle"]))
    assert native.execute(step) == 0 and len(calls) == 1
    ctx,module,args = calls[0]
    assert ctx == project.members["live"].ctx and module == "quantpits.scripts.order_gen"
    assert args[args.index("--candidate-output")+1] == step["candidate_output"]
    assert args[args.index("--ensemble-record-file")+1] == str(project.signal_dir(req["cycle"],"shared")/"ensemble.json")
    assert args[args.index("--record-file")+1] == str(project.signal_dir(req["cycle"],"shared")/"predictions.json")
    default = request(project)
    NativeSteps(project,default,Capture(),project.cycle_dir(default["cycle"])).execute(order_step(project,default))
    assert "--candidate-output" not in calls[-1][2]


def completed_order(project, *, empty=False):
    import pandas as pd
    from quantpits.order.command import OrderRunOptions, load_order_run_config, prepare_order_run
    from quantpits.order.execution import OrderExecutionHooks, UniverseSnapshot, load_resolved_prediction
    from quantpits.order.persistence import persist_order_artifacts
    from quantpits.order.service import OrderGenerationService
    from quantpits.utils import strategy
    from quantpits.utils.workspace import fingerprint_file
    req = request(project,{"order_candidates":["live"]})
    step = order_step(project,req); ctx=project.members["live"].ctx
    directory=project.signal_dir(req["cycle"],"shared"); directory.mkdir(parents=True)
    signal=directory/"signal.csv"
    signal.write_text("datetime,instrument,score\n2024-05-17,SH600001,2\n2024-05-17,SH600002,1\n")
    save_json(directory/"ensemble.json",dict(default_combo="chosen",combos={"chosen":"fixed-ensemble"},
        predictions={"chosen":dict(path=str(signal),sha256=fingerprint_file(signal))}))
    options=OrderRunOptions(anchor=req["anchor"],trade_date=req["trade_date"],combo="chosen",
        ensemble_record_file=str(directory/"ensemble.json"),run_id=step["run_id"],
        output_dir=str(Path(step["candidate_output"]).parent),candidate_output=step["candidate_output"])
    config=load_order_run_config(ctx,options)
    if empty:
        config=replace(config,merged_config=dict(config.merged_config,current_holding=[dict(instrument="SH600001",value=100)]))
    prepared=prepare_order_run(ctx=ctx,options=options,cli_args=(),run_config=config)
    hooks=OrderExecutionHooks(init_qlib=lambda:None,get_anchor_date=lambda:req["anchor"],
        get_next_trade_date=lambda _:req["trade_date"],validate_dates=lambda *a:None,
        load_predictions=load_resolved_prediction,
        get_price_data=lambda *a,**k:pd.DataFrame(dict(current_close=[10.,10.],possible_min=[9.,9.],possible_max=[11.,11.]),index=pd.Index(["SH600001","SH600002"],name="instrument")),
        create_order_generator=strategy.create_order_generator,get_strategy_params=strategy.get_strategy_params,
        build_model_opinions=lambda _:None,persist_artifacts=persist_order_artifacts,
        resolve_universe=lambda m,a:UniverseSnapshot.observe(m,a,["SH600001","SH600002"]))
    summary=OrderGenerationService(hooks).execute(prepared)
    return NativeSteps(project,req,None,project.cycle_dir(req["cycle"])),step,ctx.root/step["candidate_output"],ctx.root/summary.manifest_path


@pytest.mark.parametrize("empty",[False,True])
def test_observe_uses_actual_service_manifest_and_candidate_including_empty(project,empty):
    native,step,path,manifest=completed_order(project,empty=empty)
    proof=native.observe(step)
    assert proof["candidates"]["count"] == (0 if empty else 1)
    assert proof["candidates"]["path"] == str(path)
    assert str(path) in [ref["path"] for ref in proof["files"]]
    assert native.observe(step,completed=True) == proof


@pytest.mark.parametrize("change",["missing", "kind", "digest", "workspace", "anchor", "trade_date", "source", "signal", "ledger", "output", "count", "target"])
def test_invalid_native_candidate_cannot_be_adopted(project,change):
    native,step,path,manifest_path=completed_order(project)
    candidate=json.loads(path.read_text()); manifest=json.loads(manifest_path.read_text())
    if change == "missing": path.unlink()
    elif change == "kind": candidate["kind"] = "other"
    elif change == "digest": candidate["digest"] = "wrong"
    elif change == "workspace": candidate["workspace"] = str(project.members["paper_a"].ctx.root)
    elif change == "anchor": candidate["anchor"] = "2024-05-16"
    elif change == "trade_date": candidate["trade_date"] = "2024-05-22"
    elif change == "source": candidate["source"]["record_id"] = "wrong-source"
    elif change == "signal": candidate["source"]["signal_digest"] = "0"*64
    elif change == "ledger": del manifest["records"]["candidates"]
    elif change == "output": manifest["outputs"] = [r for r in manifest["outputs"] if r["path"] != step["candidate_output"]]
    elif change == "count": manifest["records"]["candidates"]["count"] = True
    elif change == "target": manifest["records"]["target_buy_count"] = 0
    if change not in ("missing",):
        if change not in ("digest",):
            candidate["digest"]=fingerprint_value({k:v for k,v in candidate.items() if k != "digest"})
            if "candidates" in manifest["records"]:
                manifest["records"]["candidates"]["digest"]=candidate["digest"]
        save_json(path,candidate)
    save_json(manifest_path,manifest)
    with pytest.raises((ValueError,OSError)):
        native.observe(step)
