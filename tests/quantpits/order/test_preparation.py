"""Saved candidates and explicit cash-open observations are reusable inputs."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quantpits.order.command import OrderRunConfig, OrderRunOptions, prepare_order_run
from quantpits.order.execution import OrderExecutionHooks, UniverseSnapshot, load_resolved_prediction
from quantpits.order.persistence import persist_order_artifacts
from quantpits.order.service import OrderGenerationService
from quantpits.utils.strategy import TopkDropoutOrderGenerator
from quantpits.utils.workspace import WorkspaceContext, fingerprint_file

ANCHOR, DAY = "2026-10-23", "2026-10-27"
BUYS = ["SH600001", "SH600002", "SH600003", "SH600004"]
HELD = "SH600099"
FEES = {"base_fee": "5", "base_rate": "0", "sell_tax_rate": "0.0005",
        "transfer_rates": {"SH": "0.00001", "SZ": "0"}, "lot_size": 100}
RULES = {"date": DAY, "source": "synthetic dated rules", "boards": {
    "SH_MAIN": {"normal_limit": True, "limit_rate": ".10"}}, "instruments": {}}


def scenario(root, *, candidate_output=None, cash=1050, dry_run=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(exist_ok=True)
    signal = root / "signal.csv"
    pd.DataFrame({"datetime": [ANCHOR] * 5, "instrument": BUYS + [HELD],
                  "score": [5., 4., 3., 2., 1.]}).to_csv(signal, index=False)
    config = OrderRunConfig(
        merged_config={"market": "synthetic", "current_cash": cash,
                       "current_holding": [{"instrument": HELD, "value": 100}]},
        cashflow_config={}, strategy_config={"strategy": {"name": "topk_dropout", "params": {
            "topk": 2, "n_drop": 1, "buy_suggestion_factor": 2}}}, ensemble_config={}, train_records={},
        ensemble_records={"default_combo": "saved", "combos": {"saved": "saved-signal"},
                          "predictions": {"saved": {"path": str(signal), "sha256": fingerprint_file(signal)}}},
    )
    values = dict(output_dir="output/original", run_id="order36-baseline", anchor=ANCHOR,
                  trade_date=DAY, ensemble_record_file="config/saved.json", dry_run=dry_run)
    if candidate_output is not None:
        values["candidate_output"] = candidate_output
    prepared = prepare_order_run(ctx=WorkspaceContext.from_root(root), options=OrderRunOptions(**values),
                                 cli_args=(), run_config=config)
    hooks = OrderExecutionHooks(
        init_qlib=lambda: None, get_anchor_date=lambda: ANCHOR, get_next_trade_date=lambda _: DAY,
        validate_dates=lambda *a: None, load_predictions=load_resolved_prediction,
        get_price_data=lambda *a, **k: pd.DataFrame({"current_close": [10.] * 5,
            "possible_min": [9.] * 5, "possible_max": [11.] * 5}, index=pd.Index(BUYS + [HELD], name="instrument")),
        create_order_generator=lambda cfg: TopkDropoutOrderGenerator(**cfg["strategy"]["params"]),
        get_strategy_params=lambda cfg: cfg["strategy"]["params"], build_model_opinions=lambda _: None,
        persist_artifacts=persist_order_artifacts,
        resolve_universe=lambda m, a: UniverseSnapshot.observe(m, a, BUYS + [HELD]),
    )
    return prepared, hooks


def legacy_observation(root, cash):
    from quantpits.runtime import command_plan_to_public_dict, fingerprint_command_plan
    prepared, hooks = scenario(root, cash=cash)
    summary = OrderGenerationService(hooks).execute(prepared)
    plan = command_plan_to_public_dict(prepared.plan)
    plan["metadata"].pop("plan_fingerprint", None)
    normalized = json.loads(json.dumps(plan).replace(str(root), "<workspace>"))
    return {"plan": normalized, "csv": {Path(path).name: (root / path).read_text()
            for path in summary.actual_outputs},
            "normalized_plan_fingerprint": fingerprint_command_plan(replace(prepared.plan,
                workspace="<workspace>", metadata=normalized["metadata"]))}


@pytest.mark.parametrize("cash", [1050, 10000])
def test_default_plan_and_csv_match_prechange_baseline(tmp_path, cash):
    baseline = json.loads((Path(__file__).parent / "fixtures/preparation_legacy_baseline.json").read_text())
    assert legacy_observation(tmp_path / "account", cash) == baseline[str(cash)]


def exported(tmp_path, **kwargs):
    prepared, hooks = scenario(tmp_path / "account", candidate_output="output/candidates.json", **kwargs)
    summary = OrderGenerationService(hooks).execute(prepared)
    path = prepared.ctx.output_path("candidates.json")
    return prepared, hooks, summary, json.loads(path.read_text())


def observation(*, cash="1050.00", holdings=None, quotes=None, **kwargs):
    result = dict(schema_version=1, kind="cash_open_observation", trade_date=DAY, price_basis="CASH_OPEN",
        cash=cash, holdings=[dict(instrument=HELD, quantity=100)] if holdings is None else holdings,
        quotes={instrument: dict(open="9.60", previous_close="10", factor="1", previous_factor="1")
                for instrument in BUYS} if quotes is None else quotes,
        market_rules=copy.deepcopy(RULES), execution=copy.deepcopy(FEES))
    result.update(kwargs)
    return result


def rehash(candidates):
    from quantpits.utils.workspace import fingerprint_value
    candidates["digest"] = fingerprint_value({k:v for k,v in candidates.items() if k != "digest"})
    return candidates


def test_export_preserves_zero_quantity_full_order_and_legacy_csv(tmp_path):
    from quantpits.order.preparation import prepare_orders
    prepared, _, summary, candidates = exported(tmp_path)
    assert [row["instrument"] for row in candidates["buys"]] == BUYS
    assert [row["role"] for row in candidates["buys"]] == ["PRIMARY", "PRIMARY", "BACKUP", "BACKUP"]
    assert [row["suggested_quantity"] for row in candidates["buys"]] == [0] * 4
    assert summary.buy_file is None
    manifest = json.loads((prepared.ctx.root / summary.manifest_path).read_text())
    assert manifest["records"]["candidates"] == dict(path="output/candidates.json", digest=candidates["digest"], count=4)
    assert "output/candidates.json" in summary.actual_outputs
    result = prepare_orders(candidates, observation())
    assert result["decisions"][0]["quantity"] == 100
    assert result["decisions"][0]["estimated_cost"] == "965.01"
    assert result["decisions"][0]["fees"] == dict(base="5.00", tax="0.00", transfer="0.01", total="5.01")
    assert result["pending_sells"][0]["instrument"] == HELD
    assert result["slots"] == 1 and result["slot_budget"] == "1050.00"
    assert result["estimated_remaining_cash"] == "84.99"
    assert "FILLED" not in json.dumps(result) and "after" not in result


def test_missing_primary_uses_original_backup_and_does_not_simulate_sells(tmp_path):
    from quantpits.order.preparation import prepare_orders
    _, _, _, candidates = exported(tmp_path)
    obs = observation()
    for instrument in BUYS[:2]:
        obs["quotes"][instrument].pop("open")
    result = prepare_orders(candidates, obs)
    assert [row["reason"] for row in result["decisions"]] == ["MISSING_OPEN", "MISSING_OPEN", None, "NO_SLOT"]
    assert result["decisions"][2]["quantity"] == 100 and result["decisions"][2]["role"] == "BACKUP"
    assert result["current_cash"] == "1050.00" and result["slots"] == 1
    # An independently updated observation supplies actual available cash. No
    # original sell proceeds or sell fees are added/subtracted a second time.
    updated = prepare_orders(candidates, observation(cash="2100.00", holdings=[]))
    assert updated["slots"] == 2 and updated["slot_budget"] == "1050.00"
    assert [r["quantity"] for r in updated["decisions"]] == [100,100,0,0]
    assert updated["estimated_fees"] == "10.02" and not updated["pending_sells"]


@pytest.mark.parametrize("cash,reserve,quantity", [("960.00","0",0), ("965.00","0",0),
    ("965.01","0",100), ("1930.00","0",200), ("1050.00","100",0), ("-3583.10","0",0), ("50.00","100",0)])
def test_fee_reserve_and_negative_cash_boundaries(tmp_path, cash, reserve, quantity):
    from quantpits.order.preparation import prepare_orders
    _, _, _, candidates = exported(tmp_path)
    obs = observation(cash=cash)
    obs["execution"]["cash_reserve"] = reserve
    result = prepare_orders(candidates, obs)
    assert result["decisions"][0]["quantity"] == quantity
    assert result["current_cash"] == cash
    if quantity == 0:
        assert all(row["reason"] == "INSUFFICIENT_SLOT_BUDGET" for row in result["decisions"])
        assert result["estimated_remaining_cash"] == cash


@pytest.mark.parametrize("condition,reason", [("suspended","SUSPENDED"), ("limit","OPEN_AT_LIMIT"),
    ("rule_gap","UNKNOWN_LIMIT_RULE"), ("factor","FACTOR_CHANGED"), ("held","ALREADY_HELD"),
    ("bad_interval","INVALID_MARKET_OBSERVATION")])
def test_individual_conditions_keep_all_members_and_continue(tmp_path, condition, reason):
    from quantpits.order.preparation import prepare_orders
    _, _, _, candidates = exported(tmp_path)
    obs = observation()
    if condition == "suspended":
        obs["market_rules"]["instruments"][BUYS[0]] = {"suspended": True}
    elif condition == "limit":
        obs["quotes"][BUYS[0]]["open"] = "11"
    elif condition == "rule_gap":
        obs["market_rules"]["boards"] = {}
        for instrument in BUYS[1:]:
            obs["market_rules"]["instruments"][instrument] = RULES["boards"]["SH_MAIN"]
    elif condition == "factor":
        obs["quotes"][BUYS[0]]["factor"] = "2"
    elif condition == "held":
        obs["holdings"] = [dict(instrument=BUYS[0],quantity=100)]
    elif condition == "bad_interval":
        obs["market_rules"]["instruments"][BUYS[0]] = dict(limit_up="8",limit_down="12")
    result = prepare_orders(candidates, obs)
    assert result["decisions"][0]["reason"].startswith(reason)
    assert result["decisions"][1]["status"] == "PREPARED"
    assert [row["instrument"] for row in result["decisions"]] == BUYS
    changed = copy.deepcopy(obs)
    for quote in changed["quotes"].values():
        quote.update(high="999", low=".01", close="777")
    assert prepare_orders(candidates, changed)["decisions"] == result["decisions"]


@pytest.mark.parametrize("cash", ["1050", "965.01", "965.00", "2200"])
def test_quantities_equal_paper_for_same_cash_open_conditions(tmp_path, cash):
    from quantpits.order.preparation import prepare_orders
    from quantpits.research.paper_accounting import settle_plan, amount
    _, _, _, candidates = exported(tmp_path)
    obs = observation(cash=cash)
    obs["quotes"].pop(BUYS[0])
    result = prepare_orders(candidates, obs)
    prior = dict(portfolio_id="synthetic", as_of_date=ANCHOR, cash=cash,
        positions=[dict(instrument=HELD,quantity=100,book_cost="1000")])
    plan = dict(cycle_id="probe", trade_date=DAY, execution=FEES, sells=[], buys=candidates["buys"],
        strategy=dict(topk=2), target_buys=candidates["target_buy_count"], universe=BUYS)
    paper = settle_plan(plan, prior, obs["quotes"], obs["market_rules"])
    assert [r["quantity"] for r in result["decisions"]] == [r["quantity"] for r in paper["decisions"]]
    assert result["estimated_remaining_cash"] == amount(paper["after"]["cash"])
    assert result["estimated_fees"] == paper["total_fees"]


@pytest.mark.parametrize("case", ["date","rules_date","quote_date","basis","duplicate_holdings","cash_bool","fractional_cash","fee","broken_candidate","duplicate_candidate"])
def test_structurally_invalid_inputs_fail(tmp_path, case):
    from quantpits.order.preparation import prepare_orders
    _, _, _, candidates = exported(tmp_path)
    obs = observation()
    if case == "date": obs["trade_date"] = "2026-10-28"
    elif case == "rules_date": obs["market_rules"]["date"] = "2026-10-28"
    elif case == "quote_date": obs["quotes"][BUYS[0]]["date"] = "2026-10-28"
    elif case == "basis": obs["price_basis"] = "INTRADAY"
    elif case == "duplicate_holdings": obs["holdings"] *= 2
    elif case == "cash_bool": obs["cash"] = True
    elif case == "fractional_cash": obs["cash"] = "100.001"
    elif case == "fee": obs["execution"]["lot_size"] = 200
    elif case == "broken_candidate": candidates["buys"][0]["suggested_quantity"] = 1
    elif case == "duplicate_candidate":
        candidates["buys"].append(copy.deepcopy(candidates["buys"][-1])); rehash(candidates)
    with pytest.raises(ValueError): prepare_orders(candidates, obs)


@pytest.mark.parametrize("value", ["config/candidates.json","../escape.json","output/buy_suggestion_fake.json"])
def test_export_paths_reject_non_output_locations(tmp_path, value):
    with pytest.raises(ValueError): scenario(tmp_path / "account", candidate_output=value)


def test_dry_run_and_changed_parent_symlink_do_not_write_candidate(tmp_path):
    prepared, hooks = scenario(tmp_path / "account", candidate_output="output/candidates.json", dry_run=True)
    before = {str(p):p.read_bytes() for p in prepared.ctx.root.rglob("*") if p.is_file()}
    OrderGenerationService(hooks).execute(prepared)
    assert {str(p):p.read_bytes() for p in prepared.ctx.root.rglob("*") if p.is_file()} == before
    outside = tmp_path / "outside"; outside.mkdir()
    prepared.ctx.output_dir.mkdir(exist_ok=True)
    (prepared.ctx.output_dir / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError): scenario(prepared.ctx.root,candidate_output="output/escape/file.json")


def test_candidate_conflict_partial_outputs_and_no_op_write_are_truthful(tmp_path, monkeypatch):
    from quantpits.order.persistence import OrderPersistenceError, write_derived_json
    prepared, hooks, summary, candidates = exported(tmp_path)
    before = prepared.ctx.output_path("candidates.json").read_bytes()
    assert write_derived_json(prepared.ctx,"output/candidates.json",candidates) == "reused"
    changed = copy.deepcopy(candidates); changed["source"]["description"] = "changed"; rehash(changed)
    with pytest.raises(FileExistsError): write_derived_json(prepared.ctx,"output/candidates.json",changed)
    assert prepared.ctx.output_path("candidates.json").read_bytes() == before
    # A failed extra output retains the original CSV in the failure manifest.
    other = replace(prepared, plan=replace(prepared.plan,run_id="failed-candidate"))
    prepared.ctx.output_path("candidates.json").write_text("{}")
    with pytest.raises(OrderPersistenceError): OrderGenerationService(hooks).execute(other)
    manifest = json.loads(prepared.ctx.output_path("manifests/order_gen/failed-candidate.json").read_text())
    assert manifest["status"] == "failed"
    assert [r["path"] for r in manifest["outputs"]] == [summary.sell_file,"output/manifests/order_gen/failed-candidate.json"]
    monkeypatch.setattr("quantpits.order.persistence.os.link",lambda *a,**k: None)
    with pytest.raises(OSError,match="postcondition"):
        write_derived_json(prepared.ctx,"output/no_op.json",candidates)


@pytest.mark.parametrize("control", [KeyboardInterrupt,SystemExit])
def test_member_and_write_controls_propagate(tmp_path, monkeypatch, control):
    from quantpits.order.preparation import prepare_orders
    _, _, _, candidates = exported(tmp_path)
    def interrupt(*a,**k): raise control()
    monkeypatch.setattr("quantpits.order.preparation.trade_check",interrupt)
    with pytest.raises(control): prepare_orders(candidates,observation())
    monkeypatch.setattr("quantpits.order.persistence.os.link",interrupt)
    from quantpits.order.persistence import write_derived_json
    with pytest.raises(control): write_derived_json(WorkspaceContext.from_root(tmp_path / "account"),"output/interrupted.json",candidates)


def test_cli_preview_write_conflict_and_no_backend_imports(tmp_path):
    import os
    import subprocess
    import sys
    prepared, _, _, candidates = exported(tmp_path)
    path = prepared.ctx.root / "observation.json"; path.write_text(json.dumps(observation()))
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
    environment.pop("QLIB_WORKSPACE_DIR",None); environment.pop("MLFLOW_TRACKING_URI",None)
    arguments=[sys.executable,"-m","quantpits.scripts.prepare_orders","--workspace",str(prepared.ctx.root),
               "--candidates","output/candidates.json","--observation","observation.json"]
    before={str(p):p.read_bytes() for p in prepared.ctx.root.rglob("*") if p.is_file()}
    preview=subprocess.run(arguments,env=environment,capture_output=True,text=True)
    assert preview.returncode == 0, preview.stderr
    assert json.loads(preview.stdout)["decisions"][0]["quantity"] == 100
    assert {str(p):p.read_bytes() for p in prepared.ctx.root.rglob("*") if p.is_file()} == before
    for expected in ("created","reused"):
        result=subprocess.run(arguments+["--output-dir","output/prepared"],env=environment,capture_output=True,text=True)
        assert result.returncode == 0,result.stderr
        assert json.loads(result.stdout)["output_status"] == expected
    path.write_text(json.dumps(observation(cash="1000.00")))
    result=subprocess.run(arguments+["--output-dir","output/prepared"],env=environment,capture_output=True,text=True)
    assert result.returncode == 1 and "already exists" in result.stderr
    result=subprocess.run([sys.executable,"-c","import sys; import quantpits.scripts.prepare_orders; "
        "assert 'qlib' not in sys.modules; assert 'mlflow' not in sys.modules; "
        "assert 'quantpits.utils.env' not in sys.modules"],env=environment,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr


def test_fixed_slot_budget_is_not_redistributed_and_all_estimates_stay_frozen(tmp_path):
    from quantpits.order.preparation import prepare_orders
    _, _, _, candidates = exported(tmp_path)
    saved = copy.deepcopy(candidates)
    result = prepare_orders(candidates,observation(cash="2200.00",holdings=[]))
    assert result["slots"] == 2 and result["slot_budget"] == "1100.00"
    assert [row["quantity"] for row in result["decisions"]] == [100,100,0,0]
    assert result["estimated_remaining_cash"] == "269.98"
    assert candidates == saved


def test_parent_escape_after_prepare_and_cli_input_escape_are_rejected(tmp_path, capsys):
    from quantpits.order.persistence import OrderPersistenceError
    from quantpits.scripts.prepare_orders import main
    prepared, hooks = scenario(tmp_path / "account",candidate_output="output/later/candidates.json")
    outside = tmp_path / "outside"; outside.mkdir()
    prepared.ctx.output_dir.mkdir(exist_ok=True)
    (prepared.ctx.output_dir / "later").symlink_to(outside,target_is_directory=True)
    with pytest.raises(OrderPersistenceError): OrderGenerationService(hooks).execute(prepared)
    assert not list(outside.iterdir())
    outside_input=outside / "observation.json"; outside_input.write_text(json.dumps(observation()))
    (prepared.ctx.root / "observation.json").symlink_to(outside_input)
    # Use a fresh candidate path; the original escape remains forbidden.
    prepared, hooks, _, _ = exported(tmp_path)
    assert main(["--workspace",str(prepared.ctx.root),"--candidates","output/candidates.json",
                 "--observation","observation.json"]) == 1
    assert "inside workspace" in capsys.readouterr().err


def test_output_root_symlink_into_config_is_not_an_output_boundary(tmp_path):
    from quantpits.order.candidates import output_path
    from quantpits.order.persistence import write_derived_json
    root=tmp_path / "account"; (root / "config").mkdir(parents=True)
    (root / "output").symlink_to(root / "config",target_is_directory=True)
    ctx=WorkspaceContext.from_root(root)
    for write in (lambda:output_path(ctx,"output/candidates.json"),
                  lambda:write_derived_json(ctx,"output/candidates.json",{})):
        with pytest.raises(ValueError,match="workspace/output"): write()
    assert not list((root / "config").iterdir())


@pytest.mark.parametrize("which", ["candidates", "observation"])
def test_cli_rejects_nonobject_json_without_traceback_or_output(tmp_path, capsys, which):
    from quantpits.scripts.prepare_orders import main
    prepared, _, _, _ = exported(tmp_path)
    root = prepared.ctx.root
    (root / "observation.json").write_text(json.dumps(observation()))
    path = root / ("output/candidates.json" if which == "candidates" else "observation.json")
    path.write_text("[]")
    assert main(["--workspace",str(root),"--candidates","output/candidates.json",
                 "--observation","observation.json","--output-dir","output/invalid"]) == 1
    error = capsys.readouterr().err
    assert "JSON root must be an object" in error and "Traceback" not in error
    assert not (root / "output/invalid").exists()
