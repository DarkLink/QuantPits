"""Synthetic local provider/accounts, real planner, accounting and journal I/O."""
import json
import struct
from pathlib import Path

import pandas as pd
import pytest

from quantpits.research.paper_accounting import (
    PaperError, amount, decimal, fees, settle_plan, state, trade_check, valuation,
)
from quantpits.research.paper_market import PaperMarket
from quantpits.research.paper_workflow import PaperWorkflow
from quantpits.scripts.paper_trade import main
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value


DATES = ["2024-04-05", "2024-04-08", "2024-04-12", "2024-04-15", "2024-04-19"]
BUY = ["SH600%03d" % i for i in range(1, 10)]
HELD = ["SH600090", "SZ000090", "SZ300090"]
FEES = {"base_fee": "5", "base_rate": "0", "sell_tax_rate": "0.0005",
        "transfer_rates": {"SH": "0.00001", "SZ": "0"}, "lot_size": 100}
RULES = {"source": "synthetic normal-limit declarations", "boards": {
    "SH_MAIN": {"normal_limit": True, "limit_rate": "0.10"},
    "SZ_MAIN": {"normal_limit": True, "limit_rate": "0.10"},
    "CHINEXT": {"normal_limit": True, "limit_rate": "0.20"}}, "instruments": {}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def feature(provider, instrument, field, values):
    path = provider / "features" / instrument.lower() / (field + ".day.bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<%sf" % (len(values) + 1), 0, *values))


def calendar(provider, count):
    (provider / "calendars").mkdir(parents=True, exist_ok=True)
    (provider / "calendars/day.txt").write_text("\n".join(DATES[:count]) + "\n")
    (provider / "calendars/day_future.txt").write_text("\n".join(DATES) + "\n")


def make_workspace(tmp_path, name="account_a", topk=3, pool="custom_pool", fee=None, shared=None):
    root = tmp_path / name
    provider = shared or tmp_path / "provider"
    root.mkdir()
    calendar(provider, 1)
    (provider / "instruments").mkdir(parents=True, exist_ok=True)
    (provider / "instruments" / (pool + ".txt")).write_text(
        "\n".join(i + "\t2024-01-01\t2024-12-31" for i in BUY + HELD) + "\n")
    for instrument in BUY + HELD:
        for field in ("open", "close"):
            feature(provider, instrument, field, [10] * len(DATES))
        feature(provider, instrument, "factor", [1] * len(DATES))
    signal = tmp_path / "signal.pkl"
    if not signal.exists():
        index = pd.MultiIndex.from_tuples([(pd.Timestamp(d), i) for d in DATES for i in BUY + HELD],
                                          names=["datetime", "instrument"])
        frame = pd.DataFrame({"score": list(range(12, 0, -1)) * len(DATES)}, index=index)
        frame.to_pickle(signal)
    write_json(root / "config/model_config.json", {"market": pool})
    (root / "config/strategy_config.yaml").write_text(
        "strategy:\n  name: topk_dropout\n  params:\n    topk: %s\n    n_drop: 2\n    buy_suggestion_factor: 3\n" % topk)
    write_json(root / "config/paper_config.json", {
        "account_id": name, "execution": fee or FEES, "market_rules": RULES,
        "signal_source": {"workspace": str(tmp_path), "prediction": str(signal)}})
    source = tmp_path / (name + "_actual.json")
    write_json(source, {"current_date": DATES[0], "current_cash": 200,
                        "current_holding": [{"instrument": i, "value": 200, "amount": 1500} for i in HELD]})
    ctx = WorkspaceContext.from_root(root, qlib_data_dir=provider)
    return PaperWorkflow(ctx), source, provider, signal


def boot_plan(tmp_path, **kwargs):
    workflow, source, provider, signal = make_workspace(tmp_path, **kwargs)
    workflow.bootstrap(source, DATES[0])
    planned = workflow.plan("week", DATES[0], DATES[1])["plan"]
    return workflow, source, provider, signal, planned


def event(event_id, kind, cash, effective=DATES[1], booking=None, **kwargs):
    result = {"event_id": event_id, "kind": kind, "cash": cash, "effective_date": effective,
              "booking_date": booking or effective, "instrument": None, "quantity_delta": 0,
              "cost_delta": "0", "source": "synthetic actual notice"}
    result.update(kwargs)
    return result


def test_real_first_cycle_wait_replacements_next_cycle(tmp_path):
    w, source, provider, signal, plan = boot_plan(tmp_path)
    assert w.bootstrap(source, DATES[0])["status"] == "REUSED"
    before = w.store.path.read_bytes()
    assert plan["generated_at"].endswith("Z")
    assert len(plan["buys"]) == 6
    assert w.settle("week")["status"] == "WAITING_FOR_DATA"
    assert w.store.path.read_bytes() == before
    calendar(provider, 3)
    feature(provider, BUY[0], "open", [10, float("nan"), 10, 10, 10])
    feature(provider, BUY[1], "open", [10, 11, 10, 10, 10])
    feature(provider, BUY[2], "open", [10, 9.5, 10, 10, 10])
    result = w.settle("week")["settlement"]
    decisions = [d for d in result["decisions"] if d["side"] == "BUY"]
    assert [d["reason"] for d in decisions[:2]] == ["MISSING_OPEN", "OPEN_AT_LIMIT"]
    filled = [d for d in decisions if d["status"] == "FILLED"]
    assert [d["instrument"] for d in filled] == BUY[2:4]
    assert filled[0]["quantity"] > filled[0]["suggested_quantity"]
    assert decimal(result["after"]["cash"]) >= 0
    assert len(result["after"]["positions"]) == 3
    assert w.settle("week")["settlement"] == result
    feature(provider, BUY[2], "close", [10, 10, 13, 10, 10])
    valuation_result = w.value(DATES[2])["valuation"]
    assert valuation_result["status"] == "COMPLETE"
    assert w.report(DATES[2], reference_config=source)["actual_reference"]["status"] == "PENDING_UPDATE"
    assert w.store.read()["records"][0]["result"] == result
    # Next source is a new artifact, while balance continues from the same journal.
    new_signal = tmp_path / "next.pkl"
    pd.read_pickle(signal).to_pickle(new_signal)
    next_plan = w.plan("next", DATES[2], DATES[3], source_override={"prediction": str(new_signal)})["plan"]
    assert next_plan["prior"] == result["after"]
    assert next_plan["source"]["path"] == str(new_signal)
    calendar(provider, 5)
    assert w.settle("next")["status"] == "SETTLED"
    assert w.value(DATES[4])["status"] == "VALUED"
    assert w.store.read()["bootstrap"]["initial_nav"] == "6200.00"


def test_failed_sell_does_not_release_cash_or_slot(tmp_path):
    w, _, provider, _, plan = boot_plan(tmp_path)
    calendar(provider, 2)
    first_sell = plan["sells"][0]["instrument"]
    feature(provider, first_sell, "open", [10, float("nan"), 10, 10, 10])
    result = w.settle("week")["settlement"]
    assert result["slots"] == 1
    assert [d["status"] for d in result["decisions"] if d["side"] == "SELL"] == ["SKIPPED", "FILLED"]
    assert first_sell in [p["instrument"] for p in result["after"]["positions"]]
    assert len([d for d in result["decisions"] if d["side"] == "BUY" and d["status"] == "FILLED"]) == 1
    assert len(result["after"]["positions"]) == 3


def test_both_failed_sells_and_no_fee(tmp_path):
    w, _, provider, _, plan = boot_plan(tmp_path)
    calendar(provider, 2)
    for row in plan["sells"]:
        feature(provider, row["instrument"], "open", [10, float("nan"), 10, 10, 10])
    result = w.settle("week")["settlement"]
    assert result["slots"] == 0
    assert result["total_fees"] == "0.00"
    assert result["after"]["positions"] == plan["prior"]["positions"]
    assert result["after"]["cash"] == plan["prior"]["cash"]


def test_candidates_with_zero_conservative_quantity_are_preserved(tmp_path):
    w, source, provider, _ = make_workspace(tmp_path, topk=1)
    write_json(source, {"current_date": DATES[0], "current_cash": 1050, "current_holding": []})
    w.bootstrap(source, DATES[0])
    plan = w.plan("small", DATES[0], DATES[1])["plan"]
    assert plan["buys"][0]["suggested_quantity"] == 0
    calendar(provider, 2)
    feature(provider, BUY[0], "open", [10, 9.6, 10, 10, 10])
    result = w.settle("small")["settlement"]
    assert result["decisions"][0]["quantity"] == 100
    assert result["decisions"][0]["suggested_quantity"] == 0


def test_two_accounts_shared_signal_different_configs_alternating(tmp_path, monkeypatch):
    a, sa, provider, signal = make_workspace(tmp_path, "a")
    bfee = dict(FEES, base_fee="9")
    b, sb, _, _ = make_workspace(tmp_path, "b", topk=2, pool="other_pool", fee=bfee, shared=provider)
    a.bootstrap(sa, DATES[0]); b.bootstrap(sb, DATES[0])
    pa = a.plan("same", DATES[0], DATES[1])["plan"]
    pb = b.plan("same", DATES[0], DATES[1])["plan"]
    assert pa["market"] != pb["market"]
    assert pa["source"]["sha256"] == pb["source"]["sha256"]
    before_b = b.store.path.read_bytes()
    with pytest.raises(PaperError):
        a.plan("same", DATES[0], DATES[3])
    assert b.store.path.read_bytes() == before_b
    calendar(provider, 3)
    from quantpits.research import paper_store
    original = paper_store.atomic_write_json_bytes
    def interrupt_a(path, payload):
        if path == a.store.path:
            raise KeyboardInterrupt()
        original(path, payload)
    monkeypatch.setattr(paper_store, "atomic_write_json_bytes", interrupt_a)
    with pytest.raises(KeyboardInterrupt):
        a.settle("same")
    assert b.store.path.read_bytes() == before_b
    br = b.settle("same")["settlement"]
    monkeypatch.setattr(paper_store, "atomic_write_json_bytes", original)
    ar = a.settle("same")["settlement"]
    assert len(ar["after"]["positions"]) == 3
    assert len(br["after"]["positions"]) == 2
    assert ar["decisions"][0]["fees"]["base"] == "5.00"
    assert br["decisions"][0]["fees"]["base"] == "9.00"
    assert a.settle("same")["settlement"] == ar
    assert a.store.path != b.store.path
    assert a.store.read()["account_id"] == "a"
    assert b.store.read()["account_id"] == "b"


def test_explicit_events_timing_dedup_and_pnl(tmp_path):
    w, _, provider, _, _ = boot_plan(tmp_path)
    income = event("income", "DIVIDEND", "11.20")
    deposit = event("deposit", "EXTERNAL", "500")
    later = event("later", "EXTERNAL", "700", DATES[2])
    tax = event("tax", "TAX", "-2.30", DATES[2])
    for row in (income, deposit, later, tax):
        w.add_event(row)
    assert w.add_event(income)["status"] == "REUSED"
    with pytest.raises(PaperError):
        w.add_event(dict(income, cash="12"))
    calendar(provider, 3)
    result = w.settle("week")["settlement"]
    assert result["external_cashflow"] == "500.00"
    assert result["clean_cash"] == "4699.20"
    assert w.settle("week")["settlement"] == result
    value = w.value(DATES[2])["valuation"]
    assert value["net_external_cashflow"] == "1200.00"
    assert decimal(value["pnl_excluding_external"]) == decimal(value["nav"]) - 7400
    assert value["unit_nav"] is None
    snapshot = w.store.path.read_bytes()
    assert w.value(DATES[2])["status"] == "REUSED"
    assert w.store.path.read_bytes() == snapshot


def test_stale_partial_and_no_market_wait(tmp_path):
    w, _, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 3)
    settled = w.settle("week")["settlement"]
    for p in settled["after"]["positions"]:
        feature(provider, p["instrument"], "close", [10, 10, float("nan"), 10, 10])
    # Original holding has a bootstrap price; newly bought stocks have no reliable saved close.
    result = w.value(DATES[2])["valuation"]
    assert result["status"] == "PARTIAL"
    assert result["nav"] is None
    assert any(r["stale"] for r in result["positions"])
    assert len(result["missing"]) == 2
    before = w.store.path.read_bytes()
    assert w.value(DATES[4])["status"] == "WAITING_FOR_DATA"
    assert w.store.path.read_bytes() == before
    # Fill later data at the same date, appending a corrected observation.
    for p in settled["after"]["positions"]:
        feature(provider, p["instrument"], "close", [10, 10, 12, 10, 10])
    assert w.value(DATES[2])["valuation"]["status"] == "COMPLETE"


def test_h_l_and_execution_c_do_not_affect_settlement(tmp_path):
    w, _, provider, _, plan = boot_plan(tmp_path)
    calendar(provider, 3)
    first = w.settle("week")["settlement"]
    for i in BUY + HELD:
        feature(provider, i, "high", [1e20] * 5)
        feature(provider, i, "low", [float("nan")] * 5)
        feature(provider, i, "close", [10, 999, 10, 10, 10])
    market = PaperMarket(provider)
    replay = settle_plan(plan, plan["prior"], market.execution_quotes(DATES[1], BUY + HELD),
                         plan["market_rules"])
    # The observation union includes a few extra unused symbols in this pure replay.
    for key in first:
        if key != "quotes":
            assert replay[key] == first[key]


@pytest.mark.parametrize("failure_stage", ["before", "after"])
def test_interrupt_retry_atomic_settlement(tmp_path, monkeypatch, failure_stage):
    from quantpits.research import paper_store
    w, _, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 2)
    original = paper_store.atomic_write_json_bytes
    before = w.store.path.read_bytes()
    def interrupted(path, payload):
        if failure_stage == "after":
            original(path, payload)
        raise KeyboardInterrupt()
    monkeypatch.setattr(paper_store, "atomic_write_json_bytes", interrupted)
    with pytest.raises(KeyboardInterrupt):
        w.settle("week")
    if failure_stage == "before":
        assert w.store.path.read_bytes() == before
    monkeypatch.setattr(paper_store, "atomic_write_json_bytes", original)
    result = w.settle("week")
    assert result["status"] == ("SETTLED" if failure_stage == "before" else "REUSED")
    assert len(w.store.read()["records"]) == 1


def test_corrupt_record_and_account_rejected_even_with_rehashed_journal(tmp_path):
    w, _, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 2)
    w.settle("week")
    document = json.loads(w.store.path.read_text())
    document["records"][0]["result"]["after"]["cash"] = "123"
    document.pop("digest")
    document["digest"] = fingerprint_value(document)
    write_json(w.store.path, document)
    with pytest.raises(PaperError):
        w.store.read()
    from quantpits.research.paper_store import PaperStore
    with pytest.raises(PaperError):
        PaperStore(w.ctx, "foreign").read()


def test_future_calendar_optional_and_holiday_requires_explicit_date(tmp_path):
    w, _, provider, _, _ = boot_plan(tmp_path)
    (provider / "calendars/day_future.txt").unlink()
    assert w.settle("week")["status"] == "WAITING_FOR_DATA"
    calendar(provider, 3)
    with pytest.raises(PaperError, match="not a trading day"):
        w.store.read()["plans"]["week"]  # read itself does not reschedule
        PaperMarket(provider).check_date("2024-04-07")


def test_cli_commands_use_real_workflow(tmp_path, capsys):
    w, source, provider, _ = make_workspace(tmp_path)
    base = ["--workspace", str(w.ctx.root), "--qlib-data-dir", str(provider)]
    assert main(base + ["bootstrap", "--source-config", str(source), "--date", DATES[0]]) == 0
    assert main(base + ["plan", "--cycle", "cli", "--anchor", DATES[0], "--trade-date", DATES[1]]) == 0
    assert main(base + ["settle", "--cycle", "cli"]) == 2
    calendar(provider, 3)
    assert main(base + ["settle", "--cycle", "cli"]) == 0
    assert main(base + ["value", "--date", DATES[2]]) == 0
    assert main(base + ["report", "--date", DATES[2]]) == 0
    assert '"status": "REPORT"' in capsys.readouterr().out


def test_fee_components_and_rounding_are_per_security_order():
    config = dict(FEES, base_rate="0")
    assert fees(decimal("12345.67"), "SELL", "SH600001", config) == {
        "base": "5.00", "tax": "6.17", "transfer": "0.12", "total": "11.29"}
    assert fees(decimal("12345.67"), "BUY", "SH600001", config)["total"] == "5.12"
    assert fees(decimal("12345.67"), "BUY", "SZ000001", config)["total"] == "5.00"


def test_limits_suspension_factor_change_and_unknown_category():
    rules = dict(RULES, date=DATES[1])
    quote = {"open": "11", "previous_close": "10", "factor": "1", "previous_factor": "1"}
    assert trade_check("SH600001", quote, rules, DATES[1])[0] == "OPEN_AT_LIMIT"
    assert trade_check("SZ300001", quote, rules, DATES[1])[0] is None
    assert trade_check("SH688001", quote, rules, DATES[1])[0] == "UNSUPPORTED_SECURITY"
    assert trade_check("SH600001", dict(quote, open="10", factor="2"), rules, DATES[1])[0] == "FACTOR_CHANGED"
    override = dict(rules, instruments={"SH600001": {"suspended": True}})
    assert trade_check("SH600001", dict(quote, open="10"), override, DATES[1])[0] == "SUSPENDED"
    override = dict(rules, instruments={"SH600001": {"limit_up": "12", "limit_down": "8"}})
    assert trade_check("SH600001", dict(quote, factor="2"), override, DATES[1])[0] is None


def pure_plan(cash="2200", positions=None, buys=None, sells=None, topk=2):
    prior = state({"portfolio_id": "synthetic", "as_of_date": DATES[0], "cash": cash,
                   "positions": positions or []}).to_dict()
    plan = {"cycle_id": "test", "trade_date": DATES[1], "prior": prior, "execution": FEES,
            "strategy": {"topk": topk}, "target_buys": topk, "universe": BUY,
            "buys": [{"instrument": i, "suggested_quantity": 200} for i in (buys or [])],
            "sells": sells or []}
    return plan


def quote(opening, previous=None):
    return {"open": str(opening), "previous_close": str(previous or opening),
            "factor": "1", "previous_factor": "1", "previous_date": DATES[0]}


def test_fee_affordability_fixed_budget_exhaustion_and_reserve():
    plan = pure_plan(buys=BUY[:4])
    quotes = {BUY[0]: quote(12), BUY[1]: quote(10), BUY[2]: quote(12), BUY[3]: quote(10)}
    result = settle_plan(plan, plan["prior"], quotes, dict(RULES, date=DATES[1]))
    assert [r["status"] for r in result["decisions"]] == ["SKIPPED", "FILLED", "SKIPPED", "FILLED"]
    assert {r["budget"] for r in result["decisions"]} == {"1100.00"}
    assert decimal(result["after"]["cash"]) == decimal("189.98")
    small = pure_plan(cash="1000", buys=BUY[:1], topk=1)
    blocked = settle_plan(small, small["prior"], {BUY[0]: quote(10)}, dict(RULES, date=DATES[1]))
    assert blocked["decisions"][0]["reason"] == "INSUFFICIENT_SLOT_BUDGET"
    assert blocked["total_fees"] == "0.00"
    assert blocked["after"]["positions"] == []
    for reserve in ("-1", "1001"):
        bad = dict(small, execution=dict(FEES, cash_reserve=reserve))
        with pytest.raises(PaperError):
            settle_plan(bad, small["prior"], {BUY[0]: quote(10)}, dict(RULES, date=DATES[1]))


def test_odd_lot_full_exit_same_day_buy_lock_and_small_sell_fee_deficit():
    instrument = "SH600090"
    position = {"instrument": instrument, "quantity": 137, "book_cost": "100"}
    sell = {"instrument": instrument, "suggested_quantity": 137}
    plan = pure_plan(cash="0", positions=[position], sells=[sell], topk=1)
    rules = dict(RULES, date=DATES[1])
    result = settle_plan(plan, plan["prior"], {instrument: quote(10)}, rules)
    assert result["decisions"][0]["quantity"] == 137
    assert result["after"]["positions"] == []
    assert result["decisions"][0]["removed_cost"] == "100.00"
    blocked = settle_plan(plan, plan["prior"], {instrument: quote(10)}, rules,
                          buy_locks={instrument: {"date": DATES[1], "quantity": 137}})
    assert blocked["decisions"][0]["reason"] == "INSUFFICIENT_SELLABLE_SHARES"
    tiny = pure_plan(cash="0", positions=[dict(position, quantity=1)],
                     sells=[dict(sell, suggested_quantity=1)], topk=1)
    tiny_rules = dict(rules, instruments={instrument: {"limit_up": "0.04", "limit_down": "0.01"}})
    no_cash = settle_plan(tiny, tiny["prior"], {instrument: quote("0.02")}, tiny_rules)
    assert no_cash["decisions"][0]["reason"] == "INSUFFICIENT_CASH_FOR_SELL_FEES"
    paid = pure_plan(cash="5", positions=[dict(position, quantity=1)],
                     sells=[dict(sell, suggested_quantity=1)], topk=1)
    paid_result = settle_plan(paid, paid["prior"], {instrument: quote("0.02")}, tiny_rules)
    assert paid_result["decisions"][0]["status"] == "FILLED"
    assert paid_result["after"]["cash"] == "0.02"


def test_outside_pool_buy_skipped_and_price_can_reduce_suggested_shares():
    plan = pure_plan(cash="2200", buys=BUY[:2], topk=1)
    plan["universe"] = BUY[1:]
    result = settle_plan(plan, plan["prior"], {BUY[0]: quote(10), BUY[1]: quote("12.5", "12")},
                         dict(RULES, date=DATES[1]))
    assert result["decisions"][0]["reason"] == "OUTSIDE_ANCHOR_UNIVERSE"
    assert result["decisions"][1]["quantity"] == 100
    assert result["decisions"][1]["suggested_quantity"] == 200


def test_custom_pool_forced_exit_and_foreign_predictions(tmp_path):
    w, source, provider, _ = make_workspace(tmp_path)
    (provider / "instruments/custom_pool.txt").write_text(
        "\n".join(i + "\t2024-01-01\t2024-12-31" for i in BUY[1:] + HELD[1:]) + "\n")
    w.bootstrap(source, DATES[0])
    plan = w.plan("exit", DATES[0], DATES[1])["plan"]
    assert HELD[0] in [r["instrument"] for r in plan["sells"]]
    assert BUY[0] not in [r["instrument"] for r in plan["buys"]]
    assert plan["ignored_signal_instruments"] == sorted([BUY[0], HELD[0]])
    calendar(provider, 2)
    result = w.settle("exit")["settlement"]
    assert HELD[0] not in [p["instrument"] for p in result["after"]["positions"]]


def test_whole_missing_open_surface_waits_without_closing_cycle(tmp_path):
    w, _, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 2)
    for instrument in BUY + HELD:
        feature(provider, instrument, "open", [10, float("nan"), 10, 10, 10])
    before = w.store.path.read_bytes()
    assert w.settle("week")["status"] == "WAITING_FOR_DATA"
    assert w.store.path.read_bytes() == before


def test_live_actual_report_at_exact_date_with_explicit_cashflow_reference(tmp_path):
    w, source, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 3)
    w.settle("week")
    w.value(DATES[2])
    write_json(source, {"current_date": DATES[2], "current_cash": 400,
                        "current_holding": [{"instrument": HELD[0], "value": 600, "amount": 4000}]})
    before = source.read_bytes()
    report = w.report(DATES[2], since=DATES[0], reference_config=source, reference_events=[])
    assert report["paper_period"]["since"] == DATES[0]
    assert report["actual_reference"]["nav"] == "6400.00"
    assert report["actual_reference"]["period"]["pnl_excluding_external"] == "200.00"
    assert source.read_bytes() == before


def test_share_and_late_events_are_explicit_and_never_duplicate(tmp_path):
    w, _, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 3)
    w.settle("week")
    w.add_event(event("bonus", "SHARES", "0", DATES[2], instrument=HELD[0], quantity_delta=20))
    result = w.value(DATES[2])["valuation"]
    assert next(p for p in result["positions"] if p["instrument"] == HELD[0])["quantity"] == 220
    with pytest.raises(PaperError, match="late event"):
        w.add_event(event("late", "DIVIDEND", "1", DATES[1]))
    w.add_event(event("late", "DIVIDEND", "1", DATES[1], booking=DATES[2]))
    corrected = w.value(DATES[2])["valuation"]
    assert decimal(corrected["nav"]) == decimal(result["nav"]) + 1
    assert w.value(DATES[2])["status"] == "REUSED"


def test_sqlite_signal_backend_is_explicit_read_only_and_shared(tmp_path):
    from mlflow.tracking import MlflowClient
    w, source, provider, signal = make_workspace(tmp_path)
    signal_root = tmp_path / "signal_source"
    signal_root.mkdir()
    uri = "sqlite:///" + str(signal_root / "runs.db")
    client = MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment("synthetic_signals", artifact_location=(signal_root / "artifacts").as_uri())
    run = client.create_run(experiment)
    pred = tmp_path / "pred.pkl"
    pd.read_pickle(signal).to_pickle(pred)
    client.log_artifact(run.info.run_id, str(pred))
    client.set_terminated(run.info.run_id)
    w.bootstrap(source, DATES[0])
    before = (signal_root / "runs.db").read_bytes()
    result = w.plan("database", DATES[0], DATES[1], source_override={
        "workspace": str(signal_root), "mlflow_uri": uri,
        "recorder_id": run.info.run_id, "experiment_id": experiment})["plan"]
    assert result["source"]["mlflow_uri"] == uri
    assert result["source"]["recorder_id"] == run.info.run_id
    assert (signal_root / "runs.db").read_bytes() == before
    assert not (w.ctx.root / "mlflow.db").exists()
    assert w.store.read()["plans"]["database"] == result
    calendar(provider, 3)
    assert w.settle("database")["status"] == "SETTLED"
    assert w.value(DATES[2])["valuation"]["status"] == "COMPLETE"
    assert (signal_root / "runs.db").read_bytes() == before


def test_missing_backend_is_not_created_and_symlink_writes_are_blocked(tmp_path):
    w, source, provider, _ = make_workspace(tmp_path)
    w.bootstrap(source, DATES[0])
    nonexistent = tmp_path / "absent.db"
    with pytest.raises(PaperError, match="metadata backend"):
        w.plan("missing", DATES[0], DATES[1], source_override={
            "workspace": str(tmp_path), "mlflow_uri": "sqlite:///" + str(nonexistent),
            "recorder_id": "a" * 32, "experiment_id": "1"})
    assert not nonexistent.exists()
    outside = tmp_path / "outside"
    outside.mkdir()
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "data").symlink_to(outside, target_is_directory=True)
    from quantpits.research.paper_store import PaperStore
    store = PaperStore(WorkspaceContext.from_root(fresh), "fresh")
    with pytest.raises(PaperError, match="escapes"):
        with store.locked():
            pass
    assert list(outside.iterdir()) == []


def test_explicit_cancel_allows_owner_to_replace_unverified_holiday(tmp_path):
    w, source, provider, _ = make_workspace(tmp_path)
    w.bootstrap(source, DATES[0])
    (provider / "calendars/day_future.txt").unlink()
    old = w.plan("holiday", DATES[0], "2024-04-07")["plan"]
    assert old["calendar_status"] == "PENDING_CALENDAR"
    calendar(provider, 3)
    with pytest.raises(PaperError, match="not a trading day"):
        w.settle("holiday")
    assert w.cancel("holiday", "owner explicitly selected another date")["status"] == "CANCELLED"
    replacement = w.plan("holiday_revised", DATES[0], DATES[1])["plan"]
    assert replacement["prior"] == old["prior"]
    assert w.store.read()["plans"]["holiday"] == old
    with pytest.raises(PaperError, match="cancelled"):
        w.settle("holiday")
    assert w.settle("holiday_revised")["status"] == "SETTLED"
    with pytest.raises(PaperError, match="unsettled"):
        w.cancel("holiday_revised", "cannot undo completed cash")


def test_invalid_and_missing_scores_do_not_block_eligible_members(tmp_path):
    w, source, _, signal = make_workspace(tmp_path)
    frame = pd.read_pickle(signal)
    frame.loc[(pd.Timestamp(DATES[0]), BUY[0]), "score"] = float("nan")
    frame.to_pickle(signal)
    w.bootstrap(source, DATES[0])
    plan = w.plan("partial", DATES[0], DATES[1])["plan"]
    assert len(plan["ranking"]) == 12
    assert sum(r["scored"] for r in plan["ranking"]) == 11
    assert BUY[0] not in [r["instrument"] for r in plan["buys"]]
    assert w.store.read()["plans"]["partial"] == plan


def test_canonical_post_replace_check_detects_no_op_writer(tmp_path, monkeypatch):
    from quantpits.research import paper_store
    w, _, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 2)
    original = paper_store.atomic_write_json_bytes
    monkeypatch.setattr(paper_store, "atomic_write_json_bytes", lambda path, payload: None)
    with pytest.raises(PaperError, match="uncertain"):
        w.settle("week")
    assert w.store.read()["records"] == []
    monkeypatch.setattr(paper_store, "atomic_write_json_bytes", original)
    assert w.settle("week")["status"] == "SETTLED"


def test_open_evidence_tampering_is_rejected_after_rehash(tmp_path):
    w, _, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 2)
    w.settle("week")
    document = json.loads(w.store.path.read_text())
    result = document["records"][0]["result"]
    key = next(iter(result["quotes"]))
    result["quotes"][key]["open_evidence"]["cash_price"] = "10.1"
    document.pop("digest")
    document["digest"] = fingerprint_value(document)
    write_json(w.store.path, document)
    with pytest.raises(PaperError, match="cash derivation"):
        w.store.read()


def test_net_zero_external_funding_is_not_a_unit_nav_claim(tmp_path):
    w, _, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 3)
    w.add_event(event("in", "EXTERNAL", "50"))
    w.add_event(event("out", "EXTERNAL", "-50", DATES[2]))
    w.settle("week")
    result = w.value(DATES[2])["valuation"]
    assert result["net_external_cashflow"] == "0.00"
    assert result["external_funding_observed"] is True
    assert result["unit_nav"] is None


def test_net_zero_late_events_preserve_both_valuation_versions(tmp_path):
    w, _, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 3)
    w.settle("week")
    first = w.value(DATES[1])["valuation"]
    w.add_event(event("in", "EXTERNAL", "50"))
    w.add_event(event("out", "EXTERNAL", "-50"))
    second = w.value(DATES[1])["valuation"]
    assert first["nav"] == second["nav"]
    assert first["unit_nav"] is not None and second["unit_nav"] is None
    assert len(w.store.read()["valuations"]) == 2
    assert w.value(DATES[1])["status"] == "REUSED"


def test_file_metadata_source_is_read_without_mlflow_initialization(tmp_path):
    w, source, _, signal = make_workspace(tmp_path)
    signal_root = tmp_path / "legacy_signal"
    metadata = signal_root / "mlruns"
    record_id = "b" * 32
    artifact_root = signal_root / "artifact" / record_id
    artifact_root.mkdir(parents=True)
    pd.read_pickle(signal).to_pickle(artifact_root / "pred.pkl")
    experiment = metadata / "1"
    run_root = experiment / record_id
    run_root.mkdir(parents=True)
    (experiment / "meta.yaml").write_text("experiment_id: '1'\nlifecycle_stage: active\n")
    (run_root / "meta.yaml").write_text(
        "run_id: %s\nexperiment_id: '1'\nstatus: 3\nlifecycle_stage: active\nartifact_uri: %s\n" %
        (record_id, artifact_root.as_uri()))
    before = (run_root / "meta.yaml").read_bytes()
    w.bootstrap(source, DATES[0])
    plan = w.plan("file_backend", DATES[0], DATES[1], source_override={
        "workspace": str(signal_root), "mlflow_uri": metadata.as_uri(), "recorder_id": record_id,
        "experiment_id": "1"})["plan"]
    assert plan["source"]["kind"] == "MLFLOW"
    assert plan["source"]["recorder_id"] == record_id
    assert (run_root / "meta.yaml").read_bytes() == before
    assert not (signal_root / "mlflow.db").exists()


def test_reader_and_cash_results_do_not_depend_on_caller_decimal_context(tmp_path):
    from decimal import localcontext, ROUND_DOWN
    w, _, provider, _, _ = boot_plan(tmp_path)
    calendar(provider, 3)
    w.settle("week")
    normal = w.value(DATES[2])["valuation"]
    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_DOWN
        assert w.store.read()["valuations"][-1]["result"] == normal
        assert w.value(DATES[2])["valuation"] == normal


def test_missing_bootstrap_cash_is_rejected_without_creating_a_balance(tmp_path):
    w, source, _, _ = make_workspace(tmp_path)
    write_json(source, {"current_date": DATES[0],
                        "current_holding": [{"instrument": HELD[0], "value": 200, "amount": 1500}]})
    with pytest.raises(PaperError, match="explicit current_cash"):
        w.bootstrap(source, DATES[0])
    assert not w.store.path.exists()


def test_frozen_plan_survives_current_suggestion_change_and_next_cycle_uses_it(tmp_path, monkeypatch):
    from quantpits.utils.strategy import TopkDropoutOrderGenerator
    w, _, provider, _, old_plan = boot_plan(tmp_path)
    original_bytes = w.store.path.read_bytes()
    original = TopkDropoutOrderGenerator.generate_buy_orders
    generated = []
    def revised_suggestions(self, *args, **kwargs):
        rows = original(self, *args, **kwargs)
        for row in rows:
            row["estimated_amount"] = float(decimal(row["estimated_amount"]) + decimal("0.01"))
        generated.append(rows)
        return rows
    monkeypatch.setattr(TopkDropoutOrderGenerator, "generate_buy_orders", revised_suggestions)
    assert w.store.read()["plans"]["week"] == old_plan
    assert w.plan("week", DATES[0], DATES[1]) == {"status": "REUSED", "plan": old_plan}
    assert w.store.path.read_bytes() == original_bytes
    calendar(provider, 3)
    requested = {p["instrument"] for p in old_plan["prior"]["positions"]} | {p["instrument"] for p in old_plan["buys"]}
    expected = settle_plan(old_plan, old_plan["prior"], PaperMarket(provider).execution_quotes(DATES[1], requested),
                           old_plan["market_rules"])
    actual = w.settle("week")["settlement"]
    assert actual == expected
    assert w.settle("week") == {"status": "REUSED", "settlement": actual}
    assert w.value(DATES[2])["valuation"]["status"] == "COMPLETE"
    assert w.report(DATES[2])["paper"]["status"] == "COMPLETE"
    assert generated == []
    new_plan = w.plan("next", DATES[2], DATES[3])["plan"]
    assert len(generated) == 1
    assert generated[0]
    suggestions = {r["instrument"]: amount(r["estimated_amount"]) for r in generated[0]}
    assert {r["instrument"]: r["suggested_amount"] for r in new_plan["buys"] if r["suggested_quantity"]} == suggestions
    assert w.store.read()["plans"] == {"week": old_plan, "next": new_plan}
    calendar(provider, 5)
    assert w.settle("next")["status"] == "SETTLED"
    assert w.value(DATES[4])["valuation"]["status"] == "COMPLETE"
    assert w.report(DATES[4])["paper"]["status"] == "COMPLETE"
    assert w.store.read()["records"][0]["result"] == actual
    assert len(generated) == 1


def test_old_plan_read_and_execution_do_not_call_current_position_analysis(tmp_path, monkeypatch):
    from quantpits.utils.strategy import TopkDropoutOrderGenerator
    w, _, provider, _, old_plan = boot_plan(tmp_path)
    def unavailable(*args, **kwargs):
        raise RuntimeError("current position analysis unavailable")
    monkeypatch.setattr(TopkDropoutOrderGenerator, "analyze_positions_with_universe", unavailable)
    assert w.store.read()["plans"]["week"] == old_plan
    assert w.plan("week", DATES[0], DATES[1])["plan"] == old_plan
    calendar(provider, 3)
    w.settle("week")
    w.value(DATES[2])
    assert w.report(DATES[2])["paper"]["status"] == "COMPLETE"
    before = w.store.path.read_bytes()
    with pytest.raises(RuntimeError, match="current position analysis unavailable"):
        w.plan("next", DATES[2], DATES[3])
    assert w.store.path.read_bytes() == before


@pytest.mark.parametrize("path,value", [
    (("buys", 0, "suggested_amount"), "NaN"),
    (("buys", 0, "suggested_amount"), "-1.00"),
    (("buys", 0, "suggested_amount"), "1.001"),
    (("buys", 0, "suggested_amount"), 10.0),
    (("buys", 0, "suggested_quantity"), True),
    (("buys", 0, "suggested_quantity"), 125),
    (("buys", 0, "instrument"), "SH600999"),
    (("buys", 0, "anchor_close"), "10.01"),
    (("buys", 0, "score"), "-123"),
    (("buys", 0, "role"), "BACKUP"),
    (("sells", 0, "suggested_quantity"), 201),
    (("target_buys",), 4),
    (("predecessor_record_count",), 1),
    (("source", "anchor"), DATES[1]),
    (("prior", "cash"), "201"),
    (("ranking", 0, "raw_score"), "-123"),
    (("universe", 0), BUY[1]),
    (("schema_version",), True),
])
def test_frozen_plan_structural_corruption_is_rejected_after_rehash(tmp_path, path, value):
    w, _, _, _, _ = boot_plan(tmp_path)
    document = json.loads(w.store.path.read_text())
    member = document["plans"]["week"]
    for key in path[:-1]:
        member = member[key]
    member[path[-1]] = value
    document.pop("digest")
    document["digest"] = fingerprint_value(document)
    write_json(w.store.path, document)
    with pytest.raises(PaperError):
        w.store.read()


def test_duplicate_frozen_candidates_are_rejected_after_rehash(tmp_path):
    w, _, _, _, _ = boot_plan(tmp_path)
    document = json.loads(w.store.path.read_text())
    buys = document["plans"]["week"]["buys"]
    buys[1]["instrument"] = buys[0]["instrument"]
    document.pop("digest")
    document["digest"] = fingerprint_value(document)
    write_json(w.store.path, document)
    with pytest.raises(PaperError, match="unique instruments"):
        w.store.read()


def test_fractional_frozen_scores_preserve_float_representation(tmp_path):
    w, source, provider, signal = make_workspace(tmp_path)
    frame = pd.read_pickle(signal)
    frame["score"] *= 0.1
    frame.to_pickle(signal)
    w.bootstrap(source, DATES[0])
    plan = w.plan("fractional", DATES[0], DATES[1])["plan"]
    assert any(decimal(r["score"]) != decimal(next(
        ranked["raw_score"] for ranked in plan["ranking"] if ranked["instrument"] == r["instrument"]
    )) for r in plan["buys"])
    assert w.store.read()["plans"]["fractional"] == plan
    calendar(provider, 3)
    assert w.settle("fractional")["status"] == "SETTLED"
    assert w.value(DATES[2])["valuation"]["status"] == "COMPLETE"
