"""Saved three-account financial, lineage, and read-only report boundaries."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from quantpits.research.account_comparison import (
    AccountComparisonError, compare_accounts, load_account_group,
)
from quantpits.research.account_comparison_reports import holding_rows, save_account_comparison
from quantpits.research.paper_accounting import decimal
from quantpits.scripts.compare_accounts import main
from tests.quantpits.research.test_paper_weekly import (
    DATES, calendar, event, feature, make_workspace, write_json,
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def actual(tmp_path):
    root = tmp_path / "actual"
    write_json(root / "config/prod_config.json", {"current_date": DATES[4], "last_processed_date": DATES[4],
                                                 "current_cash": 999999, "current_holding": []})
    write_csv(root / "data/daily_amount_log_full.csv", [
        {"成交日期": DATES[0], "收盘价值": "2000", "CASHFLOW": "0"},
        {"成交日期": DATES[2], "收盘价值": "2120", "CASHFLOW": "100"},
        {"成交日期": DATES[4], "收盘价值": "3000", "CASHFLOW": "0"},
    ])
    write_csv(root / "data/holding_log_full.csv", [
        {"成交日期": day, "证券代码": instrument, "持仓数量": quantity, "收盘价值": value}
        for day, rows in ((DATES[0], [("CASH", "1000", "1000"), ("SH600001", "50", "500"), ("SZ000090", "100", "500")]),
                          (DATES[2], [("CASH", "1120", "1120"), ("600001", "50", "500"), ("SZ000090", "100", "500")]),
                          (DATES[4], [("CASH", "1000", "1000"), ("SH600002", "200", "2000")]))
        for instrument, quantity, value in rows
    ])
    write_csv(root / "data/trade_log_full.csv", [
        {"成交日期": "2024-04-01", "资金账号": "synthetic-account", "费用合计": "99", "交易类别": "BUY"},
        {"成交日期": DATES[1], "资金账号": "synthetic-account", "费用合计": "4", "交易类别": "BUY"},
        {"成交日期": DATES[1], "资金账号": "synthetic-account", "费用合计": "0", "交易类别": "DIVIDEND"},
    ])
    for run_id, anchor, trade in (("settled", DATES[0], DATES[1]), ("next", DATES[4], "2024-04-22")):
        artifact = "output/buy_suggestion_%s.csv" % run_id
        (root / "output").mkdir(exist_ok=True)
        (root / artifact).write_text("instrument,quantity\nSH600001,100\n")
        write_json(root / ("output/manifests/order_gen/%s.json" % run_id), {
            "schema_version": 1, "workspace": str(root) if run_id == "settled" else root.name,
            "command": "order_gen", "run_id": run_id, "status": "success",
            "records": {"anchor_date": anchor, "trade_date": trade,
                        "source": {"mode": "ensemble", "record_id": "source-" + run_id, "experiment_name": "synthetic"}},
            "outputs": [{"path": artifact, "kind": "report"}],
        })
    return root


def paper(tmp_path, name, cash, with_events=False):
    workflow, source, provider, signal = make_workspace(tmp_path, name, topk=1)
    write_json(source, {"current_date": DATES[0], "current_cash": cash, "current_holding": []})
    workflow.bootstrap(source, DATES[0])
    workflow.plan("first", DATES[0], DATES[1])
    if with_events:
        workflow.add_event(event("funding", "EXTERNAL", "100"))
        workflow.add_event(event("distribution", "DIVIDEND", "10"))
    calendar(provider, 3)
    workflow.settle("first")
    workflow.value(DATES[2])
    workflow.plan("pending", DATES[2], DATES[3])
    # The default is mutable configuration, not the already frozen plan source.
    config = workflow.config.copy()
    config["signal_source"] = {"workspace": str(tmp_path / "unused"), "prediction": "not-these-signals.csv"}
    write_json(workflow.ctx.config_path("paper_config.json"), config)
    return workflow, provider


@pytest.fixture
def group(tmp_path):
    root = actual(tmp_path)
    a, provider = paper(tmp_path, "paper_a", 1100, with_events=True)
    b, _ = paper(tmp_path, "paper_b", 2100)
    group_path = tmp_path / "group.yaml"
    group_path.write_text(yaml.safe_dump({"name": "Synthetic comparison", "members": [
        {"id": "live", "kind": "actual", "workspace": root.name},
        {"id": "a", "label": "Paper A", "kind": "paper", "workspace": a.ctx.root.name},
        {"id": "b", "label": "Paper B", "kind": "paper", "workspace": b.ctx.root.name},
    ]}, sort_keys=False))
    return load_account_group(group_path), a, b, provider


def compare(group):
    return compare_accounts(group, since=DATES[0], selected=DATES[2])


def test_three_accounts_period_positions_sources_and_no_writes(group):
    definition, a, b, provider = group
    before = {m.id: hashes(m.ctx.root) for m in definition.members}
    provider_before = hashes(provider)
    report = compare(definition)
    assert report["status"] == "complete"
    assert [m["id"] for m in report["members"]] == ["live", "a", "b"]
    live, paper_a, paper_b = report["members"]
    assert live["account_id"] == "synthetic-account"
    assert live["period"]["pnl_excluding_external"] == "20.00"
    assert live["period"]["return_excluding_external"] == "0.01"
    assert live["period"]["fees"] == "4.00"  # no historical 99 fee and no second deduction
    assert live["period"]["end"]["cash"] == "1120.00"
    assert live["period"]["end"]["shares"] == "150"
    assert "SH600002" not in live["period"]["end"]["positions"]
    assert paper_a["period"]["net_external_cashflow"] == "100.00"
    assert paper_a["period"]["end"]["nav"] == "1204.99"
    assert paper_a["period"]["pnl_excluding_external"] == "4.99"  # dividend less fee
    assert paper_a["period"]["fees"] == "5.01"
    assert paper_b["period"]["fees"] == "5.02"
    assert paper_a["current"]["last_processed_date"] == DATES[1]
    assert paper_a["current"]["last_valuation_date"] == DATES[2]
    assert paper_a["sources"]["settled_period"][0]["anchor"] == DATES[0]
    assert paper_a["sources"]["pending_current"][0]["anchor"] == DATES[2]
    assert paper_a["current"]["next_signal"]["workspace"] == str(a.ctx.root.parent)
    pair = report["pairs"][-1]
    assert pair["common_holdings"] == 1 and pair["quantity_differences"] == ["SH600001"]
    expected_overlap = min(decimal("1000") / decimal("1204.99"), decimal("2000") / decimal("2094.98"))
    assert abs(decimal(pair["stock_weight_overlap"]) - expected_overlap) < decimal("1e-27")
    assert report["pairs"][0]["only_left"] == ["SZ000090"]
    rows = holding_rows(report)
    assert next(r for r in rows if r["instrument"] == "SH600001")["difference"] == "quantity_differs"
    paths = save_account_comparison(definition, report, definition.path.parent / "reports")
    assert len(paths) == 5
    assert all(Path(path).exists() for path in paths.values())
    assert before == {m.id: hashes(m.ctx.root) for m in definition.members}
    assert provider_before == hashes(provider)


def test_later_current_pending_does_not_change_historical_period(group):
    definition, a, _b, provider = group
    calendar(provider, 5)
    a.settle("pending")
    a.value(DATES[4])
    future = provider / "calendars/day_future.txt"
    future.write_text(future.read_text() + "2024-04-22\n")
    a.plan("later", DATES[4], "2024-04-22")
    result = compare(definition)["members"][1]
    assert result["current"]["pending"][0]["anchor"] == DATES[4]
    assert result["period"]["end"]["nav"] == "1204.99"
    assert result["period"]["fees"] == "5.01"
    later = compare_accounts(definition, since=DATES[2], selected=DATES[4])["members"][1]
    assert later["period"]["fees"] == "0.00"  # no cumulative first-cycle fee


@pytest.mark.parametrize("problem", ["missing_start", "unknown_cashflow", "corrupt", "behind"])
def test_partial_members_preserve_request_and_other_results(group, problem):
    definition, a, b, _provider = group
    live = definition.members[0].ctx.root
    if problem == "corrupt":
        b.store.path.write_text('{"broken": true}')
    elif problem == "behind":
        b.store.path.unlink()
    else:
        path = live / "data/daily_amount_log_full.csv"
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if problem == "missing_start":
            rows = [r for r in rows if r["成交日期"] != DATES[0]]
        else:
            rows[1].pop("CASHFLOW")
            for row in rows:
                row.pop("CASHFLOW", None)
        write_csv(path, rows)
    report = compare(definition)
    assert report["status"] == "partial"
    assert [m["id"] for m in report["members"]] == ["live", "a", "b"]
    assert report["members"][1]["status"] == "complete"
    if problem in ("corrupt", "behind"):
        assert report["members"][2]["status"] == "error"
        assert report["pairs"][0]["status"] == "complete"
        assert report["pairs"][1]["status"] == "unavailable"
    else:
        assert report["members"][0]["period"]["return_excluding_external"] is None
        assert report["members"][0]["period"]["end"]["nav"] == "2120.00"


def test_changed_source_discards_mixed_member_snapshot(group, monkeypatch):
    from quantpits.research import account_comparison as module
    definition, a, b, _provider = group
    original = module._read_actual

    def changed(member, reader, result, since, selected):
        original(member, reader, result, since, selected)
        path = member.ctx.config_path("prod_config.json")
        path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(module, "_read_actual", changed)
    report = compare(definition)
    assert report["members"][0]["status"] == "unstable"
    assert report["members"][0]["period"] is None
    assert report["members"][1]["status"] == report["members"][2]["status"] == "complete"


def test_earlier_member_mutation_while_reading_later_member_is_not_mixed(group, monkeypatch):
    from quantpits.research import account_comparison as module
    definition, *_ = group
    original = module._read_paper

    def changed(member, reader, result, since, selected):
        original(member, reader, result, since, selected)
        if member.id == "b":
            path = definition.members[0].ctx.config_path("prod_config.json")
            path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(module, "_read_paper", changed)
    report = compare(definition)
    assert report["members"][0]["status"] == "unstable"
    assert report["members"][0]["period"] is None


def test_stale_paper_nav_is_shown_but_not_used_as_reliable_period_return(group):
    definition, a, b, provider = group
    calendar(provider, 5)
    a.settle("pending")
    feature(provider, "SH600001", "close", [10, 10, 10, 10, float("nan")])
    assert a.value(DATES[4])["valuation"]["status"] == "STALE"
    report = compare_accounts(definition, since=DATES[2], selected=DATES[4])
    member = report["members"][1]
    assert member["period"]["end"]["nav"] is not None
    assert member["period"]["return_excluding_external"] is None
    assert report["pairs"][0]["status"] == "unavailable"
    assert all(row["quantity"] is None for row in holding_rows(report) if row["member_id"] == "a")


@pytest.mark.parametrize("cashflow", ["", "nan", "Infinity"])
def test_missing_or_nonfinite_cashflow_does_not_hide_nav_or_create_return(group, cashflow):
    definition, *_ = group
    path = definition.members[0].ctx.data_path("daily_amount_log_full.csv")
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["CASHFLOW"] = cashflow
    write_csv(path, rows)
    result = compare(definition)["members"][0]
    assert result["period"]["end"]["nav"] == "2120.00"
    assert result["period"]["return_excluding_external"] is None
    assert result["status"] == "partial"


def test_missing_paper_baseline_does_not_move_requested_start(group):
    definition, *_ = group
    report = compare_accounts(definition, since="2024-04-04", selected=DATES[2])
    member = report["members"][1]
    assert member["period"]["start"]["date"] == "2024-04-04"
    assert member["period"]["start"]["nav"] is None
    assert member["period"]["end"]["nav"] == "1204.99"
    assert member["period"]["return_excluding_external"] is None
    assert member["available_dates"][0] == DATES[0]


def test_nonpositive_start_nav_has_no_period_return(group):
    definition, *_ = group
    path = definition.members[0].ctx.data_path("daily_amount_log_full.csv")
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["收盘价值"] = "0"
    write_csv(path, rows)
    member = compare(definition)["members"][0]
    assert member["period"]["return_excluding_external"] is None
    assert any(gap["code"] == "nonpositive_start_nav" for gap in member["gaps"])


def test_interrupt_propagates(group, monkeypatch):
    definition, *_ = group
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr("quantpits.research.account_comparison._read_actual", interrupt)
    with pytest.raises(KeyboardInterrupt):
        compare(definition)


@pytest.mark.parametrize("target", ["workspace", "provider", "symlink"])
def test_report_output_cannot_enter_account_or_provider(group, target):
    definition, a, b, provider = group
    report = compare(definition)
    output = a.ctx.root / "new-report" if target == "workspace" else provider / "new-report"
    if target == "symlink":
        output = definition.path.parent / "report-link"
        output.symlink_to(a.ctx.root, target_is_directory=True)
    before = hashes(a.ctx.root), hashes(provider)
    with pytest.raises(AccountComparisonError, match="workspace or provider"):
        save_account_comparison(definition, report, output)
    assert before == (hashes(a.ctx.root), hashes(provider))


def test_cli_codes_environment_and_help(group, tmp_path, monkeypatch, capsys):
    definition, a, b, _provider = group
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(tmp_path / "unrelated"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////must-not-create.db")
    monkeypatch.chdir(tmp_path.parent)
    arguments = ["--group", str(definition.path), "--since", DATES[0], "--date", DATES[2], "--output", str(tmp_path / "reports")]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
    b.store.path.unlink()
    assert main(arguments) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "partial"
    assert main(["--group", str(tmp_path / "missing.yaml")] + arguments[2:]) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "error"
    env = dict(os.environ)
    env.pop("QLIB_WORKSPACE_DIR", None)
    env.pop("MLFLOW_TRACKING_URI", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    result = subprocess.run([sys.executable, "-m", "quantpits.scripts.compare_accounts", "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    script = """
import builtins, os
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'quantpits.utils.env' or name.split('.')[0] in ('qlib', 'mlflow'):
        raise AssertionError('forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
def no_chdir(*args):
    raise AssertionError('chdir called')
os.chdir = no_chdir
from quantpits.scripts.compare_accounts import main
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_group_rejects_duplicate_members_before_account_reads(group):
    definition, *_ = group
    value = yaml.safe_load(definition.path.read_text())
    value["members"][2]["id"] = "a"
    definition.path.write_text(yaml.safe_dump(value))
    with pytest.raises(AccountComparisonError, match="unique"):
        load_account_group(definition.path)


def test_invalid_yaml_is_a_clean_cli_error(tmp_path, capsys):
    path = tmp_path / "invalid.yaml"
    path.write_text("members: [")
    assert main(["--group", str(path), "--since", DATES[0], "--date", DATES[2], "--output", str(tmp_path / "reports")]) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "error"
