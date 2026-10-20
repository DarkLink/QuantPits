"""Account differences use saved holdings, suggestions and execution facts only."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from quantpits.research.account_comparison import _source, compare_accounts, load_account_group
from quantpits.research.account_comparison_reports import comparison_markdown, holding_rows, save_account_comparison
from quantpits.research.account_differences import _source_comparison, asset_difference, suggestion_digest
from quantpits.research.paper_accounting import decimal
from quantpits.scripts.compare_accounts import main
from tests.quantpits.research.test_account_comparison import group, hashes, write_csv  # noqa: F401
from tests.quantpits.research.test_paper_weekly import BUY, DATES, calendar, feature, make_workspace, write_json


def test_same_quantity_in_two_of_three_accounts_does_not_mean_all_hold(group):
    definition, *_ = group
    live = definition.members[0].ctx.root
    path = live / "data/holding_log_full.csv"
    path.write_text(path.read_text(encoding="utf-8-sig").replace("600001", "600002"), encoding="utf-8-sig")
    definition = replace(definition, members=definition.members[:2] + (
        replace(definition.members[2], ctx=definition.members[1].ctx),
    ))
    report = compare_accounts(definition, since=DATES[0], selected=DATES[2])
    row = next(r for r in holding_rows(report) if r["instrument"] == "SH600001")
    assert row["present_members"] == ["a", "b"]
    assert row["absent_members"] == ["live"]
    assert row["quantity_differs"] is False
    assert row["difference"] == "shared_subset_same_quantity"


def test_optional_broken_suggestion_keeps_valid_financial_snapshot(group):
    definition, *_ = group
    root = definition.members[0].ctx.root
    (root / "output/buy_suggestion_settled.csv").write_text("not,a,valid,suggestion\n")
    report = compare_accounts(definition, since=DATES[0], selected=DATES[2])
    member = report["members"][0]
    assert member["status"] == "complete"
    assert member["period"]["end"]["nav"] == "2120.00"
    assert member["source_gaps"]
    assert all(m["status"] == "complete" for m in report["members"])


def test_markdown_expands_actual_source_fields_instead_of_raw_dict(group):
    definition, *_ = group
    report = compare_accounts(definition, since=DATES[0], selected=DATES[2])
    text = comparison_markdown(report)
    assert "期间订单建议来源" in text
    assert "已结算计划来源" in text
    assert "experiment=synthetic" in text and "recorder=source-settled" in text
    assert "{'mode':" not in text and '"mode":' not in text


def test_overlap_sum_is_identical_across_python_hash_seeds():
    script = """
import json
from decimal import Decimal, localcontext
from quantpits.research.account_comparison import _pair
with localcontext() as context:
    context.prec = 50
    positions = {'SH%06d' % n: {'quantity': '100', 'weight': str(Decimal(1) / Decimal(n + 1))} for n in range(1, 60)}
    endpoint = {'holdings_available': True, 'reliable_nav': True, 'nav': '100', 'positions': positions}
    left = {'id': 'a', 'period': {'end': endpoint}}
    right = {'id': 'b', 'period': {'end': endpoint}}
    print(json.dumps(_pair(left, right), sort_keys=True))
"""
    outputs = []
    for seed in (1, 2, 3):
        environment = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=True)
        outputs.append(completed.stdout)
    assert len(set(outputs)) == 1


def settlement_rows(root):
    rows = []
    for quantity, price, gross, cash, fee, category in (
        (30, "10", "300", "-301", "1", "上海A股普通股票竞价买入"),
        (20, "12", "240", "-243", "3", "上海A股普通股票竞价买入"),
        (100, "0", "12", "12", "0", "上海A股红利入账"),
        (100, "0", "-2", "-2", "0", "上海A股红利税补缴"),
        (0, "0", "1", "1", "0", "利息归本"),
    ):
        rows.append({"成交日期": DATES[1], "交收日期": DATES[1], "证券代码": "SH600001", "成交数量": str(quantity),
            "成交价格": price, "成交金额": gross, "资金发生数": cash, "费用合计": fee,
            "交易类别": category, "资金账号": "synthetic-account"})
    write_csv(root / "data/trade_log_full.csv", rows)
    return rows


def explained(definition):
    return compare_accounts(definition, since=DATES[0], selected=DATES[2], explain=True)


def test_explanations_join_split_fills_and_keep_dividend_tax_interest_separate(group):
    definition, _a, _b, provider = group
    settlement_rows(definition.members[0].ctx.root)
    before = {m.id: hashes(m.ctx.root) for m in definition.members}
    provider_before = hashes(provider)
    report = explained(definition)
    assert report["status"] == report["explanations"]["status"] == "complete"
    day = report["members"][0]["execution"]["days"][0]
    fill, = day["executions"]
    assert (fill["quantity"], fill["gross"], fill["fees"], fill["fill_count"]) == ("50", "540.00", "4.00", 2)
    assert decimal(fill["average_price"]) == decimal("10.8")  # not the unweighted 11
    assert {e["category"] for e in day["cash_events"]} == {"上海A股红利入账", "上海A股红利税补缴", "利息归本"}
    assert day["cash_events"][0]["observed_quantity"] == "100"
    assert report["members"][0]["period"]["fees"] == "4.00"
    assert {r["date"] for r in report["explanations"]["executions"]} == {DATES[1]}
    assert report["members"][1]["execution"]["pending_plans"][0]["trade_date"] == DATES[3]
    backup = next(r for r in report["members"][1]["execution"]["days"][0]["executions"] if r["reason"] == "NO_SLOT")
    assert backup["quantity"] == "0" and backup["fees"] is None
    assert "not an order rejection" in backup["reason_label"]
    suggestion = next(s for s in report["members"][1]["execution"]["suggestion_versions"][0]["suggestions"] if s["instrument"] == backup["instrument"])
    assert suggestion["role"] == "BACKUP" and suggestion["position"] > 1
    actual_absence = next(m for r in report["explanations"]["executions"] if r["instrument"] == backup["instrument"] for m in r["members"] if m["member_id"] == "live")
    assert actual_absence["status"] == "no_recorded_fill" and actual_absence["executions"] == []
    paths = save_account_comparison(definition, report, definition.path.parent / "reports")
    assert len(paths) == 7
    path = next(path for name, path in paths.items() if name.endswith("_executions.csv"))
    with open(path, encoding="utf-8-sig") as handle:
        exported = list(csv.DictReader(handle))
    exported_fill = next(r for r in exported if r["member_id"] == "live" and r["instrument"] == "SH600001")
    assert (exported_fill["suggested_quantity"], exported_fill["recorded_quantity"], exported_fill["fees"], exported_fill["fill_count"]) == ("100", "50", "4.00", "2")
    exported_skip = next(r for r in exported if r["member_id"] == "a" and r["skip_reasons"] == "NO_SLOT")
    assert exported_skip["recorded_quantity"] == "0" and exported_skip["fees"] == ""
    text = comparison_markdown(report)
    assert "期间收益贡献" in text and "期初 NAV 差" in text and "unknown" not in fill["reason_label"]
    assert "current_readable_no_historical_digest" in text and "备选未使用" in text
    assert before == {m.id: hashes(m.ctx.root) for m in definition.members}
    assert provider_before == hashes(provider)


def test_asset_identity_historical_endpoint_quantities_and_period_bridge(group):
    definition, *_ = group
    report = explained(definition)
    pair = report["explanations"]["asset_differences"][0]
    assert pair["direction"] == "LEFT_MINUS_RIGHT" and pair["left_date"] == pair["right_date"] == DATES[2]
    assert decimal(pair["nav_difference"]) == decimal(pair["cash_difference"]) + sum(decimal(r["market_value_difference"]) for r in pair["stocks"])
    assert pair["detail_total"] == pair["nav_difference"] and pair["rounding_residual"] == "0.00"
    only = next(r for r in pair["stocks"] if r["instrument"] == "SZ000090")
    shared = next(r for r in pair["stocks"] if r["instrument"] == "SH600001")
    assert only["right_quantity"] == "0" and shared["quantity_difference"] == "-50"
    bridge = pair["period_bridge"]
    assert bridge["start_nav_difference"] == "900.00" and bridge["pnl_difference"] == "15.01"
    assert decimal(pair["nav_difference"]) == sum(decimal(v) for v in bridge.values())
    pair_ab = report["explanations"]["asset_differences"][2]
    assert pair_ab["period_bridge"]["net_external_cashflow_difference"] == "100.00"
    # Each original endpoint may legitimately contain one cent of NAV rounding.
    left, right = report["members"][:2]
    left["period"]["end"]["nav"] = "2120.01"
    residual = asset_difference(left, right, DATES[2])
    assert residual["rounding_residual"] == "0.01"
    assert residual["detail_total"] != residual["nav_difference"] and residual["status"] == "complete"


def test_missing_member_only_disables_related_pairs_and_never_becomes_zero(group):
    definition, _a, b, _provider = group
    b.store.path.write_text('{"broken": true}')
    report = explained(definition)
    assert [p["status"] for p in report["explanations"]["asset_differences"]] == ["complete", "unavailable", "unavailable"]
    rows = holding_rows(report)
    unknown = next(r for r in rows if r["member_id"] == "b")
    assert unknown["quantity"] is None and unknown["unknown_members"] == ["b"]
    assert unknown["difference"] == "incomplete"


def test_digest_checks_and_multiple_versions_do_not_select_adopted_orders(group):
    definition, *_ = group
    root = definition.members[0].ctx.root
    settlement_rows(root)
    path = root / "output/manifests/order_gen/settled.json"
    manifest = json.loads(path.read_text())
    artifact = root / manifest["outputs"][0]["path"]
    manifest["outputs"][0]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    verified = explained(definition)["members"][0]["sources"]["settled_period"][0]
    assert verified["artifact_facts"][0]["status"] == "recorded_digest_verified"
    manifest["run_id"] = "another-version"
    (path.parent / "aaa-other.json").write_text(json.dumps(manifest))
    report = explained(definition)
    versions = report["members"][0]["execution"]["suggestion_versions"]
    assert len(versions) == 2 and all(v["selection_status"] == "ambiguous_multiple_successful_versions" for v in versions)
    assert report["explanations"]["source_comparisons"][0]["status"] == "ambiguous_suggestion_versions"
    artifact.write_text("instrument,quantity\nSH600001,200\n")
    report = explained(definition)
    assert report["status"] == "complete" and report["explanations"]["status"] == "partial"
    assert report["members"][0]["execution"]["days"][0]["executions"][0]["quantity"] == "50"
    assert any("digest mismatch" in g["detail"] for g in report["members"][0]["source_gaps"])
    assert all(v["suggestions"] == [] for v in report["members"][0]["execution"]["suggestion_versions"])


@pytest.mark.parametrize("digest", ["bad", {"algorithm": "md5", "value": "a" * 64}, {"algorithm": "sha256", "value": "a" * 64, "size_bytes": 999}])
def test_unusable_declared_digest_is_not_verified(digest):
    with pytest.raises(ValueError):
        suggestion_digest({"fingerprint": digest}, b"input")


def test_recorded_fusion_matching_uses_workspace_anchor_recorder_not_file_name():
    def member(kind, source):
        return {"id": kind, "execution": {"status": "complete", "days": [], "suggestion_versions": [{"trade_date": DATES[1], "source": source}]}}
    left = member("actual", {"workspace": "/synthetic/source", "anchor": DATES[0], "reference": {"mode": "ensemble", "record_id": "fusion"}})
    right = member("paper", {"workspace": "/synthetic/source", "anchor": DATES[0], "recorder_id": "fusion", "path": "different.pkl"})
    assert _source_comparison(left, right, DATES[1])["status"] == "same_recorded_fusion_recorder"
    right["execution"]["suggestion_versions"][0]["source"]["recorder_id"] = "different"
    assert _source_comparison(left, right, DATES[1])["status"] == "different_recorded_references"
    right["execution"]["suggestion_versions"][0]["source"] = {"path": "same_filename.csv"}
    assert _source_comparison(left, right, DATES[1])["status"] == "unknown"


def test_unobserved_trade_category_is_an_explanation_gap_not_a_direction_guess(group):
    definition, *_ = group
    root = definition.members[0].ctx.root
    rows = settlement_rows(root)
    rows[0]["交易类别"] = "模糊买入说明"
    rows.append(dict(rows[1], 成交日期=DATES[2], 交收日期=DATES[2]))
    write_csv(root / "data/trade_log_full.csv", rows)
    report = explained(definition)
    assert report["status"] == "complete" and report["explanations"]["status"] == "partial"
    assert report["members"][0]["execution"]["days"][0]["executions"] == []
    assert report["members"][0]["execution"]["days"][1]["executions"][0]["quantity"] == "20"
    assert any("UnsupportedSettlementEventError" in g["detail"] for g in report["members"][0]["execution"]["gaps"])


def test_extra_suggestion_read_remains_observed_until_group_end_and_interrupt_propagates(group, monkeypatch):
    from quantpits.research import account_comparison as module
    definition, *_ = group
    root = definition.members[0].ctx.root
    settlement_rows(root)
    original = module._read_paper
    def changed(*args):
        original(*args)
        path = root / "output/buy_suggestion_settled.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(module, "_read_paper", changed)
    report = explained(definition)
    assert report["members"][0]["status"] == "unstable"
    assert report["members"][0]["period"] is report["members"][0]["execution"] is None
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, "saved_suggestions", interrupt)
    with pytest.raises(KeyboardInterrupt):
        explained(definition)


def test_explain_cli_partial_scope_and_additional_output_symlink_protection(group, tmp_path, capsys):
    definition, *_ = group
    args = ["--group", str(definition.path), "--since", DATES[0], "--date", DATES[2], "--output", str(tmp_path / "reports"), "--explain"]
    assert main(args) == 2  # old minimal trade fixture cannot support normalized executions
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete" and result["explanation_status"] == "partial" and len(result["reports"]) == 7
    root = definition.members[0].ctx.root
    settlement_rows(root)
    assert main(args) == 0
    report = explained(definition)
    output = tmp_path / "unsafe_reports"
    output.mkdir()
    (output / ("comparison_%s_%s_executions.csv" % (DATES[0], DATES[2]))).symlink_to(root / "data/trade_log_full.csv")
    before = hashes(root)
    with pytest.raises(ValueError, match="workspace or provider"):
        save_account_comparison(definition, report, output)
    assert len(list(output.iterdir())) == 1 and hashes(root) == before


def test_optional_paper_source_metadata_error_preserves_validated_financial_facts(group, monkeypatch):
    from quantpits.research import account_comparison as module
    definition, *_ = group
    original = module._source
    def unreadable_optional_metadata(source):
        original(source)
        raise ValueError("unreadable optional lineage annotation")
    monkeypatch.setattr(module, "_source", unreadable_optional_metadata)
    report = explained(definition)
    paper = report["members"][1]
    assert paper["status"] == "complete" and paper["period"]["end"]["nav"] == "1204.99"
    assert paper["source_gaps"] and paper["execution"]["days"][0]["executions"]
    assert report["explanations"]["asset_differences"][0]["status"] == "complete"


def test_skipped_limit_quote_is_not_reported_as_execution_price(tmp_path):
    workflow, source, provider, _signal = make_workspace(tmp_path, topk=1)
    write_json(source, {"current_date": DATES[0], "current_cash": 2100, "current_holding": []})
    workflow.bootstrap(source, DATES[0])
    workflow.plan("limit-test", DATES[0], DATES[1])
    calendar(provider, 3)
    feature(provider, BUY[0], "open", [10, 11, 10, 10, 10])
    workflow.settle("limit-test")
    workflow.value(DATES[2])
    path = tmp_path / "limit-group.yaml"
    write_json(path, {"name": "Limit decision", "members": [{"id": "paper", "kind": "paper", "workspace": str(workflow.ctx.root)}]})
    report = explained(load_account_group(path))
    row = next(r for r in report["members"][0]["execution"]["days"][0]["executions"] if r["reason"] == "OPEN_AT_LIMIT")
    assert row["observed_price"] == "11.00" and row["quantity"] == "0" and row["average_price"] is None
    text = comparison_markdown(report)
    assert "OPEN_AT_LIMIT" in text and "观察价=11.0000" in text


def test_source_experiment_comes_from_matching_fusion_recorder_evidence():
    source = {"recorder_id": "fusion", "lineage": [{"snapshot": {"command": "ensemble_fusion", "records": {
        "experiment_name": "training-name", "combos": [
            {"name": "other", "recorder_id": "other", "output_evidence": {"recorder_id": "other", "experiment_name": "other-name"}},
            {"name": "chosen", "recorder_id": "fusion", "output_evidence": {"recorder_id": "fusion", "experiment_name": "Fusion"}},
        ]}}}]}
    result = _source(source)
    assert result["combo_name"] == "chosen" and result["experiment_name"] == "Fusion"
