"""Human/CSV views and explicitly scoped persistence of account comparisons."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from quantpits.research.account_comparison import AccountComparisonError, _sha, _stable_read, _comparable
from quantpits.research.paper_accounting import ZERO, amount, decimal, cash_context
from quantpits.training.persistence import atomic_write_bytes


def _display(value):
    return "—" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def _percent(value):
    return "—" if value is None else format(decimal(value) * 100, ".4f") + "%"


def _price(value):
    return "—" if value is None else format(decimal(value), ".4f")


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(_display(value) for value in row) + " |" for row in rows]


def summary_rows(report):
    rows = []
    for member in report["members"]:
        period, current = member["period"] or {}, member["current"] or {}
        start, end = period.get("start", {}), period.get("end", {})
        rows.append({"id": member["id"], "label": member["label"], "kind": member["kind"],
                     "account_id": member["account_id"], "status": member["status"],
                     "since": report["since"], "date": report["date"],
                     "last_processed_date": current.get("last_processed_date"), "last_valuation_date": current.get("last_valuation_date"),
                     "start_nav": start.get("nav"), "end_nav": end.get("nav"), "cash": end.get("cash"),
                     "position_count": end.get("position_count"), "shares": end.get("shares"), "cash_ratio": end.get("cash_ratio"),
                     "net_external_cashflow": period.get("net_external_cashflow"),
                     "pnl_excluding_external": period.get("pnl_excluding_external"),
                     "return_excluding_external": period.get("return_excluding_external"), "fees": period.get("fees")})
    return rows


def holding_rows(report):
    available = {m["id"]: m["period"]["end"]["positions"] for m in report["members"] if _comparable(m)}
    instruments = sorted({i for positions in available.values() for i in positions})
    rows = []
    for instrument in instruments:
        present = [member_id for member_id, positions in available.items() if instrument in positions]
        absent = [member_id for member_id, positions in available.items() if instrument not in positions]
        unknown = [m["id"] for m in report["members"] if m["id"] not in available]
        quantities = {decimal(available[member_id][instrument]["quantity"]) for member_id in present}
        difference = ("incomplete" if unknown else "only_one_account" if len(present) == 1
                      else "quantity_differs" if len(quantities) > 1 else "shared_subset_same_quantity" if absent else "shared_same_quantity")
        for member in report["members"]:
            positions = available.get(member["id"])
            position = positions.get(instrument) if positions is not None else None
            rows.append({"instrument": instrument, "member_id": member["id"], "difference": difference,
                         "present_members": present, "absent_members": absent, "unknown_members": unknown,
                         "quantity_differs": True if len(quantities) > 1 else None if unknown else False,
                         "status": "unavailable" if positions is None else "present" if position else "absent",
                         "quantity": position["quantity"] if position else "0" if positions is not None else None,
                         "market_value": position["market_value"] if position else "0.00" if positions is not None else None,
                         "weight": position["weight"] if position else "0" if positions is not None else None})
    return rows


def _source_text(source):
    reference = source.get("reference") or {}
    fields = [("status", source.get("status")), ("workspace", source.get("workspace")),
              ("combo", source.get("combo_name") or reference.get("resolved_name")),
              ("experiment", source.get("experiment_name") or reference.get("experiment_name") or source.get("experiment_id")),
              ("recorder", source.get("recorder_id") or reference.get("record_id")), ("file", source.get("path")),
              ("anchor", source.get("anchor"))]
    return "; ".join("%s=%s" % (key, _display(value)) for key, value in fields if value is not None)


def _explanation_markdown(report):
    explanation = report.get("explanations")
    if explanation is None:
        return []
    lines = ["## 期末资产差额 / End-date asset differences", "", "说明状态 / explanation status: " + explanation["status"], "",
             "以下均为左账户减右账户的所选日期资产差额，不是证券期间收益贡献。金额保留账面精度，残差单列。", ""]
    for pair in explanation["asset_differences"]:
        lines += ["### %s − %s · %s · %s" % (pair["left"], pair["right"], pair["date"], pair["status"]), ""]
        if pair["status"] == "unavailable":
            lines += ["缺少可靠末日估值或完整持仓 / reliable end-date holdings unavailable", ""]
            continue
        lines += _table(["左数据日", "右数据日", "现金差", "股票市值差合计", "明细合计", "账面 NAV 差", "舍入残差"], [[
            pair["left_date"], pair["right_date"], pair["cash_difference"], pair["stock_difference_total"], pair["detail_total"],
            pair["nav_difference"], pair["rounding_residual"]]])
        lines += [""] + _table(["证券", "左数量", "右数量", "数量差", "左市值", "右市值", "市值差", "左权重", "右权重"], [
            [r["instrument"], r["left_quantity"], r["right_quantity"], r["quantity_difference"], r["left_market_value"],
             r["right_market_value"], r["market_value_difference"], _percent(r["left_weight"]), _percent(r["right_weight"])]
            for r in pair["stocks"] if decimal(r["market_value_difference"]) != 0 or decimal(r["quantity_difference"]) != 0])
        bridge = pair["period_bridge"]
        if bridge:
            lines += ["", "期间桥接：期初 NAV 差 %s + 净外部入金差 %s + 期间损益差 %s = 期末 NAV 差 %s。" % (
                bridge["start_nav_difference"], bridge["net_external_cashflow_difference"], bridge["pnl_difference"], pair["nav_difference"])]
        lines.append("")
    lines += ["## 本期成交与冻结决策 / Executions and frozen decisions", "",
              "实盘均价为成交金额合计 ÷ 数量合计；没有成交行时，委托、撤单和人工选择原因未知。paper NO_SLOT 表示备选未使用，不是订单被拒。", ""]
    execution_rows = []
    for row in explanation["executions"]:
        for member in row["members"]:
            suggestions = "; ".join("%s #%s %s股 [%s; %s]" % (s["role"], s["position"], s["suggested_quantity"],
                s["version_id"], s.get("verification", "not_recorded")) for s in member["suggestions"])
            fills = "; ".join("%s %s股; 均价=%s; 费用=%s; 笔数=%s; 原因=%s%s" % (f["status"], f["quantity"],
                _price(f["average_price"]), _display(f["fees"]), f["fill_count"], _display(f["reason_label"]),
                "; 观察价=" + _price(f["observed_price"]) if f["status"] == "SKIPPED" and f.get("observed_price") is not None else "")
                for f in member["executions"])
            execution_rows.append([row["date"], row["instrument"], row["side"], member["member_id"], member["status"],
                                   suggestions or "未记录", member["suggestion_selection"], fills or "未记录"])
    lines += _table(["执行日", "证券", "方向", "账户", "状态", "建议 / 顺序 / 版本", "建议绑定", "实际或模拟记录"], execution_rows)
    lines += ["", "### 保存的来源对照", ""] + _table(["执行日", "左账户", "右账户", "来源关系", "执行结果不同"], [
        [r["date"], r["left"], r["right"], r["status"], r["execution_results_differ"]] for r in explanation["source_comparisons"]])
    lines += ["", "相同融合 recorder 仅说明保存了同一来源引用，不证明模型或信号等价。", ""]
    for member in report["members"]:
        execution = member.get("execution") or {}
        events = [dict(e, date=d["date"]) for d in execution.get("days", []) for e in d["cash_events"]]
        events += execution.get("cash_events", [])
        if events:
            lines += ["### %s · 单列现金／持仓事件" % member["label"], ""] + _table(["日期", "类别", "金额", "记录数量（非买卖）"], [
                [e.get("date", e.get("applied_state_date")), e.get("category", e.get("kind")),
                 e.get("cash_effect", e.get("amount")), e.get("observed_quantity")] for e in events]) + [""]
        for gap in execution.get("gaps", []):
            lines.append("- %s · %s: %s" % (member["id"], _display(gap.get("date")), _display(gap["detail"])))
    return lines + [""]


@cash_context
def comparison_markdown(report):
    rows = summary_rows(report)
    lines = ["# " + _display(report["name"]), "", "%s → %s · %s" % (report["since"], report["date"], report["status"]),
             "", "## 所选历史期间 / Selected historical period", ""]
    lines += _table(["账户 / Account", "状态", "期初 NAV", "期末 NAV", "净外部入金", "损益", "起点本金收益率", "期间费用", "现金", "股票数", "股数", "现金占比"], [
        [r["label"], r["status"], r["start_nav"], r["end_nav"], r["net_external_cashflow"], r["pnl_excluding_external"],
         _percent(r["return_excluding_external"]), r["fees"], r["cash"], r["position_count"], r["shares"], _percent(r["cash_ratio"])] for r in rows])
    lines += ["", "## 当前进度 / Current progress", ""]
    lines += _table(["账户", "账本账户 ID", "最后处理日", "最后估值日", "当前待执行 / 建议"], [
        [m["label"], m["account_id"], (m["current"] or {}).get("last_processed_date"),
         (m["current"] or {}).get("last_valuation_date"),
         "; ".join("%s: %s → %s" % (p["cycle_id"], p["anchor"], p["trade_date"]) for p in (m["current"] or {}).get("pending", []) or [])
         if m["kind"] == "paper" else "; ".join("suggestions: %s → %s (%s)" % (s["anchor"], s["trade_date"], s["status"])
                                                      for s in (m["current"] or {}).get("suggestions", [])) or "unknown"]
        for m in report["members"]])
    lines += ["", "## 末日持仓对照 / End-date holdings", ""]
    lines += _table(["账户 A", "账户 B", "状态", "共有股票", "仅 A", "仅 B", "同股不同数量", "股票权重重合"], [
        [p["left"], p["right"], p["status"], p["common_holdings"], ", ".join(p["only_left"] or []) if p["only_left"] is not None else None,
         ", ".join(p["only_right"] or []) if p["only_right"] is not None else None,
         ", ".join(p["quantity_differences"] or []) if p["quantity_differences"] is not None else None,
         _percent(p["stock_weight_overlap"])] for p in report["pairs"]])
    lines += ["", "每股数量、市值及 NAV 权重见 holdings CSV；现金单列。", "", "## 来源与缺口 / Sources and gaps", ""]
    for member in report["members"]:
        lines += ["### " + _display(member["label"]), ""]
        next_signal = (member["current"] or {}).get("next_signal", {})
        lines.append("- 当前下一期 / current next signal: " + _source_text(next_signal))
        sources = member["sources"] or {}
        if sources.get("bootstrap"):
            lines.append("- 起点来源 / bootstrap: %s; %s" % (
                sources["bootstrap"]["date"], _display(sources["bootstrap"]["path"])))
        titles = [("期间订单建议来源" if member["kind"] == "actual" else "已结算计划来源", sources.get("settled_period", [])),
                  ("当前待执行来源 / current pending", (member["current"] or {}).get("suggestions", [])
                   if member["kind"] == "actual" else sources.get("pending_current", []))]
        for title, entries in titles:
            lines.append("- %s: %s" % (title, "未记录 / not recorded" if not entries else ""))
            for entry in entries:
                source = entry["source"]
                lines.append("  - %s → %s; %s; manifest=%s; selection=%s" % (
                    entry.get("anchor"), entry["trade_date"], _source_text(source), _display(entry.get("manifest")),
                    _display(entry.get("selection_status", "frozen_plan"))))
                for artifact in entry.get("artifact_facts", []):
                    lines.append("    - file=%s; %s" % (_display(artifact["path"]), artifact["status"]))
        for gap in member.get("source_gaps", []):
            lines.append("- 来源缺口 / source gap: %s; %s" % (_display(gap["path"]), _display(gap["detail"])))
        for gap in member["gaps"]:
            lines.append("- %s: %s" % (gap["code"], _display(gap["detail"])))
        for error in member["errors"]:
            lines.append("- %s: %s" % (error["type"], _display(error["message"])))
        if member["gaps"]:
            lines.append("- Available recorded dates: " + ", ".join(member["available_dates"]))
        lines.extend("- " + _display(note) for note in member["notes"])
        lines.append("")
    lines += _explanation_markdown(report)
    lines.extend("- " + note for note in report["notes"])
    return "\n".join(lines) + "\n"


def _csv(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ";".join(value) if isinstance(value, list) else value for key, value in row.items()})
    return stream.getvalue().encode("utf-8-sig")


@cash_context
def _explanation_csvs(stem, report):
    assets, executions = [], []
    for pair in report["explanations"]["asset_differences"]:
        common = {key: pair[key] for key in ("left", "right", "date", "direction", "status", "nav_difference", "cash_difference", "stock_difference_total", "detail_total", "rounding_residual")}
        common.update(left_date=pair.get("left_date"), right_date=pair.get("right_date"))
        assets.extend(dict(common, **row) for row in pair["stocks"])
        if not pair["stocks"]:
            assets.append(common)
    for row in report["explanations"]["executions"]:
        for member in row["members"]:
            fills = [f for f in member["executions"] if f["status"] == "FILLED"]
            quantity = sum((decimal(f["quantity"]) for f in fills), ZERO) if member["executions"] else None
            gross = sum((decimal(f["gross"]) for f in fills), ZERO) if fills else None
            fees = sum((decimal(f["fees"]) for f in fills), ZERO) if fills else None
            suggestion = member["suggestions"][0] if len(member["suggestions"]) == 1 else {}
            executions.append(dict(date=row["date"], instrument=row["instrument"], side=row["side"], member_id=member["member_id"],
                status=member["status"], suggestion_selection=member["suggestion_selection"],
                suggested_quantity=suggestion.get("suggested_quantity"), role=suggestion.get("role"), position=suggestion.get("position"),
                recorded_quantity=None if quantity is None else format(quantity, "f"), gross=None if gross is None else amount(gross),
                fees=None if fees is None else amount(fees), average_price=str(gross / quantity) if quantity else None,
                fill_count=sum(f["fill_count"] for f in fills) if member["executions"] else None,
                skip_reasons=";".join(f["reason"] for f in member["executions"] if f["status"] == "SKIPPED"),
                suggestions=json.dumps(member["suggestions"], ensure_ascii=False, sort_keys=True),
                executions=json.dumps(member["executions"], ensure_ascii=False, sort_keys=True)))
    return {stem + "_asset_differences.csv": _csv(assets, ["left", "right", "date", "left_date", "right_date", "direction", "status", "nav_difference", "cash_difference",
        "stock_difference_total", "detail_total", "rounding_residual", "instrument", "left_quantity", "right_quantity", "quantity_difference",
        "left_market_value", "right_market_value", "market_value_difference", "left_weight", "right_weight"]),
        stem + "_executions.csv": _csv(executions, ["date", "instrument", "side", "member_id", "status", "suggestion_selection",
            "suggested_quantity", "role", "position", "recorded_quantity", "gross", "fees", "average_price", "fill_count", "skip_reasons",
            "suggestions", "executions"])}


def save_account_comparison(group, report, output):
    """Write only the explicitly selected report directory, outside all source roots."""
    output = Path(output).expanduser().resolve()
    if [m["id"] for m in report["members"]] != [m.id for m in group.members]:
        raise AccountComparisonError("report members differ from requested group")
    protected = [root.resolve() for m in group.members for root in (m.ctx.root, m.ctx.qlib_data_dir)]
    protected.extend(Path(root).resolve() for member in report["members"] for root in member["protected_roots"])

    def checked(path):
        resolved = path.resolve()
        for root in protected:
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            raise AccountComparisonError("report output is inside an account workspace or provider")
        try:
            resolved.relative_to(output)
        except ValueError as exc:
            raise AccountComparisonError("report path leaves selected output directory") from exc
        if resolved == group.path:
            raise AccountComparisonError("report would overwrite its group configuration")
        return resolved

    checked(output)
    if _sha(_stable_read(group.path.parent, group.path.name)) != group.config_sha256:
        raise AccountComparisonError("group configuration changed during comparison")
    stem = "comparison_%s_%s" % (report["since"], report["date"])
    summaries, holdings = summary_rows(report), holding_rows(report)
    pair_fields = ["left", "right", "status", "common_holdings", "only_left", "only_right", "stock_weight_overlap", "quantity_differences"]
    payloads = {
        stem + ".json": (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
        stem + ".md": comparison_markdown(report).encode("utf-8"),
        stem + "_summary.csv": _csv(summaries, list(summaries[0])),
        stem + "_holdings.csv": _csv(holdings, ["instrument", "member_id", "difference", "present_members", "absent_members", "unknown_members",
                                             "quantity_differs", "status", "quantity", "market_value", "weight"]),
        stem + "_pairs.csv": _csv(report["pairs"], pair_fields),
    }
    if "explanations" in report:
        payloads.update(_explanation_csvs(stem, report))
    # Check every final path before the first write, including existing symlinks.
    paths = {name: checked(output / name) for name in payloads}
    checked(output).mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = checked(output / name)
        atomic_write_bytes(path, payload)
        if checked(output / name) != paths[name] or path.read_bytes() != payload:
            raise AccountComparisonError("report commit post-state is uncertain")
    return {name: str(path) for name, path in paths.items()}
