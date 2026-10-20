"""Human/CSV views and explicitly scoped persistence of account comparisons."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from quantpits.research.account_comparison import AccountComparisonError, _sha, _stable_read, _comparable
from quantpits.research.paper_accounting import decimal, cash_context
from quantpits.training.persistence import atomic_write_bytes


def _display(value):
    return "—" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def _percent(value):
    return "—" if value is None else format(decimal(value) * 100, ".4f") + "%"


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
        quantities = {decimal(available[member_id][instrument]["quantity"]) for member_id in present}
        difference = ("incomplete" if len(available) != len(report["members"]) else "only_one_account" if len(present) == 1
                      else "quantity_differs" if len(quantities) > 1 else "shared_same_quantity")
        for member in report["members"]:
            positions = available.get(member["id"])
            position = positions.get(instrument) if positions is not None else None
            rows.append({"instrument": instrument, "member_id": member["id"], "difference": difference,
                         "status": "unavailable" if positions is None else "present" if position else "absent",
                         "quantity": position["quantity"] if position else "0" if positions is not None else None,
                         "market_value": position["market_value"] if position else "0.00" if positions is not None else None,
                         "weight": position["weight"] if position else "0" if positions is not None else None})
    return rows


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
        lines.append("- 当前下一期 / current next signal: anchor=%s; workspace=%s" % (
            _display(next_signal.get("anchor")), _display(next_signal.get("workspace"))))
        sources = member["sources"] or {}
        if sources.get("bootstrap"):
            lines.append("- 起点来源 / bootstrap: %s; %s" % (
                sources["bootstrap"]["date"], _display(sources["bootstrap"]["path"])))
        for title, key in (("所选期间已结算 / suggested source for actual", "settled_period"), ("当前待执行来源 / current pending", "pending_current")):
            entries = sources.get(key, [])
            lines.append("- %s: %s" % (title, "未记录 / not recorded" if not entries else ""))
            for entry in entries:
                source = entry["source"]
                lines.append("  - %s → %s; workspace=%s; file/recorder=%s; manifest=%s" % (
                    entry.get("anchor"), entry["trade_date"], _display(source.get("workspace")),
                    _display(source.get("path", source.get("recorder_id", source.get("reference")))), _display(entry.get("manifest"))))
        for gap in member["gaps"]:
            lines.append("- %s: %s" % (gap["code"], _display(gap["detail"])))
        for error in member["errors"]:
            lines.append("- %s: %s" % (error["type"], _display(error["message"])))
        if member["gaps"]:
            lines.append("- Available recorded dates: " + ", ".join(member["available_dates"]))
        lines.extend("- " + _display(note) for note in member["notes"])
        lines.append("")
    lines.extend("- " + note for note in report["notes"])
    return "\n".join(lines) + "\n"


def _csv(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ";".join(value) if isinstance(value, list) else value for key, value in row.items()})
    return stream.getvalue().encode("utf-8-sig")


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
        stem + "_holdings.csv": _csv(holdings, ["instrument", "member_id", "difference", "status", "quantity", "market_value", "weight"]),
        stem + "_pairs.csv": _csv(report["pairs"], pair_fields),
    }
    # Check every final path before the first write, including existing symlinks.
    paths = {name: checked(output / name) for name in payloads}
    checked(output).mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = checked(output / name)
        atomic_write_bytes(path, payload)
        if checked(output / name) != paths[name] or path.read_bytes() != payload:
            raise AccountComparisonError("report commit post-state is uncertain")
    return {name: str(path) for name, path in paths.items()}
