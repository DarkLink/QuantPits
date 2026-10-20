"""Saved-date asset differences and observed execution comparisons, without replay."""
from __future__ import annotations

import hashlib
import re
from itertools import combinations

import pandas as pd

from quantpits.post_trade.state import normalize_date, normalize_instrument, normalize_settlement_frame
from quantpits.research.paper_accounting import ZERO, amount, cash_context, decimal


def _quantity(value):
    return format(decimal(value), "f")


def suggestion_digest(output, raw):
    """Verify a declared raw SHA-256; absence never proves historical file bytes."""
    expected = []
    for key in ("sha256", "fingerprint", "digest"):
        if key not in output:
            continue
        value = output[key]
        if isinstance(value, dict):
            if value.get("algorithm") != "sha256" or value.get("domain", "raw_bytes") != "raw_bytes":
                raise ValueError("unsupported suggestion digest")
            if "size_bytes" in value and (type(value["size_bytes"]) is not int or value["size_bytes"] != len(raw)):
                raise ValueError("suggestion digest byte count mismatch")
            value = value.get("value")
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid suggestion SHA-256")
        expected.append(value)
    observed = hashlib.sha256(raw).hexdigest()
    if expected and (len(set(expected)) != 1 or expected[0] != observed):
        raise ValueError("suggestion digest mismatch")
    return {"status": "recorded_digest_verified" if expected else "current_readable_no_historical_digest", "sha256": observed}


def saved_suggestions(rows, columns, side, trade_date, path, verification):
    quantity_column = next((key for key in ("value", "suggested_quantity", "quantity", "委托数量") if key in columns), None)
    if quantity_column is None:
        raise ValueError("saved suggestion quantity is not recorded")
    result = []
    for index, row in enumerate(rows, 1):
        instrument = normalize_instrument(row.get("instrument", row.get("证券代码")))
        quantity = decimal(row[quantity_column])
        if quantity < ZERO or quantity != int(quantity):
            raise ValueError("invalid saved suggestion quantity")
        if row.get("datetime") and normalize_date(row["datetime"]) != trade_date:
            raise ValueError("saved suggestion execution date differs from manifest")
        role = row.get("role") or "not_recorded"
        if role not in ("PRIMARY", "BACKUP", "not_recorded"):
            raise ValueError("invalid saved suggestion role")
        result.append({"instrument": instrument, "side": side, "suggested_quantity": _quantity(quantity),
                       "position": index, "role": role, "path": path, "verification": verification["status"]})
    return result


def _execution_result():
    return {"status": "complete", "ledger_available": True, "days": [], "suggestion_versions": [], "pending_plans": [], "gaps": []}


def _gap(execution, detail, selected=None):
    execution["status"] = "partial"
    execution["gaps"].append({"date": selected, "detail": detail})


@cash_context
def actual_executions(columns, rows, since, selected, versions, source_gaps):
    result = _execution_result()
    result["suggestion_versions"] = [v for v in versions if since < v["trade_date"] <= selected]
    required = {"交易类别", "成交价格", "成交数量", "成交金额", "资金发生数", "交收日期", "证券代码", "费用合计"}
    if columns is None or not required <= columns:
        result["ledger_available"] = False
        _gap(result, "actual settlement ledger is missing required execution columns")
        return result
    by_date = {}
    for row in rows:
        if since < row["_date"] <= selected:
            by_date.setdefault(row["_date"], []).append(row)
    for selected_date, records in sorted(by_date.items()):
        day = {"date": selected_date, "status": "complete", "executions": [], "cash_events": [], "notes": []}
        result["days"].append(day)
        try:
            events, warnings = normalize_settlement_frame(pd.DataFrame(records), selected_date)
            if warnings:
                raise ValueError("; ".join(warnings))
            grouped = {}
            for event in events:
                original = records[event.source_row - 1]
                if event.kind not in ("buy", "sell"):
                    day["cash_events"].append({"kind": event.kind, "category": event.normalized_trade_type,
                                               "cash_effect": amount(event.cash_effect), "instrument": event.instrument,
                                               "observed_quantity": original.get("成交数量"), "source_row": event.source_row})
                    continue
                if event.gross_amount < ZERO:
                    raise ValueError("negative recorded trade gross cannot form a summary price")
                key = (event.instrument, event.kind.upper())
                entry = grouped.setdefault(key, {"quantity": ZERO, "gross": ZERO, "fees": ZERO, "fill_count": 0, "source_rows": []})
                entry["quantity"] += event.quantity
                entry["gross"] += event.gross_amount
                entry["fees"] += decimal(original["费用合计"])
                entry["fill_count"] += 1
                entry["source_rows"].append(event.source_row)
            for (instrument, side), entry in sorted(grouped.items()):
                day["executions"].append({"instrument": instrument, "side": side, "status": "FILLED",
                    "quantity": _quantity(entry["quantity"]), "gross": amount(entry["gross"]), "fees": amount(entry["fees"]),
                    "average_price": str(entry["gross"] / entry["quantity"]), "price_method": "SUM_GROSS_OVER_SUM_QUANTITY",
                    "fill_count": entry["fill_count"], "source_rows": entry["source_rows"], "reason": None,
                    "reason_label": "actual order/manual selection reasons are not observed"})
        except Exception as exc:
            day.update(status="unavailable", executions=[], cash_events=[])
            _gap(result, "%s: %s" % (type(exc).__name__, exc), selected_date)
    for gap in source_gaps:
        if gap.get("date") is None or since < gap["date"] <= selected:
            _gap(result, "suggestion/source gap: " + gap["detail"], gap.get("date"))
    return result


def paper_executions(journal, since, selected, sources, source_gaps):
    result = _execution_result()
    source_by_cycle = {s["cycle_id"]: s for s in sources["settled_period"] + sources["pending_current"]}
    for record in journal["records"]:
        execution = record["result"]
        execution_date = execution["after"]["as_of_date"]
        if record["kind"] != "SETTLEMENT" or not since < execution_date <= selected:
            continue
        cycle = execution["cycle_id"]
        plan = journal["plans"][cycle]
        source = source_by_cycle[cycle]
        suggestions = []
        for side, key in (("BUY", "buys"), ("SELL", "sells")):
            for position, row in enumerate(plan[key], 1):
                suggestions.append({"instrument": row["instrument"], "side": side,
                    "suggested_quantity": _quantity(row["suggested_quantity"]), "role": row.get("role", "not_recorded"),
                    "position": position, "verification": "validated_frozen_plan"})
        result["suggestion_versions"].append(dict(source, version_id=cycle, suggestions=suggestions,
                                                  selection_status="bound_to_validated_settlement"))
        day = {"date": execution_date, "status": "complete", "executions": [], "cash_events": [], "notes": []}
        for row in execution["decisions"]:
            reason = row["reason"]
            label = "unused candidate / no slot; not an order rejection" if reason == "NO_SLOT" else reason
            day["executions"].append({"instrument": row["instrument"], "side": row["side"], "status": row["status"],
                "quantity": _quantity(row["quantity"]), "gross": row["gross"],
                "fees": row["fees"]["total"] if row["fees"] is not None else None,
                "average_price": row["price"] if row["status"] == "FILLED" else None,
                "observed_price": row["price"], "price_method": "RECORDED_SIMULATED_EXECUTION_PRICE" if row["status"] == "FILLED" else "RECORDED_DECISION_QUOTE",
                "suggested_quantity": _quantity(row["suggested_quantity"]), "fill_count": 1 if row["status"] == "FILLED" else 0,
                "reason": reason, "reason_label": label, "cycle_id": cycle})
        result["days"].append(day)
    for record in journal["records"]:
        selected_date = record["result"]["after"]["as_of_date"]
        if since < selected_date <= selected:
            events = record["result"]["events"]
            if events:
                result.setdefault("cash_events", []).extend(dict(e, applied_state_date=selected_date) for e in events)
    for source in sources["pending_current"]:
        plan = journal["plans"][source["cycle_id"]]
        result["pending_plans"].append(dict(source, suggestions=[
            {"instrument": row["instrument"], "side": side, "suggested_quantity": _quantity(row["suggested_quantity"]),
             "role": row.get("role", "not_recorded"), "position": position}
            for side, key in (("BUY", "buys"), ("SELL", "sells")) for position, row in enumerate(plan[key], 1)]))
    for gap in source_gaps:
        if gap.get("date") is None or since < gap["date"] <= selected:
            _gap(result, "source gap: " + gap["detail"], gap.get("date"))
    result["days"].sort(key=lambda d: d["date"])
    return result


def _reliable_end(member):
    endpoint = (member.get("period") or {}).get("end")
    return endpoint if endpoint and endpoint["holdings_available"] and endpoint["reliable_nav"] and decimal(endpoint["nav"]) > ZERO else None


@cash_context
def asset_difference(left, right, selected):
    result = {"left": left["id"], "right": right["id"], "date": selected, "direction": "LEFT_MINUS_RIGHT",
              "status": "unavailable", "nav_difference": None, "cash_difference": None, "stocks": [],
              "stock_difference_total": None, "detail_total": None, "rounding_residual": None, "period_bridge": None}
    a, b = _reliable_end(left), _reliable_end(right)
    if a is None or b is None:
        return result
    result.update(left_date=a["date"], right_date=b["date"], nav_difference=amount(decimal(a["nav"]) - decimal(b["nav"])),
                  cash_difference=amount(decimal(a["cash"]) - decimal(b["cash"])))
    zero = {"quantity": "0", "market_value": "0.00", "weight": "0"}
    for instrument in sorted(set(a["positions"]) | set(b["positions"])):
        x, y = a["positions"].get(instrument, zero), b["positions"].get(instrument, zero)
        result["stocks"].append({"instrument": instrument, "left_quantity": x["quantity"], "right_quantity": y["quantity"],
            "quantity_difference": _quantity(decimal(x["quantity"]) - decimal(y["quantity"])),
            "left_market_value": x["market_value"], "right_market_value": y["market_value"],
            "market_value_difference": amount(decimal(x["market_value"]) - decimal(y["market_value"])),
            "left_weight": x["weight"], "right_weight": y["weight"]})
    stock_total = sum((decimal(r["market_value_difference"]) for r in result["stocks"]), ZERO)
    detail_total = decimal(result["cash_difference"]) + stock_total
    residual = decimal(result["nav_difference"]) - detail_total
    result.update(stock_difference_total=amount(stock_total), detail_total=amount(detail_total), rounding_residual=amount(residual),
                  status="complete" if abs(residual) <= decimal("0.02") else "unreconciled")
    periods = left["period"], right["period"]
    if all(p["start"]["reliable_nav"] and p["net_external_cashflow"] is not None and p["pnl_excluding_external"] is not None for p in periods):
        result["period_bridge"] = {
            "start_nav_difference": amount(decimal(periods[0]["start"]["nav"]) - decimal(periods[1]["start"]["nav"])),
            "net_external_cashflow_difference": amount(decimal(periods[0]["net_external_cashflow"]) - decimal(periods[1]["net_external_cashflow"])),
            "pnl_difference": amount(decimal(periods[0]["pnl_excluding_external"]) - decimal(periods[1]["pnl_excluding_external"]))}
    return result


def _fusion_key(source):
    reference = source.get("reference") or {}
    recorder = source.get("recorder_id") or reference.get("record_id")
    is_fusion = reference.get("mode") == "ensemble" or source.get("combo_name") is not None
    if not (source.get("workspace") and source.get("anchor") and recorder):
        return None, False
    return (source["workspace"], source["anchor"], recorder), is_fusion


def _execution_map(execution, selected):
    rows = [day for day in execution["days"] if day["date"] == selected]
    if not execution.get("ledger_available", True) or any(d["status"] != "complete" for d in rows):
        return None
    values = {}
    for day in rows:
        for row in day["executions"]:
            if row["status"] == "FILLED":
                key = (row["instrument"], row["side"])
                values[key] = values.get(key, ZERO) + decimal(row["quantity"])
    return values


def _source_comparison(left, right, selected):
    result = {"left": left["id"], "right": right["id"], "date": selected, "status": "unknown",
              "execution_results_differ": None}
    a, b = left.get("execution"), right.get("execution")
    if a is None or b is None:
        return result
    versions = [[v for v in e["suggestion_versions"] if v["trade_date"] == selected] for e in (a, b)]
    if any(len(v) > 1 for v in versions):
        result["status"] = "ambiguous_suggestion_versions"
    elif all(len(v) == 1 for v in versions):
        keys = [_fusion_key(v[0]["source"]) for v in versions]
        if all(k[0] is not None for k in keys):
            if keys[0][0] == keys[1][0] and any(k[1] for k in keys):
                result["status"] = "same_recorded_fusion_recorder"
                result["recorded_reference"] = list(keys[0][0])
            elif keys[0][0] != keys[1][0]:
                result["status"] = "different_recorded_references"
    maps = [_execution_map(e, selected) for e in (a, b)]
    if all(m is not None for m in maps):
        result["execution_results_differ"] = maps[0] != maps[1]
    return result


@cash_context
def explain_accounts(members, selected):
    assets = [asset_difference(left, right, selected) for left, right in combinations(members, 2)]
    dates = sorted({day["date"] for m in members for day in (m.get("execution") or {}).get("days", [])
                    if day["executions"] or day["status"] != "complete"})
    rows, sources = [], []
    for selected_date in dates:
        instruments = set()
        for member in members:
            execution = member.get("execution") or {}
            instruments.update((r["instrument"], r["side"]) for day in execution.get("days", []) if day["date"] == selected_date for r in day["executions"])
            instruments.update((r["instrument"], r["side"]) for v in execution.get("suggestion_versions", []) if v["trade_date"] == selected_date for r in v.get("suggestions", []))
        for instrument, side in sorted(instruments):
            entries = []
            for member in members:
                execution = member.get("execution")
                versions = [v for v in (execution or {}).get("suggestion_versions", []) if v["trade_date"] == selected_date]
                days = [d for d in (execution or {}).get("days", []) if d["date"] == selected_date]
                available = execution is not None and execution["ledger_available"] and all(d["status"] == "complete" for d in days)
                fills = [r for d in days for r in d["executions"] if (r["instrument"], r["side"]) == (instrument, side)]
                suggestions = [dict(r, version_id=v.get("version_id", v.get("run_id"))) for v in versions for r in v.get("suggestions", [])
                               if (r["instrument"], r["side"]) == (instrument, side)]
                entries.append({"member_id": member["id"], "status": "recorded" if fills and available else "recorded_partial" if fills else
                    "no_recorded_fill" if available and member["kind"] == "actual" else "not_in_settled_plan" if available and days else
                    "not_settled" if available else "unavailable", "suggestions": suggestions, "executions": fills,
                    "suggestion_selection": "ambiguous" if len(versions) > 1 else "not_proof_of_actual_adoption" if member["kind"] == "actual" else "frozen_plan"})
            rows.append({"date": selected_date, "instrument": instrument, "side": side, "members": entries})
        sources.extend(_source_comparison(a, b, selected_date) for a, b in combinations(members, 2))
    complete = all(a["status"] == "complete" for a in assets) and all(
        m.get("execution") is not None and m["execution"]["status"] == "complete" for m in members)
    return {"status": "complete" if complete else "partial", "asset_differences": assets, "executions": rows,
            "source_comparisons": sources,
            "notes": ["Asset differences describe selected-date values, not security-level period return contributions.",
                      "No actual cancellation, rejection, price-limit or manual-selection cause is inferred from missing fills.",
                      "A matching recorded fusion recorder is a source reference, not proof of model or signal equivalence."]}
