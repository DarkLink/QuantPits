"""Complete, ordered order inputs, independent of estimated buy quantities."""
from __future__ import annotations

import re
from pathlib import Path

from quantpits.research.paper_accounting import amount, date, decimal, integer
from quantpits.utils.workspace import fingerprint_value


def output_path(ctx, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ctx.root / path
    path = path.resolve()
    try:
        path.relative_to(ctx.root.resolve())
        # Resolve the candidate, not the authorized boundary: an output/ link
        # into config/ must not turn that configuration directory into output.
        path.relative_to(ctx.output_dir)
    except ValueError as exc:
        raise ValueError("derived order output must stay inside workspace/output") from exc
    if path.suffix != ".json" or path.name.startswith(("buy_suggestion", "sell_suggestion", "model_opinions")):
        raise ValueError("derived order output requires a distinct JSON filename")
    return path


def holdings(rows):
    if not isinstance(rows, list):
        raise ValueError("holdings must be an array")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"instrument", "quantity"}:
            raise ValueError("holdings require instrument/quantity")
        instrument = security(row["instrument"])
        if instrument in result:
            raise ValueError("duplicate holding")
        result[instrument] = integer(row["quantity"], "holding quantity", 1)
    return result


def security(value):
    if not isinstance(value, str) or not re.fullmatch(r"(?:SH|SZ)[0-9]{6}", value):
        raise ValueError("invalid instrument")
    return value


def build_candidates(*, ctx, loaded, anchor, trade_date, strategy, analysis, current_holding,
                     buy_orders, sell_orders, sell_decisions, signal_digest):
    by_instrument = {row["instrument"]: row for row in buy_orders}
    buys = []
    for instrument, row in analysis.buy_candidates.iterrows():
        old = by_instrument.get(instrument, {})
        buys.append(dict(instrument=instrument, role="PRIMARY" if len(buys) < analysis.target_buy_count else "BACKUP",
                         score=str(float(row["score"])), anchor_close=amount(row["current_close"]),
                         possible_min=amount(row["possible_min"]), possible_max=amount(row["possible_max"]),
                         suggested_quantity=old.get("value", 0),
                         suggested_amount=amount(old.get("estimated_amount", 0))))
    positions = []
    for row in current_holding:
        quantity = decimal(row["value"])
        if quantity != int(quantity):
            raise ValueError("holding quantity must be integral")
        positions.append(dict(instrument=row["instrument"], quantity=int(quantity)))
    source = loaded.source
    result = dict(schema_version=1, kind="order_candidates", workspace=str(ctx.root),
        source=dict(mode=source.mode, resolved_name=source.resolved_name, record_id=source.record_id,
                    experiment_name=source.experiment_name, description=loaded.description,
                    signal_digest=source.prediction_sha256 or signal_digest),
        anchor=anchor, trade_date=trade_date, strategy=strategy,
        target_buy_count=analysis.target_buy_count, holdings=positions, buys=buys,
        sells=[dict(instrument=row["instrument"], suggested_quantity=row["value"],
                    suggested_amount=amount(row["estimated_amount"]), anchor_close=amount(row["current_close"]))
               for row in sell_orders],
        sell_decisions=[dict(row) for row in sell_decisions])
    result["digest"] = fingerprint_value(result)
    validate_candidates(result)
    return result


def validate_candidates(value):
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "kind", "workspace", "source", "anchor", "trade_date", "strategy",
        "target_buy_count", "holdings", "buys", "sells", "sell_decisions", "digest",
    } or type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["kind"] != "order_candidates":
        raise ValueError("invalid order candidates format")
    if value["digest"] != fingerprint_value({k: v for k, v in value.items() if k != "digest"}):
        raise ValueError("order candidates digest mismatch")
    if date(value["anchor"]) >= date(value["trade_date"]):
        raise ValueError("candidate execution must follow anchor")
    strategy = value["strategy"]
    if not isinstance(strategy, dict) or strategy.get("name") != "topk_dropout" or not isinstance(strategy.get("params"), dict):
        raise ValueError("only topk_dropout candidates are supported")
    topk = integer(strategy["params"].get("topk"), "topk", 1)
    integer(strategy["params"].get("n_drop", 3), "n_drop")
    factor = integer(strategy["params"].get("buy_suggestion_factor", 2), "buy_suggestion_factor", 1)
    target = integer(value["target_buy_count"], "target_buy_count")
    if target > topk:
        raise ValueError("candidate target exceeds topk")
    held = holdings(value["holdings"])
    source = value["source"]
    if (not isinstance(value["workspace"], str) or not value["workspace"] or not isinstance(source, dict)
            or not isinstance(source.get("signal_digest"), str)
            or re.fullmatch(r"[a-f0-9]{64}", source["signal_digest"]) is None):
        raise ValueError("candidate workspace/source identity is missing")
    if (set(source) != {"mode", "resolved_name", "record_id", "experiment_name", "description", "signal_digest"}
            or source["mode"] not in {"model", "ensemble"}
            or any(not isinstance(source.get(field), str) or not source[field]
                   for field in ("record_id", "experiment_name", "description"))):
        raise ValueError("candidate source identity is incomplete")
    seen = set()
    if not isinstance(value["buys"], list) or not isinstance(value["sells"], list) or not isinstance(value["sell_decisions"], list):
        raise ValueError("candidate members must be arrays")
    if len(value["buys"]) > target * factor:
        raise ValueError("buy candidates exceed original target/backup range")
    for index, row in enumerate(value["buys"]):
        if not isinstance(row, dict) or set(row) != {"instrument", "role", "score", "anchor_close",
                "possible_min", "possible_max", "suggested_quantity", "suggested_amount"}:
            raise ValueError("invalid buy candidate fields")
        instrument = security(row["instrument"])
        if instrument in seen or instrument in held:
            raise ValueError("duplicate or already-held original buy candidate")
        seen.add(instrument)
        if row["role"] != ("PRIMARY" if index < target else "BACKUP"):
            raise ValueError("candidate role/order mismatch")
        decimal(row["score"], "candidate score")
        for field in ("anchor_close", "possible_min", "possible_max"):
            if decimal(row[field]) <= 0 or amount(row[field]) != row[field]:
                raise ValueError("invalid candidate anchor price")
        if integer(row["suggested_quantity"], "suggested_quantity") % 100:
            raise ValueError("suggested buy quantity must use 100-share lots")
        if (decimal(row["suggested_amount"]) < 0 or amount(row["suggested_amount"]) != row["suggested_amount"]
                or (row["suggested_quantity"] == 0) != (decimal(row["suggested_amount"]) == 0)):
            raise ValueError("negative suggested amount")
    seen = set()
    for row in value["sells"]:
        if not isinstance(row, dict) or set(row) != {"instrument", "suggested_quantity", "suggested_amount", "anchor_close"}:
            raise ValueError("invalid sell candidate fields")
        instrument = security(row["instrument"])
        if instrument in seen or instrument not in held or integer(row["suggested_quantity"], "sell quantity", 1) > held[instrument]:
            raise ValueError("sell intent contradicts original holdings")
        seen.add(instrument)
        if decimal(row["suggested_amount"]) < 0 or decimal(row["anchor_close"]) <= 0:
            raise ValueError("invalid sell estimate/anchor price")
    sell_ids = set(seen)
    seen = set()
    for row in value["sell_decisions"]:
        if (not isinstance(row, dict) or set(row) != {"instrument", "reason", "status"}
                or security(row["instrument"]) not in held or not isinstance(row.get("reason"), str)
                or row["instrument"] in seen or row["status"] not in {"ORDER_GENERATED", "PENDING", "RETAINED"}):
            raise ValueError("invalid holding classification")
        seen.add(row["instrument"])
    generated = {row["instrument"] for row in value["sell_decisions"] if row["status"] == "ORDER_GENERATED"}
    if generated != sell_ids:
        raise ValueError("sell intent/classification mismatch")
