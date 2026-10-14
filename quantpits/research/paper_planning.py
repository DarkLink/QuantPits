"""Freeze the production position analysis before suggestion-size filtering."""
from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from quantpits.evidence.contracts import finite_number
from quantpits.evidence.ranking import RankingResult, canonical_full_ranking
from quantpits.order.execution import UniverseSnapshot
from quantpits.research.paper_accounting import (
    PaperError, amount, date, decimal, execution_config, integer, money, state, token, validate_rules,
)
from quantpits.utils.strategy import TopkDropoutOrderGenerator
from quantpits.research.paper_market import verify_closes


def strategy_config(raw):
    if not isinstance(raw, dict) or raw.get("name") != "topk_dropout":
        raise PaperError("weekly paper currently supports topk_dropout")
    result = {"name": raw["name"]}
    params = raw["params"]
    for key, minimum in (("topk", 1), ("n_drop", 0), ("buy_suggestion_factor", 1)):
        result[key] = integer(params[key], key, minimum)
    result["sell_out_of_universe"] = params.get("sell_out_of_universe", True)
    if type(result["sell_out_of_universe"]) is not bool:
        raise PaperError("sell_out_of_universe must be boolean")
    return result


def make_plan(*, account_id, cycle_id, anchor, trade_date, generated_at, prior,
              market, universe, scores, closes, strategy, execution, source,
              market_rules, calendar_status, predecessor_record_count):
    token(account_id)
    token(cycle_id)
    integer(predecessor_record_count, "predecessor_record_count")
    date(anchor)
    date(trade_date)
    current = state(prior)
    if current.portfolio_id != account_id or not current.as_of_date <= anchor < trade_date:
        raise PaperError("plan account/date/predecessor mismatch")
    universe_snapshot = UniverseSnapshot.observe(market, anchor, universe)
    eligible = set(universe_snapshot.instruments)
    allowed_scores = {i: v for i, v in scores.items() if i in eligible}
    ranking = canonical_full_ranking(eligible, allowed_scores)
    scored = [r for r in ranking.rows if r["scored"]]
    if not scored:
        raise PaperError("anchor has no usable predictions")
    index = pd.MultiIndex.from_tuples([(pd.Timestamp(anchor), r["instrument"]) for r in scored],
                                      names=["datetime", "instrument"])
    predictions = pd.DataFrame({"score": [float(r["raw_score"]) for r in scored]}, index=index)
    price_rows = []
    for instrument in sorted({r["instrument"] for r in scored} | {p.instrument for p in current.positions}):
        close = closes.get(instrument)
        value = float(close) if close is not None else float("nan")
        price_rows.append({"instrument": instrument, "current_close": value,
                           "possible_min": value * 0.9, "possible_max": value * 1.1})
    prices = pd.DataFrame(price_rows).set_index("instrument")
    holdings = [{"instrument": p.instrument, "value": p.quantity} for p in current.positions]
    generator = TopkDropoutOrderGenerator(**{k: v for k, v in strategy.items() if k != "name"})
    analysis = generator.analyze_positions_with_universe(
        predictions, prices, holdings, universe_snapshot if strategy["sell_out_of_universe"] else None,
    )
    sell_candidates = pd.concat([analysis.forced_exit_candidates, analysis.normal_sell_candidates])
    sells, estimated_sell = generator.generate_sell_orders(sell_candidates, holdings, trade_date)
    suggestions = generator.generate_buy_orders(
        analysis.buy_candidates, analysis.target_buy_count, float(current.cash) + estimated_sell, trade_date,
    )
    by_instrument = {s["instrument"]: s for s in suggestions}
    buy_rows = []
    for instrument, row in analysis.buy_candidates.iterrows():
        suggestion = by_instrument.get(instrument)
        buy_rows.append({"instrument": instrument,
                         "role": "PRIMARY" if len(buy_rows) < analysis.target_buy_count else "BACKUP",
                         "score": str(float(row["score"])), "anchor_close": amount(row["current_close"]),
                         "suggested_quantity": 0 if suggestion is None else suggestion["value"],
                         "suggested_amount": "0.00" if suggestion is None else amount(suggestion["estimated_amount"])})
    sell_rows = [{"instrument": s["instrument"], "suggested_quantity": s["value"],
                  "suggested_amount": amount(s["estimated_amount"]), "anchor_close": amount(s["current_close"])}
                 for s in sells]
    # Persist inputs and decisions for offline consistency checks, without rerunning selection.
    normalized_scores = {}
    for instrument, value in scores.items():
        normalized_scores[instrument] = float(value) if finite_number(value) else None
    return {"schema_version": 1, "account_id": account_id, "cycle_id": cycle_id,
            "anchor": anchor, "trade_date": trade_date, "generated_at": generated_at,
            "prior": current.to_dict(), "market": market,
            "predecessor_record_count": predecessor_record_count,
            "universe": list(universe_snapshot.instruments), "scores": normalized_scores,
            "anchor_closes": closes, "strategy": strategy, "execution": execution_config(execution),
            "source": source, "market_rules": market_rules, "calendar_status": calendar_status,
            "ranking": [dict(r) for r in ranking.rows], "sells": sell_rows, "buys": buy_rows,
            "target_buys": analysis.target_buy_count,
            "pending_forced_exits": list(analysis.pending_forced_exit_instruments),
            "eligible_unscored_holdings": list(analysis.eligible_unscored_instruments),
            "ignored_signal_instruments": sorted(set(scores) - eligible),
            "estimated_sell_proceeds": amount(estimated_sell)}


def _instruments(rows, field):
    if not isinstance(rows, list) or any(
        not isinstance(i, str) or not re.fullmatch(r"(?:SH|SZ)[0-9]{6}", i) for i in rows
    ) or len(set(rows)) != len(rows):
        raise PaperError("%s must contain unique instruments" % field)
    return set(rows)


def _suggested_money(value):
    if not isinstance(value, str):
        raise PaperError("suggested amounts must be decimal strings")
    parsed = decimal(value)
    if parsed < 0 or parsed != money(parsed):
        raise PaperError("suggested amounts must be nonnegative cent amounts")


def validate_plan(plan):
    """Validate a frozen decision, independently of today's production planner.

    Suggestions are historical budget references. Their numerical relationship
    to quantity or anchor close is not a condition for reading/executing them.
    Actual cash/fees/positions are still checked by the journal's accounting replay.
    """
    fields = {
        "schema_version", "account_id", "cycle_id", "anchor", "trade_date", "generated_at",
        "prior", "market", "predecessor_record_count", "universe", "scores", "anchor_closes",
        "strategy", "execution", "source", "market_rules", "calendar_status", "ranking",
        "sells", "buys", "target_buys", "pending_forced_exits", "eligible_unscored_holdings",
        "ignored_signal_instruments", "estimated_sell_proceeds",
    }
    if not isinstance(plan, dict) or set(plan) != fields or type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise PaperError("persisted plan schema mismatch")
    token(plan["account_id"])
    token(plan["cycle_id"])
    token(plan["market"])
    integer(plan["predecessor_record_count"], "predecessor_record_count")
    current = state(plan["prior"])
    if current.portfolio_id != plan["account_id"] or not current.as_of_date <= date(plan["anchor"]) < date(plan["trade_date"]):
        raise PaperError("plan account/date/predecessor mismatch")
    if not isinstance(plan["generated_at"], str):
        raise PaperError("invalid plan generation time")
    created = datetime.fromisoformat(plan["generated_at"].replace("Z", "+00:00"))
    if created.tzinfo is None:
        raise PaperError("plan generation time requires a timezone")
    if plan["calendar_status"] not in ("VERIFIED", "PENDING_CALENDAR"):
        raise PaperError("invalid plan calendar status")
    strategy = strategy_config({"name": plan["strategy"]["name"], "params": plan["strategy"]})
    if strategy != plan["strategy"]:
        raise PaperError("persisted strategy configuration mismatch")
    execution = execution_config(plan["execution"])
    if execution != plan["execution"]:
        raise PaperError("persisted execution configuration mismatch")
    validate_rules(plan["market_rules"], plan["trade_date"])
    eligible = _instruments(plan["universe"], "universe")
    if not eligible or plan["universe"] != sorted(eligible):
        raise PaperError("plan universe must be nonempty and sorted")
    held = {p.instrument: p.quantity for p in current.positions}
    if not isinstance(plan["anchor_closes"], dict) or set(plan["anchor_closes"]) != eligible | set(held):
        raise PaperError("anchor price scope must equal universe and holdings")
    source = plan["source"]
    if not isinstance(source, dict) or source["kind"] not in ("FILE", "MLFLOW") or source["anchor"] != plan["anchor"]:
        raise PaperError("frozen signal source identity mismatch")
    for field in ("workspace", "path", "sha256"):
        if not isinstance(source[field], str) or not source[field].strip():
            raise PaperError("invalid frozen signal source field")
    if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) or not isinstance(source["requested_source"], dict):
        raise PaperError("invalid frozen signal source hash/request")
    if source["kind"] == "MLFLOW":
        token(source["recorder_id"])
        token(str(source["experiment_id"]))
        if not isinstance(source["mlflow_uri"], str) or not source["mlflow_uri"].strip() or (
            source["requested_source"].get("recorder_id") != source["recorder_id"]
            or str(source["requested_source"].get("experiment_id")) != source["experiment_id"]
        ):
            raise PaperError("frozen signal recorder request mismatch")
        metadata = source["metadata"]
        if (metadata["run_id"], metadata["experiment_id"], metadata["artifact_uri"], metadata["status"], metadata["lifecycle_stage"]) != (
            source["recorder_id"], source["experiment_id"], source["artifact_uri"], "FINISHED", "active",
        ):
            raise PaperError("frozen signal recorder metadata mismatch")
    elif not isinstance(source["requested_source"].get("prediction"), str) or not source["requested_source"]["prediction"]:
        raise PaperError("invalid frozen prediction file request")
    verify_closes(source["price_receipts"], plan["anchor"], plan["anchor_closes"])
    scores = plan["scores"]
    if not isinstance(scores, dict) or any(
        not isinstance(i, str) or not i or (v is not None and not finite_number(v)) for i, v in scores.items()
    ) or not isinstance(plan["ranking"], list):
        raise PaperError("invalid frozen scores/ranking")
    scored_count = sum(r["scored"] is True for r in plan["ranking"])
    ranking = RankingResult(tuple(plan["ranking"]), len(eligible), scored_count, len(eligible) - scored_count,
                            scored_count == len(eligible))
    if {r["instrument"] for r in ranking.rows} != eligible or not scored_count:
        raise PaperError("ranking must cover the frozen universe with usable scores")
    ranked = {r["instrument"]: r for r in ranking.rows}
    for instrument, row in ranked.items():
        score = scores.get(instrument)
        if row["scored"] != (score is not None) or (row["scored"] and row["raw_score"] != format(float(score), ".17g")):
            raise PaperError("ranking differs from frozen source scores")
        if not row["scored"] and row["coverage_status"] != ("invalid_prediction" if instrument in scores else "missing_prediction"):
            raise PaperError("ranking missing-score status mismatch")
    if plan["ignored_signal_instruments"] != sorted(set(scores) - eligible):
        raise PaperError("ignored signal scope mismatch")
    target = integer(plan["target_buys"], "target_buys")
    for side, fields in (("sells", {"instrument", "suggested_quantity", "suggested_amount", "anchor_close"}),
                         ("buys", {"instrument", "role", "score", "anchor_close", "suggested_quantity", "suggested_amount"})):
        rows = plan[side]
        if not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != fields for r in rows):
            raise PaperError("invalid frozen %s rows" % side)
        instruments = _instruments([r["instrument"] for r in rows], side)
        if not instruments <= (set(held) if side == "sells" else eligible - set(held)):
            raise PaperError("frozen candidate does not belong to its account/universe")
        for index, row in enumerate(rows):
            quantity = integer(row["suggested_quantity"], "suggested_quantity", 1 if side == "sells" else 0)
            _suggested_money(row["suggested_amount"])
            close = plan["anchor_closes"].get(row["instrument"])
            if close is None or row["anchor_close"] != close:
                raise PaperError("candidate anchor price differs from frozen observation")
            if side == "sells" and quantity > held[row["instrument"]]:
                raise PaperError("planned sell exceeds prior holding")
            if side == "buys":
                score = ranked[row["instrument"]]
                if quantity % execution["lot_size"] or not score["scored"] or float(decimal(row["score"])) != float(score["raw_score"]):
                    raise PaperError("invalid buy lot/source score")
                if row["role"] != ("PRIMARY" if index < target else "BACKUP"):
                    raise PaperError("frozen primary/backup role mismatch")
    exits = sum(r["suggested_quantity"] == held[r["instrument"]] for r in plan["sells"])
    if target > max(0, strategy["topk"] - len(held) + exits) or len(plan["buys"]) > target * strategy["buy_suggestion_factor"]:
        raise PaperError("frozen buy target/candidate count exceeds capacity")
    diagnostics = set()
    for field in ("pending_forced_exits", "eligible_unscored_holdings"):
        instruments = _instruments(plan[field], field)
        if not instruments <= set(held) or instruments & (diagnostics | {r["instrument"] for r in plan["sells"]}):
            raise PaperError("holding diagnostics overlap or contain foreign holdings")
        diagnostics.update(instruments)
    if set(plan["pending_forced_exits"]) & eligible:
        raise PaperError("pending forced exits must be outside the anchor universe")
    _suggested_money(plan["estimated_sell_proceeds"])
