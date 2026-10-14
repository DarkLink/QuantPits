"""Freeze the production position analysis before suggestion-size filtering."""
from __future__ import annotations

import pandas as pd

from quantpits.evidence.contracts import finite_number
from quantpits.evidence.ranking import canonical_full_ranking
from quantpits.order.execution import UniverseSnapshot
from quantpits.research.paper_accounting import (
    PaperError, amount, date, execution_config, integer, state, token,
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
    # Persist inputs as well as decisions so offline readers can recompute the plan.
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


def validate_plan(plan):
    strategy = strategy_config({"name": plan["strategy"]["name"], "params": plan["strategy"]})
    if strategy != plan["strategy"]:
        raise PaperError("persisted strategy configuration mismatch")
    verify_closes(plan["source"]["price_receipts"], plan["anchor"], plan["anchor_closes"])
    expected = make_plan(
        account_id=plan["account_id"], cycle_id=plan["cycle_id"], anchor=plan["anchor"],
        trade_date=plan["trade_date"], generated_at=plan["generated_at"], prior=plan["prior"],
        market=plan["market"], universe=plan["universe"], scores=plan["scores"],
        closes=plan["anchor_closes"], strategy=plan["strategy"], execution=plan["execution"],
        source=plan["source"], market_rules=plan["market_rules"], calendar_status=plan["calendar_status"],
        predecessor_record_count=plan["predecessor_record_count"],
    )
    if expected != plan:
        raise PaperError("persisted plan is inconsistent with production analysis")
