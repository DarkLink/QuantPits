"""Cash-open quantity preparation; no account mutation or sell simulation."""
from __future__ import annotations

from decimal import ROUND_DOWN

from quantpits.order.candidates import holdings, validate_candidates
from quantpits.research.paper_accounting import (
    CENT, ZERO, PaperError, affordable_buy_quantity, amount, cash_context, date, decimal,
    execution_config, fees, money, trade_check, validate_rules,
)
from quantpits.utils.workspace import fingerprint_value


@cash_context
def prepare_orders(candidates, observation):
    validate_candidates(candidates)
    if not isinstance(observation, dict) or set(observation) != {
        "schema_version", "kind", "trade_date", "price_basis", "cash", "holdings",
        "quotes", "market_rules", "execution",
    } or type(observation["schema_version"]) is not int or observation["schema_version"] != 1:
        raise PaperError("invalid cash-open observation format")
    if observation["kind"] != "cash_open_observation" or observation["price_basis"] != "CASH_OPEN":
        raise PaperError("explicit CASH_OPEN price basis required")
    selected = date(observation["trade_date"])
    if selected != candidates["trade_date"]:
        raise PaperError("observation date differs from candidate execution date")
    cash = decimal(observation["cash"], "available cash")
    if cash != money(cash):
        raise PaperError("available cash must use cent precision")
    initial_cash = cash
    held = holdings(observation["holdings"])
    quotes, rules = observation["quotes"], observation["market_rules"]
    if not isinstance(quotes, dict) or any(not isinstance(row, dict) for row in quotes.values()):
        raise PaperError("quotes must map instruments to cash-open observations")
    for row in quotes.values():
        if "date" in row and row["date"] != selected:
            raise PaperError("quote date differs from execution date")
    validate_rules(rules, selected)
    config = execution_config(observation["execution"])
    slots = min(candidates["target_buy_count"], max(0, candidates["strategy"]["params"]["topk"] - len(held)))
    reserve = decimal(config["cash_reserve"])
    spendable = max(ZERO, cash - reserve)
    budget = (spendable / slots).quantize(CENT, rounding=ROUND_DOWN) if slots else ZERO
    decisions, prepared = [], 0
    for row in candidates["buys"]:
        instrument = row["instrument"]
        reason, price, limits = None, None, None
        if instrument in held:
            reason = "ALREADY_HELD"
        elif prepared == slots:
            reason = "NO_SLOT"
        else:
            try:
                reason, price, limits = trade_check(instrument, quotes.get(instrument, {}), rules, selected)
            except PaperError as exc:
                reason = "INVALID_MARKET_OBSERVATION: " + str(exc)
        quantity = 0
        if reason is None:
            quantity = affordable_buy_quantity(min(budget, max(ZERO, cash - reserve)), price, instrument, config)
            if quantity == 0:
                reason = "INSUFFICIENT_SLOT_BUDGET"
        gross = money(price * quantity) if quantity else ZERO
        fee = fees(gross, "BUY", instrument, config) if quantity else None
        cost = gross + decimal(fee["total"]) if fee else ZERO
        cash -= cost
        prepared += bool(quantity)
        maximum = config["observed_max_gross"]
        decisions.append(dict(instrument=instrument, role=row["role"], score=row["score"],
            suggested_quantity=row["suggested_quantity"], status="PREPARED" if quantity else "SKIPPED",
            reason=reason, quantity=quantity, price=None if price is None else amount(price), limits=limits,
            gross=amount(gross), fees=fee, estimated_cost=amount(cost), budget=amount(budget),
            fee_estimate_outside_observed_scale=maximum is not None and gross > decimal(maximum)))
    pending = [dict(row, currently_held_quantity=held[row["instrument"]])
               for row in candidates["sell_decisions"]
               if row["instrument"] in held and row.get("status") != "RETAINED"]
    return dict(schema_version=1, kind="order_preparation", candidate_digest=candidates["digest"],
        observation_digest=fingerprint_value(observation), trade_date=selected, price_basis="CASH_OPEN",
        execution_assumption="CURRENT_CASH_NO_SELL_SIMULATION", source=candidates["source"],
        current_cash=amount(initial_cash), slots=slots, slot_budget=amount(budget),
        cash_reserve=amount(reserve), estimated_remaining_cash=amount(cash),
        estimated_purchase_cost=amount(initial_cash - cash),
        estimated_fees=amount(sum((decimal(row["fees"]["total"]) for row in decisions if row["fees"]), ZERO)),
        pending_sells=pending, decisions=decisions,
        notice="Prepared quantities and assumed fees; no trades executed and no account changed.")
