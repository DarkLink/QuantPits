"""Cash accounting for the weekly paper workflow; legacy B0 is unchanged."""
from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP, localcontext
from functools import wraps

from quantpits.research.accounting import ShadowPortfolioState
from quantpits.utils.workspace import fingerprint_value


class PaperError(ValueError):
    """Invalid paper input or inconsistent persisted account."""


CENT = Decimal("0.01")
ZERO = Decimal("0")


def cash_context(function):
    """Keep persisted cent amounts and return strings independent of caller context."""
    @wraps(function)
    def calculate(*args, **kwargs):
        with localcontext() as context:
            context.prec = 50
            context.rounding = ROUND_HALF_UP
            return function(*args, **kwargs)
    return calculate


def decimal(value, field="amount"):
    if isinstance(value, bool):
        raise PaperError("%s must be numeric" % field)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise PaperError("invalid %s" % field) from exc
    if not result.is_finite():
        raise PaperError("%s must be finite" % field)
    return result


@cash_context
def money(value):
    return decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def amount(value):
    return format(money(value), ".2f")


def date(value):
    if not isinstance(value, str):
        raise PaperError("date must be YYYY-MM-DD")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise PaperError("date must be YYYY-MM-DD") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise PaperError("date must be YYYY-MM-DD")
    return value


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise PaperError("invalid account/cycle/event ID")
    return value


def integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise PaperError("%s must be an integer >= %s" % (field, minimum))
    return value


def state(value):
    payload = dict(value)
    supplied = payload.pop("digest", None)
    parsed = ShadowPortfolioState.from_dict(payload)
    if supplied is not None and supplied != parsed.digest:
        raise PaperError("state digest mismatch")
    if parsed.cash < ZERO or parsed.cash != money(parsed.cash) or any(
        p.book_cost != money(p.book_cost) for p in parsed.positions
    ):
        raise PaperError("cash/cost must be nonnegative cent amounts")
    token(parsed.portfolio_id)
    return parsed


def execution_config(raw):
    """All fee observations belong to the workspace, never engine defaults."""
    if not isinstance(raw, dict) or set(raw) - {
        "base_fee", "base_rate", "sell_tax_rate", "transfer_rates", "lot_size",
        "cash_reserve", "slippage_rate", "observed_max_gross",
    }:
        raise PaperError("invalid execution configuration")
    result = {}
    for key in ("base_fee", "base_rate", "sell_tax_rate"):
        value = decimal(raw[key], key)
        if value < ZERO:
            raise PaperError("negative fee/rate")
        result[key] = str(value)
    transfers = raw["transfer_rates"]
    if not isinstance(transfers, dict) or set(transfers) != {"SH", "SZ"}:
        raise PaperError("explicit SH/SZ transfer rates required")
    result["transfer_rates"] = {}
    for exchange, rate in transfers.items():
        value = decimal(rate, "transfer rate")
        if value < ZERO:
            raise PaperError("negative transfer rate")
        result["transfer_rates"][exchange] = str(value)
    result["lot_size"] = integer(raw["lot_size"], "lot_size", 1)
    if result["lot_size"] != 100:
        raise PaperError("this workflow supports ordinary 100-share buys")
    reserve = decimal(raw.get("cash_reserve", "0"))
    if reserve < ZERO or reserve != money(reserve):
        raise PaperError("reserve must be a nonnegative cent amount")
    result["cash_reserve"] = amount(reserve)
    if decimal(raw.get("slippage_rate", "0")) != ZERO:
        raise PaperError("this workflow currently supports zero slippage")
    result["slippage_rate"] = "0"
    maximum = raw.get("observed_max_gross")
    if maximum is not None and decimal(maximum) <= ZERO:
        raise PaperError("observed_max_gross must be positive")
    result["observed_max_gross"] = None if maximum is None else amount(maximum)
    return result


@cash_context
def fees(gross, side, instrument, config):
    basic = money(max(decimal(config["base_fee"]), gross * decimal(config["base_rate"])))
    tax = money(gross * decimal(config["sell_tax_rate"])) if side == "SELL" else money(0)
    transfer = money(gross * decimal(config["transfer_rates"][instrument[:2]]))
    return {"base": amount(basic), "tax": amount(tax), "transfer": amount(transfer),
            "total": amount(basic + tax + transfer)}


def board(instrument):
    if re.fullmatch(r"SH(?:600|601|603|605)[0-9]{3}", instrument):
        return "SH_MAIN"
    if re.fullmatch(r"SZ(?:000|001|002|003)[0-9]{3}", instrument):
        return "SZ_MAIN"
    if re.fullmatch(r"SZ(?:300|301)[0-9]{3}", instrument):
        return "CHINEXT"
    return None


def validate_rules(rules, selected):
    if not isinstance(rules, dict) or set(rules) != {"date", "source", "boards", "instruments"}:
        raise PaperError("rules require date/source/boards/instruments")
    if date(rules["date"]) != selected or not isinstance(rules["source"], str) or not rules["source"].strip():
        raise PaperError("dated rule source required")
    if not isinstance(rules["boards"], dict) or not isinstance(rules["instruments"], dict):
        raise PaperError("rule boards/instruments must be objects")
    for key, rule in list(rules["boards"].items()) + list(rules["instruments"].items()):
        if key not in {"SH_MAIN", "SZ_MAIN", "CHINEXT"} and not re.fullmatch(r"(?:SH|SZ)[0-9]{6}", key):
            raise PaperError("invalid rule security/category")
        if not isinstance(rule, dict) or set(rule) - {"suspended", "normal_limit", "limit_rate",
                                                       "reference_close", "limit_up", "limit_down"}:
            raise PaperError("invalid security rule")
        for flag in ("suspended", "normal_limit"):
            if flag in rule and type(rule[flag]) is not bool:
                raise PaperError("rule flags must be boolean")
        for field in ("limit_rate", "reference_close", "limit_up", "limit_down"):
            if field in rule and decimal(rule[field]) <= ZERO:
                raise PaperError("rule prices/rates must be positive")
        if ("limit_up" in rule) != ("limit_down" in rule):
            raise PaperError("both explicit price limits are required")


def trade_check(instrument, quote, rules, trade_date):
    """O and previous C only. Known suspension overrides a positive O."""
    category = board(instrument)
    if category is None:
        return "UNSUPPORTED_SECURITY", None, None
    if rules.get("date") != trade_date:
        raise PaperError("market rules date does not match execution date")
    rule = dict(rules.get("boards", {}).get(category, {}))
    rule.update(rules.get("instruments", {}).get(instrument, {}))
    if rule.get("suspended") is True:
        return "SUSPENDED", None, None
    opening = quote.get("open")
    if opening is None or decimal(opening) <= ZERO:
        return "MISSING_OPEN", None, None
    price = money(opening)
    if price <= ZERO:
        return "MISSING_OPEN", None, None
    if "limit_up" in rule and "limit_down" in rule:
        upper, lower = decimal(rule["limit_up"]), decimal(rule["limit_down"])
        if upper != money(upper) or lower != money(lower):
            raise PaperError("explicit price limits must match price tick")
    else:
        if rule.get("normal_limit") is not True or "limit_rate" not in rule:
            return "UNKNOWN_LIMIT_RULE", price, None
        reference = rule.get("reference_close")
        if reference is None:
            if quote.get("factor") is None or quote.get("previous_factor") is None:
                return "UNKNOWN_FACTOR", price, None
            if decimal(quote["factor"]) != decimal(quote["previous_factor"]):
                return "FACTOR_CHANGED", price, None
            reference = quote.get("previous_close")
        if reference is None or decimal(reference) <= ZERO:
            return "MISSING_PREVIOUS_CLOSE", price, None
        rate = decimal(rule["limit_rate"])
        if not ZERO < rate < Decimal("1"):
            raise PaperError("limit rate must be between zero and one")
        upper = money(money(reference) * (1 + rate))
        lower = money(money(reference) * (1 - rate))
    if not ZERO < lower < upper:
        raise PaperError("invalid price limit interval")
    limits = {"down": amount(lower), "up": amount(upper)}
    if price >= upper or price <= lower:
        return "OPEN_AT_LIMIT", price, limits
    return None, price, limits


@cash_context
def apply_events(prior, events):
    prior = state(prior)
    cash = prior.cash
    positions = {p.instrument: [p.quantity, p.book_cost] for p in prior.positions}
    external = ZERO
    for event in events:
        validate_event(event)
        cash += decimal(event["cash"])
        if cash < ZERO:
            raise PaperError("event would make cash negative")
        if event["kind"] == "EXTERNAL":
            external += decimal(event["cash"])
        if event["kind"] == "SHARES":
            instrument = event["instrument"]
            if instrument not in positions:
                raise PaperError("share event requires an existing position")
            positions[instrument][0] += event["quantity_delta"]
            positions[instrument][1] += decimal(event["cost_delta"])
            if positions[instrument][0] <= 0 or positions[instrument][1] < ZERO:
                raise PaperError("invalid share event after-position")
    after = state({"portfolio_id": prior.portfolio_id,
                   "as_of_date": max([prior.as_of_date] + [e["booking_date"] for e in events]),
                   "cash": amount(cash), "positions": [
                       {"instrument": i, "quantity": q, "book_cost": amount(c)}
                       for i, (q, c) in sorted(positions.items())]})
    return after.to_dict(), amount(external)


def validate_event(event):
    if set(event) != {"event_id", "effective_date", "booking_date", "kind", "cash",
                      "instrument", "quantity_delta", "cost_delta", "source"}:
        raise PaperError("event fields mismatch")
    token(event["event_id"])
    if date(event["booking_date"]) < date(event["effective_date"]):
        raise PaperError("booking date precedes effective date")
    if not isinstance(event["source"], str) or not event["source"].strip():
        raise PaperError("event source required")
    cash, cost = decimal(event["cash"]), decimal(event["cost_delta"])
    if type(event["quantity_delta"]) is not int:
        raise PaperError("event quantity delta must be an integer")
    if cash != money(cash) or cost != money(cost):
        raise PaperError("events require cent amounts")
    if event["kind"] not in {"DIVIDEND", "TAX", "EXTERNAL", "SHARES"}:
        raise PaperError("unknown event kind")
    if event["kind"] == "DIVIDEND" and cash <= ZERO or event["kind"] == "TAX" and cash >= ZERO:
        raise PaperError("event cash direction mismatch")
    if event["kind"] == "SHARES":
        if cash != ZERO or type(event["quantity_delta"]) is not int or event["quantity_delta"] == 0:
            raise PaperError("share event requires a nonzero integer delta and zero cash")
        if not re.fullmatch(r"(?:SH|SZ)[0-9]{6}", event["instrument"] or ""):
            raise PaperError("invalid event instrument")
    elif event["quantity_delta"] != 0 or cost != ZERO:
        raise PaperError("cash event cannot change shares/cost")


@cash_context
def settle_plan(plan, prior, quotes, rules, events=(), buy_locks=None):
    """Replayable settlement with fixed per-slot budgets and ordered replacement."""
    config = execution_config(plan["execution"])
    opening_state, external = apply_events(prior, events)
    opening = state(opening_state)
    trade_date = date(plan["trade_date"])
    validate_rules(rules, trade_date)
    if opening.as_of_date > trade_date:
        raise PaperError("state is later than execution date")
    cash = opening.cash
    positions = {p.instrument: [p.quantity, p.book_cost] for p in opening.positions}
    locks = buy_locks or {}
    decisions = []

    def decision(row, side, reason=None, price=None, limits=None, **facts):
        result = {"instrument": row["instrument"], "side": side,
                  "suggested_quantity": row["suggested_quantity"],
                  "status": "SKIPPED" if reason else "FILLED", "reason": reason,
                  "price": None if price is None else amount(price), "limits": limits,
                  "quantity": 0, "gross": "0.00", "fees": None, "cash_effect": "0.00",
                  "budget": None, "removed_cost": "0.00", "realized_pnl": "0.00",
                  "fee_estimate_outside_observed_scale": False}
        result.update(facts)
        decisions.append(result)

    def filled(row, side, quantity, price, limits, budget=None):
        nonlocal cash
        instrument = row["instrument"]
        gross = money(price * quantity)
        fee = fees(gross, side, instrument, config)
        total_fee = decimal(fee["total"])
        effect = -(gross + total_fee) if side == "BUY" else gross - total_fee
        removed = ZERO
        if side == "BUY":
            positions[instrument] = [quantity, -effect]
        else:
            held, cost = positions[instrument]
            removed = cost if held == quantity else money(cost * quantity / held)
            if held == quantity:
                del positions[instrument]
            else:
                positions[instrument] = [held - quantity, cost - removed]
        cash = money(cash + effect)
        maximum = config["observed_max_gross"]
        decision(row, side, price=price, limits=limits, quantity=quantity, gross=amount(gross),
                 fees=fee, cash_effect=amount(effect), budget=budget, removed_cost=amount(removed),
                 realized_pnl=amount(effect - removed) if side == "SELL" else "0.00",
                 fee_estimate_outside_observed_scale=maximum is not None and gross > decimal(maximum))

    for row in plan["sells"]:
        instrument = row["instrument"]
        reason, price, limits = trade_check(instrument, quotes.get(instrument, {}), rules, trade_date)
        held = positions.get(instrument, [0, ZERO])[0]
        locked = locks.get(instrument, {})
        unavailable = locked.get("quantity", 0) if locked.get("date") == trade_date else 0
        quantity = row["suggested_quantity"]
        if quantity > held - unavailable:
            reason = "INSUFFICIENT_SELLABLE_SHARES"
        if reason:
            decision(row, "SELL", reason, price, limits)
            continue
        gross = money(price * quantity)
        fee = decimal(fees(gross, "SELL", instrument, config)["total"])
        if cash + gross - fee < ZERO:
            decision(row, "SELL", "INSUFFICIENT_CASH_FOR_SELL_FEES", price, limits)
            continue
        filled(row, "SELL", quantity, price, limits)

    slots = min(plan["target_buys"], max(0, plan["strategy"]["topk"] - len(positions)))
    clean_cash = cash
    reserve = decimal(config["cash_reserve"])
    if reserve > cash:
        raise PaperError("cash reserve exceeds clean cash")
    budget = ((cash - reserve) / slots).quantize(CENT, rounding=ROUND_DOWN) if slots else ZERO
    filled_buys = 0
    lot = config["lot_size"]
    for row in plan["buys"]:
        instrument = row["instrument"]
        if filled_buys == slots:
            decision(row, "BUY", "NO_SLOT", budget=amount(budget))
            continue
        reason, price, limits = trade_check(instrument, quotes.get(instrument, {}), rules, trade_date)
        if instrument not in plan["universe"]:
            reason = "OUTSIDE_ANCHOR_UNIVERSE"
        elif instrument in positions:
            reason = "ALREADY_HELD"
        if reason:
            decision(row, "BUY", reason, price, limits, budget=amount(budget))
            continue
        cap = min(budget, cash - reserve)
        quantity = int(cap / (price * lot)) * lot
        while quantity > 0:
            gross = money(price * quantity)
            if gross + decimal(fees(gross, "BUY", instrument, config)["total"]) <= cap:
                break
            quantity -= lot
        if quantity == 0:
            decision(row, "BUY", "INSUFFICIENT_SLOT_BUDGET", price, limits, budget=amount(budget))
            continue
        filled(row, "BUY", quantity, price, limits, amount(budget))
        filled_buys += 1

    after = state({"portfolio_id": opening.portfolio_id, "as_of_date": trade_date,
                   "cash": amount(cash), "positions": [
                       {"instrument": i, "quantity": q, "book_cost": amount(c)}
                       for i, (q, c) in sorted(positions.items())]})
    return {"cycle_id": plan["cycle_id"], "plan_digest": fingerprint_value(plan),
            "prior_digest": state(prior).digest, "after": after.to_dict(),
            "events": list(events), "external_cashflow": external, "buy_locks": locks,
            "quotes": quotes, "rules": rules, "decisions": decisions, "slots": slots,
            "slot_budget": amount(budget), "clean_cash": amount(clean_cash),
            "total_fees": amount(sum((decimal(d["fees"]["total"]) for d in decisions if d["fees"]), ZERO)),
            "execution_assumption": "CASH_OPEN_SELL_THEN_BUY_NO_VOLUME_MODEL"}


@cash_context
def valuation(prior, valuation_date, closes, cached, baseline_nav, external, has_external=False):
    current = state(prior)
    date(valuation_date)
    if valuation_date < current.as_of_date:
        raise PaperError("valuation cannot precede current state")
    if decimal(baseline_nav) <= ZERO:
        raise PaperError("valuation baseline must be positive")
    for value in closes.values():
        if value is not None and (decimal(value) <= ZERO or decimal(value) != money(value)):
            raise PaperError("valuation prices must be positive cash prices at cent precision")
    rows, known = [], ZERO
    for p in current.positions:
        price, price_date = closes.get(p.instrument), valuation_date
        if price is None:
            old = cached.get(p.instrument, {})
            price, price_date = old.get("price"), old.get("price_date")
            if price is not None and (decimal(price) <= ZERO or date(price_date) > valuation_date):
                raise PaperError("invalid stale price/date")
        stale = price is not None and price_date != valuation_date
        value = None if price is None else money(p.quantity * decimal(price))
        if value is not None:
            known += value
        rows.append({"instrument": p.instrument, "quantity": p.quantity, "book_cost": amount(p.book_cost),
                     "price": None if price is None else amount(price), "price_date": price_date,
                     "stale": stale, "market_value": None if value is None else amount(value)})
    missing = [r["instrument"] for r in rows if r["price"] is None]
    nav = None if missing else money(current.cash + known)
    pnl = None if nav is None else nav - decimal(baseline_nav) - decimal(external)
    return {"date": valuation_date, "state_digest": current.digest, "cash": amount(current.cash),
            "positions": rows, "known_market_value": amount(known),
            "status": "PARTIAL" if missing else "STALE" if any(r["stale"] for r in rows) else "COMPLETE",
            "missing": missing, "nav": None if nav is None else amount(nav),
            "net_external_cashflow": amount(external),
            "pnl_excluding_external": None if pnl is None else amount(pnl),
            "return_excluding_external": None if pnl is None else str(pnl / decimal(baseline_nav)),
            "external_funding_observed": has_external,
            "unit_nav": str(nav / decimal(baseline_nav)) if nav is not None and decimal(external) == ZERO and not has_external else None,
            "return_method": "PNL_EXCLUDING_NET_EXTERNAL_OVER_INITIAL_NAV"}
