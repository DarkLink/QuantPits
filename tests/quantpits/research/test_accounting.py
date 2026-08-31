from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, ROUND_DOWN, getcontext
from pathlib import Path
import subprocess
import sys

import pytest

import quantpits.research.accounting as accounting_module
from quantpits.research.accounting import (
    ExecutionAssumption,
    ShadowAccountingContractError,
    ShadowIntentBatch,
    ShadowIntentResult,
    ShadowPortfolioState,
    ShadowPortfolioTransition,
    ShadowQuoteSnapshot,
    ShadowTransitionResult,
)


def assumption_raw(**changes):
    value = {
        "schema_version": 1,
        "assumption_id": "synthetic-v1",
        "deal_price": "NEXT_OPEN",
        "sell_before_buy": True,
        "partial_fill": False,
        "short_sell": False,
        "lot_size": 100,
        "money_quantum": "0.01",
        "price_quantum": "0.01",
        "rounding_mode": "HALF_UP",
        "buy_slippage_rate": "0.02",
        "sell_slippage_rate": "0.01",
        "buy_fee_rate": "0.01",
        "sell_fee_rate": "0.01",
        "minimum_buy_fee": "1",
        "minimum_sell_fee": "1",
        "corporate_action_mode": "UNMODELED",
    }
    value.update(changes)
    return value


def state_raw(cash="1000", positions=()):
    return {
        "portfolio_id": "portfolio-synthetic",
        "as_of_date": "2026-01-02",
        "cash": cash,
        "positions": list(positions),
    }


def position(instrument="SH600001", quantity=100, book_cost="800"):
    return {"instrument": instrument, "quantity": quantity, "book_cost": book_cost}


def intent(intent_id, side, instrument, quantity, **changes):
    value = {
        "intent_id": intent_id,
        "portfolio_id": "portfolio-synthetic",
        "cycle_id": "cycle-synthetic",
        "trade_date": "2026-01-05",
        "side": side,
        "instrument": instrument,
        "quantity": quantity,
    }
    value.update(changes)
    return value


def quote(instrument, price="10", status="OBSERVED", **changes):
    value = {
        "instrument": instrument,
        "trade_date": "2026-01-05",
        "field": "NEXT_OPEN",
        "status": status,
        "price": price if status == "OBSERVED" else None,
    }
    value.update(changes)
    return value


def build(prior_positions=(), cash="1000", intent_rows=(), quote_rows=None, assumption_changes=None):
    prior = ShadowPortfolioState.from_dict(state_raw(cash, prior_positions))
    intents = ShadowIntentBatch.from_iterable(
        portfolio_id="portfolio-synthetic",
        cycle_id="cycle-synthetic",
        trade_date="2026-01-05",
        rows=intent_rows,
    )
    requested = tuple(sorted(
        {item["instrument"] for item in prior_positions}
        | {item["instrument"] for item in intent_rows}
    ))
    if quote_rows is None:
        quote_rows = [quote(item) for item in requested]
    quotes = ShadowQuoteSnapshot.from_iterable(
        portfolio_id="portfolio-synthetic",
        cycle_id="cycle-synthetic",
        trade_date="2026-01-05",
        requested_instruments=requested,
        rows=quote_rows,
    )
    assumptions = ExecutionAssumption.from_dict(assumption_raw(**(assumption_changes or {})))
    return prior, intents, quotes, assumptions


def apply(**kwargs):
    return ShadowPortfolioTransition.apply(
        **dict(zip(("prior", "intents", "quotes", "assumption"), build(**kwargs)))
    )


def test_empty_portfolio_and_empty_intents_is_exact_no_op():
    result = apply()
    assert result.after_state.cash == Decimal("1000.00")
    assert result.after_state.positions == ()
    assert result.terminal_results == ()
    assert (result.filled_count, result.no_fill_count, result.all_filled) == (0, 0, True)
    assert (result.nav_before, result.nav_after, result.nav_reconciled) == (
        Decimal("1000.00"), Decimal("1000.00"), True
    )


def test_negative_opening_cash_is_exactly_represented_and_carried_by_empty_transition():
    state = ShadowPortfolioState.from_dict(state_raw("-12.34"))
    assert state.to_dict()["cash"] == "-12.34"
    result = apply(cash="-12.34")
    assert result.after_state.cash == Decimal("-12.34")
    assert (result.nav_before, result.nav_after) == (Decimal("-12.34"), Decimal("-12.34"))


def test_negative_cash_sell_can_improve_without_eliminating_opening_deficit():
    result = apply(
        cash="-2000",
        prior_positions=(position(),),
        intent_rows=(intent("sell-1", "SELL", "SH600001", 100),),
    )
    assert result.terminal_results[0].status == "FILLED"
    assert result.after_state.cash == Decimal("-1019.90")


def test_negative_cash_sell_can_cross_zero_and_fund_an_affordable_buy():
    result = apply(
        cash="-100",
        prior_positions=(position(),),
        intent_rows=(
            intent("sell-1", "SELL", "SH600001", 100),
            intent("buy-1", "BUY", "SZ000002", 100),
        ),
        quote_rows=(quote("SH600001", "10"), quote("SZ000002", "5")),
    )
    assert [item.status for item in result.terminal_results] == ["FILLED", "FILLED"]
    assert result.after_state.cash == Decimal("365.00")


def test_negative_cash_buy_is_no_fill_and_does_not_worsen_deficit():
    result = apply(
        cash="-1",
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),),
    )
    assert result.terminal_results[0].status == "NO_FILL_INSUFFICIENT_CASH"
    assert result.after_state.cash == Decimal("-1.00")


def test_sell_with_fee_above_gross_fails_before_worsening_cash_or_positions():
    values = build(
        cash="-1",
        prior_positions=(position(),),
        intent_rows=(intent("sell-1", "SELL", "SH600001", 100),),
        assumption_changes={"minimum_sell_fee": "1000"},
    )
    with pytest.raises(ShadowAccountingContractError, match="non-negative"):
        ShadowPortfolioTransition.apply(
            **dict(zip(("prior", "intents", "quotes", "assumption"), values))
        )


def test_full_sell_then_buy_has_exact_cash_cost_fee_slippage_and_nav():
    result = apply(
        prior_positions=(position(),),
        intent_rows=(
            intent("sell-1", "SELL", "SH600001", 100),
            intent("buy-1", "BUY", "SZ000002", 100),
        ),
        quote_rows=(quote("SH600001", "10"), quote("SZ000002", "5")),
    )
    assert [item.status for item in result.terminal_results] == ["FILLED", "FILLED"]
    assert result.after_state.cash == Decimal("1465.00")
    assert result.after_state.positions[0].book_cost == Decimal("515.10")
    assert result.gross_sell == Decimal("990.00")
    assert result.gross_buy == Decimal("510.00")
    assert result.total_fee == Decimal("15.00")
    assert result.slippage_cost == Decimal("20.00")
    assert result.rounding_adjustment == Decimal("0.00")
    assert result.realized_pnl == Decimal("180.10")
    assert (result.nav_before, result.nav_after) == (Decimal("2000.00"), Decimal("1965.00"))


def test_partial_sell_removes_proportional_cost():
    result = apply(
        prior_positions=(position(quantity=300, book_cost="1000.01"),),
        intent_rows=(intent("sell-1", "SELL", "SH600001", 100),),
    )
    terminal = result.terminal_results[0]
    assert terminal.removed_book_cost == Decimal("333.34")
    assert result.after_state.positions[0].quantity == 200
    assert result.after_state.positions[0].book_cost == Decimal("666.67")


def test_full_sell_removes_entire_rounding_residue():
    result = apply(
        prior_positions=(position(quantity=3, book_cost="10.01"),),
        intent_rows=(intent("sell-1", "SELL", "SH600001", 3),),
    )
    assert result.terminal_results[0].removed_book_cost == Decimal("10.01")
    assert result.after_state.positions == ()


def test_buy_adds_to_existing_position_cost():
    result = apply(
        cash="2000",
        prior_positions=(position(quantity=100, book_cost="800"),),
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),),
    )
    assert result.after_state.positions[0].quantity == 200
    assert result.after_state.positions[0].book_cost == Decimal("1830.20")


def test_sell_proceeds_fund_later_buy():
    result = apply(
        cash="0",
        prior_positions=(position(),),
        intent_rows=(
            intent("sell-1", "SELL", "SH600001", 100),
            intent("buy-1", "BUY", "SZ000002", 100),
        ),
        quote_rows=(quote("SH600001", "10"), quote("SZ000002", "5")),
    )
    assert [item.status for item in result.terminal_results] == ["FILLED", "FILLED"]


def test_first_buy_consumes_cash_and_later_buy_is_whole_order_no_fill():
    result = apply(
        cash="1200",
        intent_rows=(
            intent("buy-1", "BUY", "SH600001", 100),
            intent("buy-2", "BUY", "SZ000002", 100),
        ),
        quote_rows=(quote("SH600001", "5"), quote("SZ000002", "10")),
    )
    assert [item.status for item in result.terminal_results] == [
        "FILLED", "NO_FILL_INSUFFICIENT_CASH"
    ]
    assert result.terminal_results[1].gross is None
    assert result.after_state.positions[0].quantity == 100


def test_later_buy_continues_after_an_earlier_cash_no_fill():
    result = apply(
        cash="1200",
        intent_rows=(
            intent("buy-1", "BUY", "SH600001", 100),
            intent("buy-2", "BUY", "SH600002", 100),
            intent("buy-3", "BUY", "SZ000003", 100),
        ),
        quote_rows=(
            quote("SH600001", "5"), quote("SH600002", "10"), quote("SZ000003", "1")
        ),
    )
    assert [item.status for item in result.terminal_results] == [
        "FILLED", "NO_FILL_INSUFFICIENT_CASH", "FILLED"
    ]
    assert [item.instrument for item in result.after_state.positions] == [
        "SH600001", "SZ000003"
    ]


def test_multiple_terminal_members_preserve_exact_requested_identity_and_order():
    rows = (
        intent("sell-1", "SELL", "SH600001", 100),
        intent("sell-2", "SELL", "SH600002", 100),
        intent("buy-1", "BUY", "SZ000001", 100),
        intent("buy-2", "BUY", "SZ000002", 100),
    )
    result = apply(
        cash="2000",
        prior_positions=(position("SH600001"), position("SH600002")),
        intent_rows=rows,
        quote_rows=tuple(quote(row["instrument"], "5") for row in rows),
    )
    assert [item.intent_id for item in result.terminal_results] == [
        row["intent_id"] for row in rows
    ]
    assert len(result.terminal_results) == len(rows)


@pytest.mark.parametrize("quantity", [1, 99, 101])
def test_buy_quantity_must_match_lot_contract(quantity):
    values = build(intent_rows=(intent("buy-1", "BUY", "SH600001", quantity),))
    with pytest.raises(ShadowAccountingContractError, match="lot_size"):
        ShadowPortfolioTransition.apply(
            **dict(zip(("prior", "intents", "quotes", "assumption"), values))
        )


def test_sell_odd_lot_is_allowed():
    result = apply(
        prior_positions=(position(quantity=3, book_cost="20"),),
        intent_rows=(intent("sell-1", "SELL", "SH600001", 3),),
    )
    assert result.terminal_results[0].status == "FILLED"


@pytest.mark.parametrize(
    "prior_positions,intent_row,status",
    [
        ((), intent("sell-1", "SELL", "SH600001", 1), "NO_FILL_INSUFFICIENT_POSITION"),
        ((position(quantity=3),), intent("sell-1", "SELL", "SH600001", 4), "NO_FILL_INSUFFICIENT_POSITION"),
    ],
)
def test_sell_position_no_fill_precedes_quote_check(prior_positions, intent_row, status):
    result = apply(
        prior_positions=prior_positions,
        intent_rows=(intent_row,),
        quote_rows=(quote("SH600001", status="MISSING"),),
    )
    assert result.terminal_results[0].status == status


def test_missing_buy_quote_preserves_later_member_and_complete_nav():
    result = apply(
        intent_rows=(
            intent("buy-1", "BUY", "SH600001", 100),
            intent("buy-2", "BUY", "SZ000002", 100),
        ),
        quote_rows=(quote("SH600001", status="MISSING"), quote("SZ000002", "5")),
    )
    assert [item.status for item in result.terminal_results] == [
        "NO_FILL_MISSING_PRICE", "FILLED"
    ]
    assert result.valuation_status == "COMPLETE"


def test_missing_sell_quote_is_no_fill_and_preserves_state_exactly():
    result = apply(
        prior_positions=(position(),),
        intent_rows=(intent("sell-1", "SELL", "SH600001", 100),),
        quote_rows=(quote("SH600001", status="MISSING"),),
    )
    assert result.terminal_results[0].status == "NO_FILL_MISSING_PRICE"
    assert result.after_state.cash == result.prior_state.cash
    assert result.after_state.positions == result.prior_state.positions


def test_missing_retained_holding_quote_denies_nav_not_transition():
    result = apply(
        prior_positions=(position(),),
        quote_rows=(quote("SH600001", status="MISSING"),),
    )
    assert result.transition_status == "COMPLETE"
    assert result.valuation_status == "PARTIAL"
    assert result.missing_valuation_instruments == ("SH600001",)
    assert result.nav_before is None and result.nav_after is None
    assert result.nav_reconciled is False


def test_no_fill_accounting_fields_are_null_not_zero():
    result = apply(
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),),
        quote_rows=(quote("SH600001", status="MISSING"),),
    )
    payload = result.to_dict()["terminal_results"][0]
    for field in (
        "reference_price", "execution_price", "gross", "fee", "cash_effect",
        "removed_book_cost", "realized_pnl",
    ):
        assert payload[field] is None


def test_two_arms_do_not_mutate_shared_inputs():
    prior, intents, quotes, assumptions = build(
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),)
    )
    before = (prior.to_dict(), intents.to_dict(), quotes.to_dict(), assumptions.to_dict())
    first = ShadowPortfolioTransition.apply(
        prior=prior, intents=intents, quotes=quotes, assumption=assumptions
    )
    second = ShadowPortfolioTransition.apply(
        prior=prior, intents=intents, quotes=quotes, assumption=assumptions
    )
    assert first.to_dict() == second.to_dict()
    assert first.result_digest == second.result_digest
    assert before == (prior.to_dict(), intents.to_dict(), quotes.to_dict(), assumptions.to_dict())


def test_transition_is_independent_of_caller_decimal_context():
    values = build(
        cash="2000",
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),),
        quote_rows=(quote("SH600001", "10.005"),),
    )
    baseline = ShadowPortfolioTransition.apply(
        **dict(zip(("prior", "intents", "quotes", "assumption"), values))
    )
    context = getcontext()
    old_precision, old_rounding = context.prec, context.rounding
    try:
        context.prec = 3
        context.rounding = ROUND_DOWN
        repeated = ShadowPortfolioTransition.apply(
            **dict(zip(("prior", "intents", "quotes", "assumption"), values))
        )
    finally:
        context.prec = old_precision
        context.rounding = old_rounding
    assert repeated.to_canonical_json_bytes() == baseline.to_canonical_json_bytes()


def test_result_and_inputs_are_deeply_immutable():
    result = apply()
    with pytest.raises(FrozenInstanceError):
        result.after_state.cash = Decimal("0")
    assert isinstance(result.after_state.positions, tuple)
    assert isinstance(result.terminal_results, tuple)


@pytest.mark.parametrize(
    "value", [None, "", " ", True, 1, 1.0, "abc", "NaN", "Infinity", "-Infinity", " 1"]
)
def test_cash_rejects_non_strict_decimal_representations(value):
    with pytest.raises(ShadowAccountingContractError):
        ShadowPortfolioState.from_dict(state_raw(cash=value))


@pytest.mark.parametrize("value", [None, "", " ", True, 100.0, 0, -1])
def test_quantity_rejects_non_positive_exact_integer_contract(value):
    with pytest.raises(ShadowAccountingContractError):
        ShadowPortfolioState.from_dict(state_raw(positions=(position(quantity=value),)))


@pytest.mark.parametrize("value", ["sh600001", "SH60001", "US000001", " SH600001"])
def test_instrument_requires_canonical_identity(value):
    with pytest.raises(ShadowAccountingContractError):
        ShadowPortfolioState.from_dict(state_raw(positions=(position(instrument=value),)))


@pytest.mark.parametrize("value", ["2026-1-02", "2026-02-30", " 2026-01-02", None])
def test_date_requires_exact_valid_calendar_representation(value):
    raw = state_raw()
    raw["as_of_date"] = value
    with pytest.raises(ShadowAccountingContractError):
        ShadowPortfolioState.from_dict(raw)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True}, {"schema_version": 1.0}, {"schema_version": 2},
        {"deal_price": "next_open"},
        {"sell_before_buy": 1}, {"sell_before_buy": False}, {"partial_fill": True},
        {"short_sell": True}, {"lot_size": True}, {"lot_size": 0},
        {"money_quantum": "0"}, {"price_quantum": "-0.01"},
        {"rounding_mode": "DOWN"}, {"buy_fee_rate": "1"},
        {"sell_fee_rate": "-0.01"}, {"minimum_buy_fee": 1},
        {"corporate_action_mode": "MODELED"},
    ],
)
def test_assumption_rejects_unsupported_or_non_strict_values(changes):
    with pytest.raises(ShadowAccountingContractError):
        ExecutionAssumption.from_dict(assumption_raw(**changes))


def test_state_rejects_unsorted_and_duplicate_positions():
    with pytest.raises(ShadowAccountingContractError, match="unique and sorted"):
        ShadowPortfolioState.from_dict(state_raw(positions=(
            position("SZ000002"), position("SH600001")
        )))
    with pytest.raises(ShadowAccountingContractError, match="unique and sorted"):
        ShadowPortfolioState.from_dict(state_raw(positions=(position(), position())))


def test_intents_reject_duplicate_id_instrument_and_buy_before_sell():
    base = dict(
        portfolio_id="portfolio-synthetic", cycle_id="cycle-synthetic",
        trade_date="2026-01-05",
    )
    with pytest.raises(ShadowAccountingContractError, match="unique"):
        ShadowIntentBatch.from_iterable(rows=(
            intent("same", "BUY", "SH600001", 100),
            intent("same", "BUY", "SZ000002", 100),
        ), **base)
    with pytest.raises(ShadowAccountingContractError, match="unique"):
        ShadowIntentBatch.from_iterable(rows=(
            intent("one", "BUY", "SH600001", 100),
            intent("two", "BUY", "SH600001", 100),
        ), **base)
    with pytest.raises(ShadowAccountingContractError, match="precede"):
        ShadowIntentBatch.from_iterable(rows=(
            intent("buy", "BUY", "SH600001", 100),
            intent("sell", "SELL", "SZ000002", 100),
        ), **base)


@pytest.mark.parametrize(
    "rows,requested",
    [
        ((quote("SH600001"),), ("SH600001", "SZ000002")),
        ((quote("SH600001"), quote("SH600001")), ("SH600001", "SH600001")),
        ((quote("SH600001"), quote("SZ000002")), ("SH600001",)),
    ],
)
def test_quote_snapshot_rejects_missing_duplicate_or_foreign_members(rows, requested):
    with pytest.raises(ShadowAccountingContractError):
        ShadowQuoteSnapshot.from_iterable(
            portfolio_id="portfolio-synthetic", cycle_id="cycle-synthetic",
            trade_date="2026-01-05", requested_instruments=requested, rows=rows,
        )


@pytest.mark.parametrize(
    "row",
    [
        {**quote("SH600001", status="MISSING"), "price": "10"},
        quote("SH600001", status="OBSERVED", price=None),
        quote("SH600001", status="OBSERVED", price="0"),
        quote("SH600001", status="OBSERVED", trade_date="2026-01-06"),
        quote("SH600001", status="OBSERVED", field="OPEN"),
    ],
)
def test_quote_snapshot_rejects_impossible_or_foreign_observation(row):
    with pytest.raises(ShadowAccountingContractError):
        ShadowQuoteSnapshot.from_iterable(
            portfolio_id="portfolio-synthetic", cycle_id="cycle-synthetic",
            trade_date="2026-01-05", requested_instruments=("SH600001",), rows=(row,),
        )


def test_apply_rejects_cross_aggregate_identity_and_non_later_date():
    prior, intents, quotes, assumptions = build()
    foreign = ShadowIntentBatch.from_iterable(
        portfolio_id="foreign", cycle_id="cycle-synthetic", trade_date="2026-01-05", rows=()
    )
    with pytest.raises(ShadowAccountingContractError, match="prior portfolio"):
        ShadowPortfolioTransition.apply(
            prior=prior, intents=foreign, quotes=quotes, assumption=assumptions
        )
    same_day = ShadowIntentBatch.from_iterable(
        portfolio_id="portfolio-synthetic", cycle_id="cycle-synthetic",
        trade_date="2026-01-02", rows=(),
    )
    same_quotes = ShadowQuoteSnapshot.from_iterable(
        portfolio_id="portfolio-synthetic", cycle_id="cycle-synthetic",
        trade_date="2026-01-02", requested_instruments=(), rows=(),
    )
    with pytest.raises(ShadowAccountingContractError, match="later"):
        ShadowPortfolioTransition.apply(
            prior=prior, intents=same_day, quotes=same_quotes, assumption=assumptions
        )


def test_intent_and_quote_identity_mismatches_fail_closed():
    with pytest.raises(ShadowAccountingContractError, match="intent identity"):
        ShadowIntentBatch.from_iterable(
            portfolio_id="portfolio-synthetic", cycle_id="cycle-synthetic",
            trade_date="2026-01-05",
            rows=(intent("buy", "BUY", "SH600001", 100, cycle_id="foreign"),),
        )
    prior, intents, _, assumptions = build()
    foreign_quotes = ShadowQuoteSnapshot.from_iterable(
        portfolio_id="portfolio-synthetic", cycle_id="foreign",
        trade_date="2026-01-05", requested_instruments=(), rows=(),
    )
    with pytest.raises(ShadowAccountingContractError, match="quote snapshot identity"):
        ShadowPortfolioTransition.apply(
            prior=prior, intents=intents, quotes=foreign_quotes, assumption=assumptions
        )


def test_apply_revalidates_forged_aggregate_member():
    prior, intents, quotes, assumptions = build(prior_positions=(position(),))
    object.__setattr__(prior.positions[0], "instrument", "FOREIGN")
    with pytest.raises(ShadowAccountingContractError):
        ShadowPortfolioTransition.apply(
            prior=prior, intents=intents, quotes=quotes, assumption=assumptions
        )


def test_apply_revalidates_every_forged_aggregate_boundary():
    prior, intents, quotes, assumptions = build(
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),)
    )
    object.__setattr__(intents.intents[0], "quantity", True)
    with pytest.raises(ShadowAccountingContractError):
        ShadowPortfolioTransition.apply(
            prior=prior, intents=intents, quotes=quotes, assumption=assumptions
        )

    prior, intents, quotes, assumptions = build(
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),)
    )
    object.__setattr__(quotes.quotes[0], "status", "FORGED")
    with pytest.raises(ShadowAccountingContractError):
        ShadowPortfolioTransition.apply(
            prior=prior, intents=intents, quotes=quotes, assumption=assumptions
        )

    prior, intents, quotes, assumptions = build()
    object.__setattr__(assumptions, "schema_version", 1.0)
    with pytest.raises(ShadowAccountingContractError):
        ShadowPortfolioTransition.apply(
            prior=prior, intents=intents, quotes=quotes, assumption=assumptions
        )


def test_callers_cannot_construct_or_replace_capability_results():
    with pytest.raises(ShadowAccountingContractError):
        ShadowTransitionResult()
    with pytest.raises(ShadowAccountingContractError):
        ShadowIntentResult()
    result = apply()
    with pytest.raises(ShadowAccountingContractError):
        replace(result, promotion_capability=True)
    tampered = result.to_dict()
    tampered["transition_status"] = "COMPLETE"
    with pytest.raises(ShadowAccountingContractError):
        ShadowTransitionResult(**tampered)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("promotion_capability", True),
        ("filled_count", 99),
        ("gross_buy", Decimal("999")),
        ("rounding_adjustment", Decimal("99")),
    ],
)
def test_truth_owner_rejects_impossible_cross_field_result(field, replacement):
    result = apply(
        cash="2000", intent_rows=(intent("buy-1", "BUY", "SH600001", 100),)
    )
    values = dict(vars(result))
    values[field] = replacement
    with pytest.raises(ShadowAccountingContractError):
        ShadowTransitionResult._create(accounting_module._RESULT_TOKEN, **values)


def test_public_constructors_reject_caller_authority_fields():
    raw = state_raw()
    raw["digest"] = "0" * 64
    with pytest.raises(ShadowAccountingContractError, match="fields mismatch"):
        ShadowPortfolioState.from_dict(raw)
    raw_assumption = assumption_raw()
    raw_assumption["verified"] = True
    with pytest.raises(ShadowAccountingContractError, match="fields mismatch"):
        ExecutionAssumption.from_dict(raw_assumption)


@pytest.mark.parametrize(
    "constructor",
    [ExecutionAssumption, ShadowPortfolioState, ShadowIntentBatch, ShadowQuoteSnapshot],
)
def test_direct_input_construction_cannot_bypass_strict_raw_contract(constructor):
    with pytest.raises(ShadowAccountingContractError):
        constructor()


@pytest.mark.parametrize("minimum,expected", [("4.99", "5"), ("5", "5"), ("5.01", "5.01")])
def test_minimum_fee_boundary_and_money_rounding(minimum, expected):
    result = apply(
        cash="2000",
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),),
        assumption_changes={
            "buy_slippage_rate": "0", "buy_fee_rate": "0.005", "minimum_buy_fee": minimum,
        },
    )
    assert result.terminal_results[0].fee == Decimal(expected)


def test_half_up_price_and_money_rounding_are_exact():
    result = apply(
        cash="2000",
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),),
        quote_rows=(quote("SH600001", "10.005"),),
        assumption_changes={
            "buy_slippage_rate": "0", "buy_fee_rate": "0", "minimum_buy_fee": "0",
        },
    )
    terminal = result.terminal_results[0]
    assert terminal.execution_price == Decimal("10.01")
    assert terminal.gross == Decimal("1001.00")


def test_non_power_of_ten_quantums_are_real_increments_not_decimal_places():
    result = apply(
        cash="2000",
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),),
        quote_rows=(quote("SH600001", "10.03"),),
        assumption_changes={
            "money_quantum": "0.05", "price_quantum": "0.05",
            "buy_slippage_rate": "0", "buy_fee_rate": "0", "minimum_buy_fee": "1.03",
        },
    )
    terminal = result.terminal_results[0]
    assert terminal.execution_price == Decimal("10.05")
    assert terminal.gross == Decimal("1005.00")
    assert terminal.fee == Decimal("1.05")


def test_odd_lot_high_precision_reference_exposes_rounding_adjustment():
    result = apply(
        prior_positions=(position(quantity=961, book_cost="100.00"),),
        intent_rows=(intent("sell-1", "SELL", "SH600001", 961),),
        quote_rows=(quote("SH600001", "59.515"),),
        assumption_changes={
            "sell_slippage_rate": "0", "sell_fee_rate": "0.001",
            "minimum_sell_fee": "0",
        },
    )
    assert result.slippage_cost == Decimal("-4.81")
    assert result.rounding_adjustment == Decimal("0.01")
    assert result.nav_after == (
        result.nav_before - result.total_fee - result.slippage_cost
        - result.rounding_adjustment
    )


@pytest.mark.parametrize(
    "cash,book_cost", [("1000.001", "0"), ("-1000.001", "0"), ("1000", "1.001")],
)
def test_apply_rejects_prior_money_not_aligned_to_assumption_quantum(cash, book_cost):
    positions = () if book_cost == "0" else (position(book_cost=book_cost),)
    values = build(cash=cash, prior_positions=positions)
    with pytest.raises(ShadowAccountingContractError, match="money_quantum"):
        ShadowPortfolioTransition.apply(
            **dict(zip(("prior", "intents", "quotes", "assumption"), values))
        )


def test_price_rounding_cannot_create_zero_execution_price():
    values = build(
        cash="10",
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),),
        quote_rows=(quote("SH600001", "0.001"),),
        assumption_changes={
            "buy_slippage_rate": "0", "buy_fee_rate": "0", "minimum_buy_fee": "0",
        },
    )
    with pytest.raises(ShadowAccountingContractError, match="execution price"):
        ShadowPortfolioTransition.apply(
            **dict(zip(("prior", "intents", "quotes", "assumption"), values))
        )


def test_sell_fee_cannot_drive_after_cash_negative():
    values = build(
        cash="0",
        prior_positions=(position(quantity=1, book_cost="1"),),
        intent_rows=(intent("sell-1", "SELL", "SH600001", 1),),
        quote_rows=(quote("SH600001", "1"),),
        assumption_changes={
            "sell_slippage_rate": "0", "sell_fee_rate": "0", "minimum_sell_fee": "2",
        },
    )
    with pytest.raises(ShadowAccountingContractError, match="negative cash"):
        ShadowPortfolioTransition.apply(
            **dict(zip(("prior", "intents", "quotes", "assumption"), values))
        )


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_unexpected_and_process_control_exceptions_propagate(monkeypatch, error):
    prior, intents, quotes, assumptions = build(
        intent_rows=(intent("buy-1", "BUY", "SH600001", 100),)
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(ShadowPortfolioTransition, "_execute_intent", fail)
    with pytest.raises(type(error)):
        ShadowPortfolioTransition.apply(
            prior=prior, intents=intents, quotes=quotes, assumption=assumptions
        )


def test_result_payload_is_json_safe_and_carries_non_promotion_warning():
    result = apply()
    payload = result.to_dict()
    assert payload["evidence_class"] == "RETROSPECTIVE_TECHNICAL_REPLAY"
    assert payload["prospective_claim"] is False
    assert payload["promotion_capability"] is False
    assert payload["corporate_action_mode"] == "UNMODELED"
    assert payload["warnings"] == ["CORPORATE_ACTIONS_UNMODELED"]
    assert len(payload["result_digest"]) == 64
    assert result.to_canonical_json_bytes() == apply().to_canonical_json_bytes()


def test_public_import_runs_without_site_or_optional_dependencies():
    code = (
        "from quantpits.research import ExecutionAssumption, ShadowPortfolioTransition; "
        "import sys; "
        "assert 'quantpits.research.replay' not in sys.modules; "
        "assert 'pandas' not in sys.modules; "
        "assert ExecutionAssumption.__module__ == 'quantpits.research.accounting'"
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=str(Path(__file__).resolve().parents[3]),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
