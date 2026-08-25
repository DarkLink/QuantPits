from dataclasses import replace
from decimal import Decimal

import pytest

from quantpits.evidence.ranking import canonical_full_ranking
from quantpits.research.accounting import (
    ShadowAccountingContractError,
    ShadowIntentBatch,
    ShadowPortfolioState,
)
from quantpits.research.intents import (
    AnchorPriceSnapshot,
    CurrentRuleIntentDefinition,
    CurrentRuleShadowIntentPlanner,
    IntentPlanningContractError,
    IntentPlanningParityError,
    IntentPlanningResult,
)


ANCHOR = "2026-07-17"
TRADE = "2026-07-20"
PORTFOLIO = "shadow-champion"
CYCLE = "cycle-2026-07-17"
MARKET = "csi300"


def definition_raw(**changes):
    value = {
        "schema_version": 1,
        "definition_id": "current-rule-v1",
        "strategy_name": "topk_dropout",
        "topk": 2,
        "n_drop": 1,
        "buy_suggestion_factor": 2,
        "sell_out_of_universe": True,
        "production_buy_lot_size": 100,
        "cashflow_mode": "ZERO_EXTERNAL_CASHFLOW",
        "planning_price_rule": "CASH_CLOSE_WITH_0_9_1_1_ESTIMATES_V1",
        "buy_selection_rule": "FIRST_N_GENERATED_SUGGESTIONS_V1",
        "production_primitive_id": "TOPK_DROPOUT_ORDER_GENERATOR_CURRENT_RULE_V1",
    }
    value.update(changes)
    return value


def state(*, cash="10000", positions=(), as_of="2026-07-16"):
    return ShadowPortfolioState.from_dict({
        "portfolio_id": PORTFOLIO,
        "as_of_date": as_of,
        "cash": cash,
        "positions": [
            {"instrument": instrument, "quantity": quantity, "book_cost": cost}
            for instrument, quantity, cost in sorted(positions)
        ],
    })


def ranking(*, eligible=None, scores=None):
    eligible = eligible or ("SH000001", "SZ000002", "SH000003", "SZ000004")
    scores = scores if scores is not None else {
        "SH000001": 0.9,
        "SZ000002": 0.8,
        "SH000003": 0.7,
        "SZ000004": 0.6,
    }
    return canonical_full_ranking(eligible, scores)


def snapshot(ranking_value, prior, *, missing=(), close="10", extra=None):
    scored = {row["instrument"] for row in ranking_value.rows if row["scored"]}
    holdings = {item.instrument for item in prior.positions}
    requested = tuple(sorted(scored | holdings))
    rows = []
    for instrument in requested:
        is_missing = instrument in set(missing)
        rows.append({
            "instrument": instrument,
            "anchor_date": ANCHOR,
            "status": "MISSING" if is_missing else "OBSERVED",
            "cash_close": None if is_missing else close,
        })
    if extra is not None:
        rows[0].update(extra)
    return AnchorPriceSnapshot.from_iterable(
        anchor_date=ANCHOR, requested_instruments=requested, rows=rows,
    )


def plan(*, ranking_value=None, prior=None, prices=None, definition=None, **request_changes):
    ranking_value = ranking() if ranking_value is None else ranking_value
    prior = state() if prior is None else prior
    prices = snapshot(ranking_value, prior) if prices is None else prices
    request = {
        "portfolio_id": PORTFOLIO,
        "cycle_id": CYCLE,
        "market": MARKET,
        "anchor_date": ANCHOR,
        "trade_date": TRADE,
        "definition": definition or CurrentRuleIntentDefinition.from_dict(definition_raw()),
        "ranking": ranking_value,
        "prior": prior,
        "prices": prices,
    }
    request.update(request_changes)
    return CurrentRuleShadowIntentPlanner.plan(**request)


def intent_facts(result):
    return tuple((item.side, item.instrument, item.quantity) for item in result.intents.intents)


def test_empty_holdings_produce_first_n_buys_and_complete_authority():
    result = plan()
    assert result.status == "COMPLETE"
    assert result.target_buy_count == 2
    assert len(result.production_buy_suggestions) == 4
    assert result.selected_buy_instruments == ("SH000001", "SZ000002")
    assert result.excluded_buy_instruments == ("SH000003", "SZ000004")
    assert intent_facts(result) == (("BUY", "SH000001", 400), ("BUY", "SZ000002", 400))
    assert result.buy_intent_shortage == 0
    assert result.parity_checked is True and result.selection_checked is True
    assert result.prospective_claim is False and result.promotion_capability is False


def test_normal_dropout_sell_precedes_replacement_buy():
    ranked = ranking(
        eligible=("SH000001", "SZ000002", "SH000003", "SZ000004", "SH000005"),
        scores={
            "SH000001": .9, "SZ000002": .8, "SH000003": .7,
            "SZ000004": .6, "SH000005": .5,
        },
    )
    prior = state(positions=(("SH000005", 125, "900"), ("SH000003", 100, "800")))
    result = plan(ranking_value=ranked, prior=prior, prices=snapshot(ranked, prior))
    assert result.holding_classifications["normal_sell"] == ("SH000005",)
    assert intent_facts(result)[0] == ("SELL", "SH000005", 125)
    assert intent_facts(result)[1][0:2] == ("BUY", "SH000001")


def test_all_topk_holdings_are_retained_without_turnover():
    prior = state(positions=(("SH000001", 100, "800"), ("SZ000002", 100, "900")))
    result = plan(prior=prior, prices=snapshot(ranking(), prior))
    assert set(result.holding_classifications["continuing"]) == {"SH000001", "SZ000002"}
    assert result.production_sell_proposal == ()
    assert result.production_buy_suggestions == ()
    assert result.intents.intents == ()


def test_executable_universe_exit_is_before_normal_sell_and_buys():
    prior = state(positions=(("SH999999", 75, "500"), ("SZ000004", 100, "700")))
    prices = snapshot(ranking(), prior)
    result = plan(prior=prior, prices=prices)
    assert result.holding_classifications["forced_executable"] == ("SH999999",)
    assert result.holding_classifications["normal_sell"] == ()
    assert intent_facts(result)[0] == ("SELL", "SH999999", 75)


def test_forced_exits_can_exceed_drop_budget_and_preserve_odd_lots():
    prior = state(positions=(("SH999998", 25, "100"), ("SH999999", 75, "500")))
    result = plan(prior=prior, prices=snapshot(ranking(), prior))
    assert result.holding_classifications["forced_executable"] == ("SH999998", "SH999999")
    assert result.holding_classifications["normal_sell"] == ()
    assert intent_facts(result)[:2] == (
        ("SELL", "SH999998", 25), ("SELL", "SH999999", 75),
    )


def test_pending_forced_exit_and_unscored_holding_consume_no_capacity():
    ranked = ranking(
        eligible=("SH000001", "SZ000002", "SH000003"),
        scores={"SH000001": 0.9, "SZ000002": 0.8},
    )
    prior = state(positions=(("SH000003", 100, "700"), ("SH999999", 100, "800")))
    prices = snapshot(ranked, prior, missing=("SH999999",))
    result = plan(ranking_value=ranked, prior=prior, prices=prices)
    assert result.holding_classifications["forced_pending"] == ("SH999999",)
    assert result.holding_classifications["eligible_unscored_retained"] == ("SH000003",)
    assert result.target_buy_count == 0
    assert result.intents.intents == ()


def test_scored_missing_price_is_visible_and_not_silently_ranked():
    ranked = ranking()
    prior = state()
    result = plan(ranking_value=ranked, prior=prior, prices=snapshot(ranked, prior, missing=("SH000001",)))
    assert result.ranking_partitions["scored_with_missing_price"] == ("SH000001",)
    assert result.price_partitions["missing"] == ("SH000001",)
    assert "SH000001" not in result.raw_buy_candidate_instruments
    assert result.selected_buy_instruments == ("SZ000002", "SH000003")


def test_later_affordable_candidate_survives_earlier_unaffordable_candidate():
    ranked = ranking()
    prior = state(cash="10000")
    requested = tuple(sorted(row["instrument"] for row in ranked.rows if row["scored"]))
    rows = [{
        "instrument": instrument,
        "anchor_date": ANCHOR,
        "status": "OBSERVED",
        "cash_close": "1000000" if instrument == "SH000001" else "10",
    } for instrument in requested]
    prices = AnchorPriceSnapshot.from_iterable(
        anchor_date=ANCHOR, requested_instruments=requested, rows=rows,
    )
    result = plan(ranking_value=ranked, prior=prior, prices=prices)
    assert "SH000001" in result.raw_buy_candidate_instruments
    assert "SH000001" not in tuple(item["instrument"] for item in result.production_buy_suggestions)
    assert result.selected_buy_instruments == ("SZ000002", "SH000003")


def test_disabled_universe_has_no_membership_authority_or_forced_exit():
    prior = state(positions=(("SH999999", 100, "800"),))
    definition = CurrentRuleIntentDefinition.from_dict(
        definition_raw(sell_out_of_universe=False)
    )
    result = plan(prior=prior, prices=snapshot(ranking(), prior), definition=definition)
    assert result.universe["membership_check_status"] == "DISABLED_BY_DEFINITION"
    assert result.holding_classifications["forced_executable"] == ()
    assert result.holding_classifications["eligible_unscored_retained"] == ("SH999999",)
    assert result.target_buy_count == 1


def test_insufficient_planning_cash_exposes_buy_shortage():
    prior = state(cash="50")
    result = plan(prior=prior, prices=snapshot(ranking(), prior))
    assert result.target_buy_count == 2
    assert result.production_buy_suggestions == ()
    assert result.selected_buy_instruments == ()
    assert result.excluded_buy_instruments == ()
    assert result.buy_intent_shortage == 2


def test_sell_estimate_increases_cash_used_for_buy_sizing():
    prior = state(cash="0", positions=(("SH999999", 1000, "5000"),))
    result = plan(prior=prior, prices=snapshot(ranking(), prior, close="10"))
    buys = [item for item in result.intents.intents if item.side == "BUY"]
    assert buys and buys[0].quantity >= 100


def test_multi_sell_parity_replays_production_left_to_right_float_accumulation():
    prior = state(positions=(
        ("SH999997", 100, "1000"),
        ("SH999998", 100, "1000"),
        ("SH999999", 100, "1000"),
    ))
    ranked = ranking()
    closes = {
        "SH999997": "135.22987986828883",
        "SH999998": "847.58630320029556",
        "SH999999": "764.01084435763732",
    }
    requested = tuple(sorted(
        {row["instrument"] for row in ranked.rows if row["scored"]}
        | {item.instrument for item in prior.positions}
    ))
    prices = AnchorPriceSnapshot.from_iterable(
        anchor_date=ANCHOR,
        requested_instruments=requested,
        rows=tuple({
            "instrument": instrument,
            "anchor_date": ANCHOR,
            "status": "OBSERVED",
            "cash_close": closes.get(instrument, "10"),
        } for instrument in requested),
    )

    result = plan(ranking_value=ranked, prior=prior, prices=prices)
    assert result.status == "COMPLETE"
    assert intent_facts(result)[:3] == (
        ("SELL", "SH999997", 100),
        ("SELL", "SH999998", 100),
        ("SELL", "SH999999", 100),
    )


def test_book_cost_changes_proposal_facts_but_not_order_semantics():
    first = state(positions=(("SZ000004", 100, "500"),))
    second = state(positions=(("SZ000004", 100, "999"),))
    first_result = plan(prior=first, prices=snapshot(ranking(), first))
    second_result = plan(prior=second, prices=snapshot(ranking(), second))
    assert intent_facts(first_result) == intent_facts(second_result)
    assert first_result.intents.digest != second_result.intents.digest


def test_same_input_is_deterministic_and_inputs_do_not_cross_mutate():
    ranked = ranking()
    prior = state()
    prices = snapshot(ranked, prior)
    first = plan(ranking_value=ranked, prior=prior, prices=prices)
    second = plan(ranking_value=ranked, prior=prior, prices=prices)
    assert first.to_dict() == second.to_dict()
    assert first.result_digest == second.result_digest
    payload = first.to_dict()
    payload["request"]["cycle_id"] = "mutated"
    assert first.request["cycle_id"] == CYCLE


def test_tied_scores_preserve_canonical_production_input_order():
    ranked = ranking(scores={
        "SH000001": 1.0, "SZ000002": 1.0, "SH000003": 0.5, "SZ000004": 0.4,
    })
    result = plan(ranking_value=ranked, prices=snapshot(ranked, state()))
    assert result.raw_buy_candidate_instruments[:2] == ("SH000001", "SZ000002")


@pytest.mark.parametrize("field,bad", [
    ("schema_version", True), ("schema_version", 2), ("definition_id", ""),
    ("topk", True), ("topk", 0), ("n_drop", -1),
    ("buy_suggestion_factor", 0), ("sell_out_of_universe", 1),
    ("production_buy_lot_size", 200), ("strategy_name", "other"),
    ("cashflow_mode", "WITH_CASHFLOW"),
])
def test_definition_rejects_closest_invalid_representations(field, bad):
    with pytest.raises(IntentPlanningContractError):
        CurrentRuleIntentDefinition.from_dict(definition_raw(**{field: bad}))


def test_definition_rejects_absent_and_extra_fields():
    missing = definition_raw()
    missing.pop("topk")
    with pytest.raises(IntentPlanningContractError, match="fields mismatch"):
        CurrentRuleIntentDefinition.from_dict(missing)
    with pytest.raises(IntentPlanningContractError, match="fields mismatch"):
        CurrentRuleIntentDefinition.from_dict(definition_raw(fee="0.001"))


@pytest.mark.parametrize("extra", [
    {"next_open": "11"}, {"execution_price": "11"}, {"fee": "1"}, {"slippage": "0.01"},
])
def test_price_rows_reject_future_and_settlement_fields(extra):
    ranked = ranking()
    prior = state()
    with pytest.raises(IntentPlanningContractError, match="fields mismatch"):
        snapshot(ranked, prior, extra=extra)


def test_price_snapshot_rejects_missing_duplicate_foreign_and_wrong_order():
    row = lambda instrument: {
        "instrument": instrument, "anchor_date": ANCHOR,
        "status": "OBSERVED", "cash_close": "10",
    }
    requested = ("SH000001", "SZ000002")
    invalid_rows = (
        [row("SH000001")],
        [row("SH000001"), row("SH000001")],
        [row("SH000001"), row("SH000003")],
        [row("SZ000002"), row("SH000001")],
    )
    for rows in invalid_rows:
        with pytest.raises(IntentPlanningContractError, match="exactly cover"):
            AnchorPriceSnapshot.from_iterable(
                anchor_date=ANCHOR, requested_instruments=requested, rows=rows,
            )


@pytest.mark.parametrize("row", [
    {"instrument": "SH000001", "anchor_date": "2026-07-18", "status": "OBSERVED", "cash_close": "10"},
    {"instrument": "SH000001", "anchor_date": ANCHOR, "status": "MISSING", "cash_close": "10"},
    {"instrument": "SH000001", "anchor_date": ANCHOR, "status": "OBSERVED", "cash_close": None},
    {"instrument": "SH000001", "anchor_date": ANCHOR, "status": "OBSERVED", "cash_close": "0"},
    {"instrument": "SH000001", "anchor_date": ANCHOR, "status": "OBSERVED", "cash_close": "Infinity"},
    {"instrument": "SH000001", "anchor_date": ANCHOR, "status": "UNKNOWN", "cash_close": None},
])
def test_price_row_status_date_and_value_contract(row):
    with pytest.raises(IntentPlanningContractError):
        AnchorPriceSnapshot.from_iterable(
            anchor_date=ANCHOR, requested_instruments=("SH000001",), rows=(row,),
        )


def test_decimal_to_float_overflow_fails_closed():
    row = {
        "instrument": "SH000001", "anchor_date": ANCHOR,
        "status": "OBSERVED", "cash_close": "1e10000",
    }
    with pytest.raises(IntentPlanningContractError, match="production arithmetic"):
        AnchorPriceSnapshot.from_iterable(
            anchor_date=ANCHOR, requested_instruments=("SH000001",), rows=(row,),
        )

    bound_overflow = dict(row, cash_close="1.7e308")
    with pytest.raises(IntentPlanningContractError, match="production arithmetic"):
        AnchorPriceSnapshot.from_iterable(
            anchor_date=ANCHOR,
            requested_instruments=("SH000001",),
            rows=(bound_overflow,),
        )


def test_request_identity_dates_and_requested_price_set_are_exact():
    ranked = ranking()
    prior = state()
    prices = snapshot(ranked, prior)
    with pytest.raises(IntentPlanningContractError):
        plan(ranking_value=ranked, prior=prior, prices=prices, cycle_id=" ")
    with pytest.raises(IntentPlanningContractError, match="precede"):
        plan(ranking_value=ranked, prior=prior, prices=prices, trade_date=ANCHOR)
    wrong = AnchorPriceSnapshot.from_iterable(
        anchor_date=ANCHOR,
        requested_instruments=("SH000001",),
        rows=({"instrument": "SH000001", "anchor_date": ANCHOR, "status": "OBSERVED", "cash_close": "10"},),
    )
    with pytest.raises(IntentPlanningContractError, match="requested set"):
        plan(ranking_value=ranked, prior=prior, prices=wrong)


def test_wrong_portfolio_and_future_prior_are_denied():
    wrong = ShadowPortfolioState.from_dict({
        "portfolio_id": "other", "as_of_date": "2026-07-16", "cash": "1", "positions": [],
    })
    with pytest.raises(IntentPlanningContractError, match="portfolio"):
        plan(prior=wrong, prices=snapshot(ranking(), wrong))
    future = state(as_of="2026-07-18")
    with pytest.raises(IntentPlanningContractError, match="later"):
        plan(prior=future, prices=snapshot(ranking(), future))


def test_noncanonical_or_zero_scored_ranking_is_denied():
    placeholder = AnchorPriceSnapshot.from_iterable(
        anchor_date=ANCHOR,
        requested_instruments=("SH000001",),
        rows=({"instrument": "SH000001", "anchor_date": ANCHOR, "status": "OBSERVED", "cash_close": "10"},),
    )
    with pytest.raises(IntentPlanningContractError, match="RankingResult"):
        plan(ranking_value={}, prices=placeholder)
    empty_scores = ranking(scores={})
    prior = state()
    # No scored members means the requested price set would otherwise depend only on holdings.
    prices = AnchorPriceSnapshot.from_iterable(
        anchor_date=ANCHOR,
        requested_instruments=("SH999999",),
        rows=({"instrument": "SH999999", "anchor_date": ANCHOR, "status": "MISSING", "cash_close": None},),
    )
    with pytest.raises(IntentPlanningContractError, match="no scored"):
        plan(ranking_value=empty_scores, prior=prior, prices=prices)


def test_forged_ranking_state_price_and_definition_are_revalidated():
    ranked = ranking()
    prior = state()
    prices = snapshot(ranked, prior)
    definition = CurrentRuleIntentDefinition.from_dict(definition_raw())
    object.__setattr__(ranked, "scored_count", 99)
    with pytest.raises(IntentPlanningContractError, match="revalidation"):
        plan(ranking_value=ranked, prior=prior, prices=prices, definition=definition)
    ranked = ranking()
    object.__setattr__(prior, "cash", Decimal("-1"))
    with pytest.raises(ShadowAccountingContractError):
        plan(ranking_value=ranked, prior=prior, prices=snapshot(ranked, state()), definition=definition)
    prior = state()
    prices = snapshot(ranked, prior)
    object.__setattr__(prices, "requested_instruments", ("SH000001",))
    with pytest.raises(IntentPlanningContractError):
        plan(ranking_value=ranked, prior=prior, prices=prices, definition=definition)
    object.__setattr__(definition, "topk", 0)
    with pytest.raises(IntentPlanningContractError):
        plan(ranking_value=ranked, prior=prior, prices=snapshot(ranked, prior), definition=definition)


def test_public_result_construction_and_replace_cannot_mint_authority():
    with pytest.raises(IntentPlanningContractError, match="planner-owned"):
        IntentPlanningResult()
    result = plan()
    with pytest.raises(IntentPlanningContractError):
        replace(result, parity_checked=False)


def test_terminal_batch_is_b0_valid_and_rejects_tampering():
    result = plan()
    copied = ShadowIntentBatch.from_iterable(
        portfolio_id=result.intents.portfolio_id,
        cycle_id=result.intents.cycle_id,
        trade_date=result.intents.trade_date,
        rows=[item._raw() for item in result.intents.intents],
    )
    assert copied.digest == result.intents.digest
    rows = [item._raw() for item in result.intents.intents]
    rows[0]["quantity"] = 0
    with pytest.raises(ShadowAccountingContractError):
        ShadowIntentBatch.from_iterable(
            portfolio_id=PORTFOLIO, cycle_id=CYCLE, trade_date=TRADE, rows=rows,
        )


def test_malformed_analysis_count_is_denied(monkeypatch):
    import quantpits.research.intents as module

    original = module._production_generator_class()

    class Bad(original):
        def analyze_positions_with_universe(self, *args, **kwargs):
            result = super().analyze_positions_with_universe(*args, **kwargs)
            object.__setattr__(result, "target_buy_count", result.target_buy_count + 1)
            return result

    monkeypatch.setattr(module, "_production_generator_class", lambda: Bad)
    with pytest.raises(IntentPlanningParityError, match="counts"):
        plan()


def test_malformed_analysis_representation_and_candidate_partition_are_denied(monkeypatch):
    import quantpits.research.intents as module

    original = module._production_generator_class()

    class ListTerminal(original):
        def analyze_positions_with_universe(self, *args, **kwargs):
            result = super().analyze_positions_with_universe(*args, **kwargs)
            object.__setattr__(result, "eligible_unscored_instruments", [])
            return result

    monkeypatch.setattr(module, "_production_generator_class", lambda: ListTerminal)
    with pytest.raises(IntentPlanningParityError, match="exact sorted unique tuple"):
        plan()

    class MissingCandidate(original):
        def analyze_positions_with_universe(self, *args, **kwargs):
            result = super().analyze_positions_with_universe(*args, **kwargs)
            object.__setattr__(result, "buy_candidates", result.buy_candidates.iloc[1:])
            return result

    monkeypatch.setattr(module, "_production_generator_class", lambda: MissingCandidate)
    with pytest.raises(IntentPlanningParityError, match="counts"):
        plan()


def test_generated_sell_mismatch_and_buy_overlap_are_denied(monkeypatch):
    import quantpits.research.intents as module

    original = module._production_generator_class()
    prior = state(positions=(("SH999999", 100, "500"),))

    class MissingSell(original):
        def generate_sell_orders(self, *args, **kwargs):
            return [], 0.0

    monkeypatch.setattr(module, "_production_generator_class", lambda: MissingSell)
    with pytest.raises(IntentPlanningParityError, match="generated sells"):
        plan(prior=prior, prices=snapshot(ranking(), prior))

    class BadSellAmount(original):
        def generate_sell_orders(self, *args, **kwargs):
            orders, amount = super().generate_sell_orders(*args, **kwargs)
            return orders, amount + 0.01

    monkeypatch.setattr(module, "_production_generator_class", lambda: BadSellAmount)
    with pytest.raises(IntentPlanningParityError, match="estimated sell amount"):
        plan(prior=prior, prices=snapshot(ranking(), prior))

    class OverlapBuy(original):
        def generate_buy_orders(self, candidates, count, cash, trade_date):
            return [{
                "instrument": "SH999999", "datetime": trade_date, "value": 100,
                "estimated_amount": 1000.0, "score": 0.1, "current_close": 10.0,
            }]

    monkeypatch.setattr(module, "_production_generator_class", lambda: OverlapBuy)
    with pytest.raises(IntentPlanningParityError, match="generated buys"):
        plan(prior=prior, prices=snapshot(ranking(), prior))


def test_proposal_quantity_and_audit_field_tampering_are_denied(monkeypatch):
    import quantpits.research.intents as module

    original = module._production_generator_class()
    prior = state(positions=(("SH999999", 1000, "5000"),))

    class BadSellQuantity(original):
        def generate_sell_orders(self, *args, **kwargs):
            orders, amount = super().generate_sell_orders(*args, **kwargs)
            orders[0]["value"] += 1
            return orders, amount

    monkeypatch.setattr(module, "_production_generator_class", lambda: BadSellQuantity)
    with pytest.raises(IntentPlanningParityError, match="generated sells"):
        plan(prior=prior, prices=snapshot(ranking(), prior))

    class BadBuyEstimate(original):
        def generate_buy_orders(self, *args, **kwargs):
            orders = super().generate_buy_orders(*args, **kwargs)
            orders[0]["estimated_amount"] += 0.01
            return orders

    monkeypatch.setattr(module, "_production_generator_class", lambda: BadBuyEstimate)
    with pytest.raises(IntentPlanningParityError, match="generated buys"):
        plan()


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_primitive_exceptions_and_process_control_propagate(monkeypatch, error):
    import quantpits.research.intents as module

    class Exploding:
        def __init__(self, *args, **kwargs):
            raise error

    monkeypatch.setattr(module, "_production_generator_class", lambda: Exploding)
    with pytest.raises(type(error)):
        plan()


def test_result_float_fields_are_deterministic_text_and_result_is_deep_immutable():
    result = plan()
    for proposal in result.production_buy_suggestions:
        assert isinstance(proposal["score"], str)
        assert isinstance(proposal["current_close"], str)
        assert isinstance(proposal["estimated_amount"], str)
        with pytest.raises(TypeError):
            proposal["quantity"] = 1
    with pytest.raises(TypeError):
        result.holding_classifications["continuing"] = ()


def test_module_has_no_future_settlement_or_io_public_inputs():
    import inspect
    import quantpits.research.intents as module

    signature = inspect.signature(module.CurrentRuleShadowIntentPlanner.plan)
    forbidden = {"next_open", "execution_price", "fee", "slippage", "path", "workspace"}
    assert forbidden.isdisjoint(signature.parameters)
    source = inspect.getsource(module)
    for forbidden_import in ("import qlib", "import mlflow", "subprocess", "open(", "Path(", "random", "time.time"):
        assert forbidden_import not in source
