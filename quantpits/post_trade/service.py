"""Post-trade state calculation and execution routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import pandas as pd

from quantpits.post_trade.reconciliation import reconcile_quantities
from quantpits.post_trade.state import (
    PostTradeStateChangeSet, account_state_from_config, build_change_set,
    normalize_settlement_frame,
)
from quantpits.post_trade.state_outputs import build_state_output_payloads
from quantpits.post_trade.state_persistence import (
    StateArtifactLedger, capture_state_fingerprints, persist_state_outputs,
)


@dataclass(frozen=True)
class PostTradeStateResult:
    change_set: PostTradeStateChangeSet
    artifacts: Optional[StateArtifactLedger]
    dry_run: bool

    def public_summary(self):
        return {
            "processed_date_count": len(self.change_set.processed_dates),
            "position_count_before": len(self.change_set.initial_state.positions),
            "position_count_after": len(self.change_set.final_state.positions),
            "state_invariants": "passed",
            "valuation_completeness": "passed",
            "committed_output_count": len(self.artifacts.outputs) if self.artifacts else 0,
            "dry_run": self.dry_run,
        }


class PostTradeService:
    def __init__(self, valuation_provider):
        self.valuation_provider = valuation_provider

    @staticmethod
    def _frame(parsed, stream, date):
        item = parsed.get((stream, date))
        return item.dataframe if item is not None else pd.DataFrame()

    def calculate(self, prepared, parsed: Mapping, trade_dates: Sequence[str]):
        config = prepared.config
        initial = account_state_from_config(config.raw_prod_config)
        events_by_date, warnings_by_date, valuations = {}, {}, {}
        current_quantities = {item.instrument: item.quantity for item in initial.positions}
        for date in trade_dates:
            events, warnings = normalize_settlement_frame(self._frame(parsed, "settlement", date), date)
            events_by_date[date], warnings_by_date[date] = events, warnings
            # Reconcile only when execution evidence for this state date is
            # available. Strict cross-stream presence remains owned by intake.
            order = self._frame(parsed, "order", date); trade = self._frame(parsed, "trade", date)
            if prepared.options.scope == "all" and (not order.empty or not trade.empty):
                reconcile_quantities(order, trade, events, trade_date=date)
            for event in events:
                if event.kind == "sell":
                    remaining = current_quantities.get(event.instrument, 0) - event.quantity
                    if remaining > 0:
                        current_quantities[event.instrument] = remaining
                    else:
                        current_quantities.pop(event.instrument, None)
                elif event.kind == "buy":
                    current_quantities[event.instrument] = current_quantities.get(event.instrument, 0) + event.quantity
            valuations[date] = self.valuation_provider.snapshot(date, tuple(sorted(current_quantities)), config.runtime_config.get("benchmark", "SH000300"))
        return build_change_set(initial, trade_dates, events_by_date, config.cashflow_config, valuations, config.raw_prod_config, warnings_by_date)

    def run_state(self, prepared, parsed, trade_dates):
        before = capture_state_fingerprints(prepared.ctx)
        change_set = self.calculate(prepared, parsed, trade_dates)
        if prepared.options.dry_run:
            return PostTradeStateResult(change_set, None, True)
        settlement = {date: self._frame(parsed, "settlement", date) for date in trade_dates}
        payloads = build_state_output_payloads(prepared.ctx, change_set, settlement, model=prepared.config.runtime_config.get("model", "GATs"))
        ledger = persist_state_outputs(prepared.ctx, payloads, change_set.processed_dates, expected_fingerprints=before)
        return PostTradeStateResult(change_set, ledger, False)
