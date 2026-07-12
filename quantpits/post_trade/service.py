"""Post-trade state calculation and execution routing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Optional, Sequence

import pandas as pd

from quantpits.post_trade.reconciliation import reconcile_quantities
from quantpits.post_trade.state import (
    PostTradeStateChangeSet, account_state_from_config, build_change_set,
    normalize_settlement_frame,
)
from quantpits.post_trade.state_outputs import build_state_output_payloads
from quantpits.post_trade.state_persistence import (
    StateArtifactLedger, capture_state_fingerprints, transaction_payloads,
)
from quantpits.post_trade.cashflow import build_cashflow_commit
from quantpits.post_trade.transaction import PostTradeTransactionManager, sha256_bytes
from quantpits.runtime import generate_run_id
from quantpits.utils.workspace import fingerprint_value


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
            if prepared.options.scope == "all":
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
        if not change_set.processed_dates:
            return PostTradeStateResult(change_set, StateArtifactLedger((), (), False), False)
        settlement = {date: self._frame(parsed, "settlement", date) for date in trade_dates}
        if before != capture_state_fingerprints(prepared.ctx):
            from quantpits.post_trade.contracts import PostTradeStateConflictError
            raise PostTradeStateConflictError("Post-trade state inputs changed before commit")
        cashflow = build_cashflow_commit(prepared.config.cashflow_config, change_set.processed_dates)
        payloads = build_state_output_payloads(
            prepared.ctx, change_set, settlement,
            model=prepared.config.runtime_config.get("model", "GATs"),
            cashflow_config=prepared.config.cashflow_config,
        )
        manager = PostTradeTransactionManager(prepared.ctx)
        run_id = prepared.options.run_id or generate_run_id("post_trade")
        targets = transaction_payloads(prepared.ctx, payloads)
        def _frame_fingerprint(item):
            frame = item.dataframe
            canonical = frame.sort_index(axis=1).to_json(
                orient="split", date_format="iso", date_unit="ns", force_ascii=True,
            )
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        initial = change_set.initial_state
        transition_evidence = tuple(
            {
                "date": transition.trade_date,
                "events": tuple(
                    (
                        event.trade_date, event.instrument, event.kind,
                        str(event.quantity), str(event.price), str(event.gross_amount),
                        str(event.cash_effect), event.source_row, event.normalized_trade_type,
                    )
                    for event in transition.settlement_events
                ),
                "valuation": (
                    tuple((instrument, str(close)) for instrument, close in transition.valuation.closes),
                    str(transition.valuation.benchmark),
                ),
            }
            for transition in change_set.transitions
        )
        resolved = fingerprint_value({
            "schema": 1,
            "light": prepared.plan_fingerprint,
            "dates": tuple(change_set.processed_dates),
            "sources": tuple(
                (item.stream, item.trade_date, item.display_path, item.status, item.fingerprint)
                for stream in ("settlement", "order", "trade")
                for item in prepared.catalog.sources_for(stream)
            ),
            "parsed_evidence": tuple(
                (stream, date, _frame_fingerprint(item))
                for (stream, date), item in sorted(parsed.items())
            ),
            "initial_state": (
                initial.as_of_date, str(initial.cash),
                tuple((item.instrument, str(item.quantity), str(item.cost)) for item in initial.positions),
            ),
            "transitions": transition_evidence,
            "cashflow_source": fingerprint_value(prepared.config.cashflow_config),
            "baselines": before,
            "targets": tuple(
                (order, role, path.resolve().relative_to(prepared.ctx.root).as_posix(), sha256_bytes(payload))
                for order, role, path, payload in targets
            ),
        })
        journal = manager.prepare(
            transaction_id=run_id, run_id=run_id, scope=prepared.options.scope,
            light_fingerprint=prepared.plan_fingerprint, resolved_fingerprint=resolved,
            cursor_before=prepared.config.state_cursor,
            cursor_after=change_set.processed_dates[-1],
            processed_dates=change_set.processed_dates,
            consumed_cashflow_dates=cashflow.consumed_dates,
            payloads=targets,
        )
        try:
            journal = manager.commit(journal)
        except Exception as exc:
            from quantpits.post_trade.contracts import (
                PostTradeTransactionError, PostTradeTransactionRecoveryError,
            )
            try:
                current = manager.load(journal.transaction_id)
            except PostTradeTransactionError:
                current = journal
            verified = manager.verified_target_paths(current)
            if isinstance(exc, PostTradeTransactionError):
                # Preserve precise conflict/corruption types while attaching
                # enough context for failure audit and operator recovery.
                exc.transaction_id = current.transaction_id
                exc.committed_outputs = verified
                raise
            raise PostTradeTransactionRecoveryError(
                "Post-trade transaction was interrupted and requires recovery: %s" % exc,
                transaction_id=current.transaction_id,
                committed_outputs=verified,
            ) from exc
        paths = tuple(prepared.ctx.root / item.path for item in journal.artifacts)
        ledger = StateArtifactLedger(paths, tuple(change_set.processed_dates), True, journal.transaction_id)
        return PostTradeStateResult(change_set, ledger, False)
