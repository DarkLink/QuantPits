"""Single-account bootstrap → plan → settle → close valuation/report service."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from quantpits.research.paper_accounting import (
    PaperError, ZERO, amount, apply_events, date, decimal, execution_config,
    settle_plan, state, token, validate_event, validate_rules, valuation, cash_context,
)
from quantpits.research.paper_market import PaperMarket, load_signal, strict_json
from quantpits.research.paper_planning import make_plan, strategy_config
from quantpits.research.paper_store import (
    PaperStore, buy_locks, cashflow, has_external_funding, imported_state, latest_state, price_cache, used_events,
)
from quantpits.training.persistence import atomic_write_bytes, sha256_bytes
from quantpits.utils.config_loader import load_workspace_config
from quantpits.utils.workspace import WorkspaceContext, fingerprint_value


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class PaperWorkflow:
    def __init__(self, ctx, *, config_path=None):
        self.ctx = ctx
        path = Path(config_path) if config_path else ctx.config_path("paper_config.json")
        if not path.is_absolute():
            path = ctx.root / path
        self.config = strict_json(path.read_bytes())
        self.account_id = token(self.config["account_id"])
        self.store = PaperStore(ctx, self.account_id)

    def _input_path(self, value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.ctx.root / path).resolve(strict=True)

    def _market(self):
        return PaperMarket(self.ctx.qlib_data_dir)

    def _journal(self):
        journal = self.store.read()
        if journal is None:
            raise PaperError("bootstrap the paper account first")
        return journal

    @staticmethod
    def _settled(journal):
        return {r["result"]["cycle_id"]: r["result"] for r in journal["records"] if r["kind"] == "SETTLEMENT"}

    @classmethod
    def _pending(cls, journal):
        completed = cls._settled(journal)
        return [p for key, p in journal["plans"].items() if key not in completed and key not in journal["cancelled_cycles"]]

    @staticmethod
    def _events(journal, through):
        consumed = used_events(journal)
        return sorted((e for key, e in journal["events"].items()
                       if key not in consumed and e["booking_date"] <= through),
                      key=lambda e: (e["booking_date"], e["event_id"]))

    @cash_context
    def bootstrap(self, source_config, selected):
        date(selected)
        source_path = self._input_path(source_config)
        raw = source_path.read_bytes()
        source_sha = sha256_bytes(raw)
        with self.store.locked():
            existing = self.store.read()
            if existing is not None:
                original = existing["bootstrap"]
                if (original["date"], original["source_path"], original["source_sha256"]) != (
                    selected, str(source_path), source_sha,
                ):
                    raise PaperError("account already bootstrapped from a different source")
                return {"status": "REUSED", "bootstrap": original}
            snapshot = strict_json(raw)
            initial = imported_state(snapshot, self.account_id, selected)
            market = self._market()
            if market.check_date(selected, actual=True) != "VERIFIED":
                return {"status": "WAITING_FOR_DATA", "date": selected}
            closes = market.closes(selected, [p["instrument"] for p in initial["positions"]])
            if any(closes.get(p["instrument"]) is None for p in initial["positions"]):
                raise PaperError("bootstrap requires reliable closes for every initial holding")
            nav = decimal(initial["cash"]) + sum((
                p["quantity"] * decimal(closes[p["instrument"]]) for p in initial["positions"]
            ), ZERO)
            if nav <= ZERO:
                raise PaperError("initial NAV must be positive")
            baseline = valuation(initial, selected, closes, {}, amount(nav), "0.00")
            bootstrap = {"date": selected, "copied_at": now(), "source_path": str(source_path),
                         "source_sha256": source_sha, "source_snapshot": snapshot, "source_json": raw.decode("utf-8"),
                         "state": initial, "closes": closes, "initial_nav": amount(nav),
                         "valuation": baseline, "price_receipts": list(market.receipts)}
            journal = {"schema_version": 1, "account_id": self.account_id, "bootstrap": bootstrap,
                       "plans": {}, "cancelled_cycles": {}, "events": {}, "records": [], "valuations": []}
            self.store.write(journal)
            return {"status": "BOOTSTRAPPED", "bootstrap": bootstrap}

    @cash_context
    def plan(self, cycle_id, anchor, trade_date, *, source_override=None, rules=None):
        token(cycle_id)
        date(anchor)
        date(trade_date)
        source = dict(self.config["signal_source"])
        source.update(source_override or {})
        signal_root = Path(source.pop("workspace", str(self.ctx.root))).expanduser()
        signal_root = (signal_root if signal_root.is_absolute() else self.ctx.root / signal_root).resolve()
        signal_uri = source.pop("mlflow_uri", self.ctx.mlflow_uri if signal_root == self.ctx.root else None)
        signal_ctx = WorkspaceContext.from_root(
            signal_root, mlflow_uri=signal_uri,
            qlib_data_dir=self.ctx.qlib_data_dir, qlib_region=self.ctx.qlib_region,
        )
        if "prediction" in (source_override or {}):
            source.pop("recorder_id", None)
            source.pop("experiment_id", None)
        elif "recorder_id" in (source_override or {}):
            source.pop("prediction", None)
        with self.store.locked():
            journal = self._journal()
            if cycle_id in journal["cancelled_cycles"]:
                raise PaperError("cycle was explicitly cancelled; select a new cycle ID")
            old = journal["plans"].get(cycle_id)
            if old is not None:
                if (old["anchor"], old["trade_date"], old["source"]["requested_source"], old["source"]["workspace"]) != (
                    anchor, trade_date, source, str(signal_ctx.root),
                ) or (old["source"]["kind"] == "MLFLOW" and old["source"]["mlflow_uri"] != signal_ctx.mlflow_uri):
                    raise PaperError("cycle already planned with different dates/source")
                if rules is not None and rules != old["market_rules"]:
                    raise PaperError("cycle already planned with different rules")
                return {"status": "REUSED", "plan": old}
            if self._pending(journal):
                raise PaperError("settle the pending cycle before planning another")
            previous = latest_state(journal)
            if journal["valuations"] and anchor < journal["valuations"][-1]["result"]["date"]:
                raise PaperError("anchor precedes the last account valuation")
            # Events already known at the signal anchor are part of its starting account.
            events = self._events(journal, anchor)
            if events:
                after, external = apply_events(previous, events)
                journal["records"].append({"kind": "EVENTS", "result": {
                    "prior_digest": state(previous).digest, "after": after,
                    "events": events, "external_cashflow": external}})
                previous = after
            market = self._market()
            if market.check_date(anchor, actual=True) != "VERIFIED":
                return {"status": "WAITING_FOR_DATA", "date": anchor}
            calendar_status = market.check_date(trade_date)
            merged = load_workspace_config(self.ctx.root)
            strategy = strategy_config(merged["strategy"])
            pool, pool_source = market.universe(merged["market"], anchor)
            scores, identity = load_signal(signal_ctx, source, anchor)
            identity["universe_source"] = pool_source
            identity["provider_root"] = str(market.root)
            closes = market.closes(anchor, set(pool) | {p["instrument"] for p in previous["positions"]})
            identity["price_receipts"] = list(market.receipts)
            selected_rules = rules if rules is not None else dict(self.config["market_rules"], date=trade_date)
            validate_rules(selected_rules, trade_date)
            plan = make_plan(account_id=self.account_id, cycle_id=cycle_id, anchor=anchor,
                             trade_date=trade_date, generated_at=now(), prior=previous,
                             market=merged["market"], universe=pool, scores=scores, closes=closes,
                             strategy=strategy, execution=self.config["execution"], source=identity,
                             market_rules=selected_rules, calendar_status=calendar_status,
                             predecessor_record_count=len(journal["records"]))
            journal["plans"][cycle_id] = plan
            self.store.write(journal)
            return {"status": "PLANNED", "plan": plan}

    @cash_context
    def settle(self, cycle_id, *, rules=None):
        token(cycle_id)
        with self.store.locked():
            journal = self._journal()
            old = self._settled(journal).get(cycle_id)
            if old is not None:
                if rules is not None and rules != old["rules"]:
                    raise PaperError("cycle already settled using different rules")
                return {"status": "REUSED", "settlement": old}
            if cycle_id not in journal["plans"]:
                raise PaperError("unknown paper cycle")
            if cycle_id in journal["cancelled_cycles"]:
                raise PaperError("cycle was explicitly cancelled")
            plan = journal["plans"][cycle_id]
            prior = latest_state(journal)
            if prior != plan["prior"] or len(journal["records"]) != plan["predecessor_record_count"]:
                raise PaperError("plan predecessor differs from current account")
            market = self._market()
            if market.check_date(plan["trade_date"], actual=True) != "VERIFIED":
                return {"status": "WAITING_FOR_DATA", "cycle_id": cycle_id, "date": plan["trade_date"]}
            requested = sorted({p["instrument"] for p in prior["positions"]} |
                               {p["instrument"] for p in plan["buys"]})
            quotes = market.execution_quotes(plan["trade_date"], requested)
            selected_rules = rules if rules is not None else plan["market_rules"]
            validate_rules(selected_rules, plan["trade_date"])
            if requested and all(q["open"] is None for q in quotes.values()):
                # No quote for the whole requested execution surface is not proof
                # that every order failed. Keep the cycle open for provider repair.
                from quantpits.research.paper_accounting import board
                def suspended(instrument):
                    rule = dict(selected_rules["boards"].get(board(instrument), {}))
                    rule.update(selected_rules["instruments"].get(instrument, {}))
                    return rule.get("suspended") is True
                if not all(suspended(i) for i in requested):
                    return {"status": "WAITING_FOR_DATA", "reason": "EXECUTION_OPEN_SURFACE_MISSING",
                            "cycle_id": cycle_id, "date": plan["trade_date"]}
            events = self._events(journal, plan["trade_date"])
            result = settle_plan(plan, prior, quotes, selected_rules,
                                 events, buy_locks(journal))
            journal["records"].append({"kind": "SETTLEMENT", "result": result, "recorded_at": now(),
                                       "calendar": list(market.calendar)})
            self.store.write(journal)
            return {"status": "SETTLED", "settlement": result}

    def cancel(self, cycle_id, reason):
        """Explicitly retire a pending plan while retaining its original facts."""
        token(cycle_id)
        if not isinstance(reason, str) or not reason.strip():
            raise PaperError("an explicit cancellation reason is required")
        with self.store.locked():
            journal = self._journal()
            if cycle_id in self._settled(journal) or cycle_id not in journal["plans"]:
                raise PaperError("only a known unsettled cycle can be cancelled")
            old = journal["cancelled_cycles"].get(cycle_id)
            if old is not None:
                if old["reason"] != reason:
                    raise PaperError("cycle was cancelled with a different reason")
                return {"status": "REUSED", "cycle_id": cycle_id, "cancellation": old}
            cancellation = {"reason": reason, "cancelled_at": now()}
            journal["cancelled_cycles"][cycle_id] = cancellation
            self.store.write(journal)
            return {"status": "CANCELLED", "cycle_id": cycle_id, "cancellation": cancellation}

    def add_event(self, event):
        validate_event(event)
        with self.store.locked():
            journal = self._journal()
            old = journal["events"].get(event["event_id"])
            if old is not None:
                if old != event:
                    raise PaperError("event ID already used for different facts")
                return {"status": "REUSED", "event": old}
            frontier = max([state(latest_state(journal)).as_of_date] +
                           [v["result"]["date"] for v in journal["valuations"]])
            if event["booking_date"] < frontier:
                raise PaperError("late event requires an explicit booking date at/after account frontier")
            if event["effective_date"] <= journal["bootstrap"]["date"]:
                raise PaperError("event must belong to this paper account after bootstrap")
            journal["events"][event["event_id"]] = event
            self.store.write(journal)
            return {"status": "EVENT_RECORDED", "event": event}

    @cash_context
    def value(self, selected):
        date(selected)
        with self.store.locked():
            journal = self._journal()
            if any(p["trade_date"] <= selected for p in self._pending(journal)):
                raise PaperError("settle the due plan before valuing the account")
            if journal["valuations"] and selected < journal["valuations"][-1]["result"]["date"]:
                raise PaperError("valuation dates must not move backwards")
            market = self._market()
            if market.check_date(selected, actual=True) != "VERIFIED":
                return {"status": "WAITING_FOR_DATA", "date": selected}
            previous = latest_state(journal)
            events = self._events(journal, selected)
            if events:
                if self._pending(journal):
                    raise PaperError("value would change the predecessor of a pending plan")
                after, external = apply_events(previous, events)
                journal["records"].append({"kind": "EVENTS", "result": {
                    "prior_digest": state(previous).digest, "after": after,
                    "events": events, "external_cashflow": external}})
                previous = after
            closes = market.closes(selected, [p["instrument"] for p in previous["positions"]])
            result = valuation(previous, selected, closes, price_cache(journal, selected),
                               journal["bootstrap"]["initial_nav"], cashflow(journal), has_external_funding(journal))
            if (journal["valuations"] and journal["valuations"][-1]["result"] == result
                    and journal["valuations"][-1]["record_count"] == len(journal["records"])):
                return {"status": "REUSED", "valuation": result}
            journal["valuations"].append({"result": result, "observed_closes": closes, "recorded_at": now(),
                                          "price_receipts": list(market.receipts), "record_count": len(journal["records"])})
            self.store.write(journal)
            return {"status": "VALUED", "valuation": result}

    @cash_context
    def report(self, selected, *, since=None, reference_config=None,
               reference_start_config=None, reference_events=None):
        """Read-only paper and optional actual-account comparison, at an exact date."""
        date(selected)
        journal = self._journal()
        match = [v["result"] for v in journal["valuations"] if v["result"]["date"] == selected]
        if selected == journal["bootstrap"]["date"]:
            match.insert(0, journal["bootstrap"]["valuation"])
        paper = match[-1] if match else {"status": "WAITING_FOR_VALUATION", "date": selected}
        available = [journal["bootstrap"]["valuation"]] + [v["result"] for v in journal["valuations"]]
        previous_values = [v for v in available if v["date"] < selected and (since is None or v["date"] == since)]
        start = previous_values[-1] if previous_values else None
        if since is not None:
            date(since)
            if since >= selected or start is None:
                raise PaperError("--since requires a saved earlier paper valuation")
        weekly = {"status": "MISSING_BASELINE" if start is None else "WAITING_FOR_VALUATION"}
        if start is not None and start["nav"] is not None and paper.get("nav") is not None:
            net = decimal(paper["net_external_cashflow"]) - decimal(start["net_external_cashflow"])
            pnl = decimal(paper["nav"]) - decimal(start["nav"]) - net
            weekly = {"status": "STALE" if "STALE" in (start["status"], paper["status"]) else "COMPLETE",
                      "since": start["date"], "start_nav": start["nav"], "end_nav": paper["nav"],
                      "net_external_cashflow": amount(net), "pnl_excluding_external": amount(pnl),
                      "return_excluding_external": str(pnl / decimal(start["nav"])) if decimal(start["nav"]) > ZERO else None,
                      "return_method": "PNL_EXCLUDING_NET_EXTERNAL_OVER_PERIOD_START_NAV"}
        reference = {"status": "NOT_SELECTED" if reference_config is None else "PENDING_UPDATE", "date": selected}
        if reference_config is not None:
            path = self._input_path(reference_config)
            raw = path.read_bytes()
            snapshot = strict_json(raw)
            source_date = snapshot.get("last_processed_date", snapshot.get("current_date"))
            reference["observed_date"] = source_date
            if source_date == selected:
                reference_state = imported_state(snapshot, self.account_id, selected)
                market = self._market()
                if market.check_date(selected, actual=True) == "VERIFIED":
                    closes = market.closes(selected, [p["instrument"] for p in reference_state["positions"]])
                    reference = valuation(reference_state, selected, closes, {},
                                          journal["bootstrap"]["initial_nav"], "0.00")
                    # Actual cashflows cannot be inferred from a current prod_config snapshot.
                    for field in ("pnl_excluding_external", "return_excluding_external", "unit_nav", "net_external_cashflow"):
                        reference[field] = None
                    reference.update(source_sha256=sha256_bytes(raw), observed_date=source_date,
                                     performance_status="ACTUAL_CASHFLOW_REFERENCE_REQUIRED")
                    reference["period"] = {"status": "MISSING_ACTUAL_CASHFLOW_REFERENCE"}
                    if reference_events is not None and start is not None and reference.get("nav") is not None:
                        ids = set()
                        external = ZERO
                        for event in reference_events:
                            validate_event(event)
                            if event["event_id"] in ids:
                                raise PaperError("duplicate actual-reference event ID")
                            ids.add(event["event_id"])
                            if start["date"] < event["booking_date"] <= selected and event["kind"] == "EXTERNAL":
                                external += decimal(event["cash"])
                        start_nav = None
                        if start["date"] == journal["bootstrap"]["date"]:
                            start_nav = decimal(journal["bootstrap"]["initial_nav"])
                        if reference_start_config is not None:
                            start_path = self._input_path(reference_start_config)
                            start_raw = start_path.read_bytes()
                            start_state = imported_state(strict_json(start_raw), self.account_id, start["date"])
                            start_closes = market.closes(start["date"], [p["instrument"] for p in start_state["positions"]])
                            actual_start = valuation(start_state, start["date"], start_closes, {},
                                                     journal["bootstrap"]["initial_nav"], "0.00")
                            start_nav = None if actual_start["nav"] is None else decimal(actual_start["nav"])
                            reference["start_source_sha256"] = sha256_bytes(start_raw)
                        if start_nav is not None and start_nav > ZERO:
                            pnl = decimal(reference["nav"]) - start_nav - external
                            reference["period"] = {"status": "COMPLETE", "since": start["date"],
                                                   "start_nav": amount(start_nav), "end_nav": reference["nav"],
                                                   "net_external_cashflow": amount(external),
                                                   "pnl_excluding_external": amount(pnl),
                                                   "return_excluding_external": str(pnl / start_nav)}
                            reference["performance_status"] = "EXPLICIT_ACTUAL_CASHFLOW_REFERENCE"
                        else:
                            reference["period"] = {"status": "MISSING_ACTUAL_START_NAV"}
        fees = sum((decimal(r["result"]["total_fees"]) for r in journal["records"]
                    if r["kind"] == "SETTLEMENT" and r["result"]["after"]["as_of_date"] <= selected), ZERO)
        return {"status": "REPORT", "account_id": self.account_id, "date": selected,
                "paper": paper, "paper_period": weekly, "actual_reference": reference,
                "cumulative_simulated_fees": amount(fees),
                "events": [e for r in journal["records"] if r["result"]["after"]["as_of_date"] <= selected
                           for e in r["result"]["events"] if e["booking_date"] <= selected],
                "pending_events": [e for key, e in journal["events"].items() if key not in used_events(journal)],
                "corporate_actions": "ONLY_EXPLICIT_BOOKED_EVENTS_INCLUDED; UNKNOWN_EVENTS_NOT_INCLUDED",
                "execution_assumption": "CASH_OPEN_SELL_THEN_BUY_NO_VOLUME_MODEL",
                "drawdown": None, "journal_digest": fingerprint_value(journal)}

    def save_report(self, report):
        """Write derived, reviewable JSON/Markdown without changing the account."""
        selected = date(report["date"])
        if report["account_id"] != self.account_id:
            raise PaperError("report belongs to a different account")
        folder = self.ctx.output_path("paper")
        paths = [folder / ("report_%s.%s" % (selected, suffix)) for suffix in ("json", "md")]
        for path in paths:
            self.store._contained(path)
        json_bytes = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2,
                                allow_nan=False).encode("utf-8") + b"\n"
        lines = ["# Weekly paper account", "", "Account: `%s` — %s" % (self.account_id, selected), "",
                 "| Account | Status | Cash | Known market value | NAV |", "| --- | --- | ---: | ---: | ---: |"]
        for label, result in (("Paper", report["paper"]), ("Actual reference", report["actual_reference"])):
            lines.append("| %s | %s | %s | %s | %s |" % (
                label, result["status"], result.get("cash", "—"), result.get("known_market_value", "—"),
                result.get("nav") if result.get("nav") is not None else "—"))
        period = report["paper_period"]
        actual_period = report["actual_reference"].get("period", {"status": report["actual_reference"]["status"]})
        lines.extend(["", "Paper period: %s → %s; profit excluding net external cashflow: %s; return: %s." % (
            period.get("since", "—"), selected, period.get("pnl_excluding_external", "—"),
            period.get("return_excluding_external", "—")),
            "", "Actual period: %s; profit excluding net external cashflow: %s; return: %s." % (
                actual_period["status"], actual_period.get("pnl_excluding_external", "—"),
                actual_period.get("return_excluding_external", "—")),
            "", "Simulated fees already included in cash/cost: %s." % report["cumulative_simulated_fees"],
            "", "Execution uses cash open, sells before buys; no volume, queue or extra slippage model.",
            "Only explicitly booked corporate/cash events are included. Unknown events are not included.",
            "This observation does not establish daily drawdown.", "", "| Holding | Quantity | Book cost | Close | Price date | Stale | Value |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: |"])
        for row in report["paper"].get("positions", []):
            lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                row["instrument"], row["quantity"], row["book_cost"], row["price"] or "—", row["price_date"] or "—",
                row["stale"], row["market_value"] if row["market_value"] is not None else "—"))
        markdown = ("\n".join(lines) + "\n").encode("utf-8")
        for path, payload in zip(paths, (json_bytes, markdown)):
            self.store._contained(path)
            atomic_write_bytes(path, payload)
            if path.read_bytes() != payload:
                raise PaperError("report write post-state is uncertain")
        return [str(p) for p in paths]
