"""One authoritative paper journal and workspace-scoped execution lease."""
from __future__ import annotations

import json
from contextlib import contextmanager

from quantpits.post_trade.state import account_state_from_config
from quantpits.research.paper_accounting import (
    PaperError, ZERO, amount, apply_events, date, decimal, settle_plan, state,
    token, validate_event, valuation, cash_context,
)
from quantpits.research.paper_market import strict_json, verify_closes, verify_price_row
from quantpits.research.paper_planning import validate_plan
from quantpits.training.lease import TrainingExecutionLease
from quantpits.training.persistence import atomic_write_json_bytes, sha256_bytes
from quantpits.utils.workspace import fingerprint_value


def imported_state(snapshot, account_id, selected):
    """Reuse production quantity/cost conversion, without importing its writer."""
    if not isinstance(snapshot, dict) or not {"current_cash", "current_holding"} <= set(snapshot):
        raise PaperError("source account requires explicit current_cash/current_holding")
    if not isinstance(snapshot["current_holding"], list):
        raise PaperError("source account holdings must be a list")
    source = account_state_from_config(snapshot)
    if str(source.as_of_date) != selected:
        raise PaperError("source account date does not match bootstrap date")
    positions = []
    for p in source.positions:
        if p.quantity != int(p.quantity) or p.quantity <= ZERO:
            raise PaperError("bootstrap requires positive integer shares")
        positions.append({"instrument": p.instrument, "quantity": int(p.quantity),
                          "book_cost": amount(p.cost)})
    return state({"portfolio_id": account_id, "as_of_date": selected,
                  "cash": amount(source.cash), "positions": sorted(positions, key=lambda p: p["instrument"])}).to_dict()


def latest_state(journal):
    return journal["records"][-1]["result"]["after"] if journal["records"] else journal["bootstrap"]["state"]


def used_events(journal):
    return {e["event_id"] for r in journal["records"] for e in r["result"]["events"]}


def cashflow(journal):
    return amount(sum((decimal(r["result"]["external_cashflow"]) for r in journal["records"]), ZERO))


def has_external_funding(journal):
    return any(e["kind"] == "EXTERNAL" for r in journal["records"] for e in r["result"]["events"])


def price_cache(journal, before=None):
    cache = {}
    values = [journal["bootstrap"]["valuation"]] + journal["valuations"]
    for item in values:
        item = item.get("result", item)
        if before is not None and item["date"] > before:
            continue
        for row in item["positions"]:
            if row["price"] is not None and not row["stale"]:
                cache[row["instrument"]] = {"price": row["price"], "price_date": row["price_date"]}
    return cache


def buy_locks(journal):
    locks = {}
    for record in journal["records"]:
        if record["kind"] != "SETTLEMENT":
            continue
        result = record["result"]
        for row in result["decisions"]:
            if row["status"] == "FILLED" and row["side"] == "BUY":
                locks[row["instrument"]] = {"date": result["after"]["as_of_date"], "quantity": row["quantity"]}
    return locks


@cash_context
def validate_journal(journal, account_id):
    if set(journal) != {"schema_version", "account_id", "bootstrap", "plans", "cancelled_cycles",
                        "events", "records", "valuations"}:
        raise PaperError("paper journal schema mismatch")
    if type(journal["schema_version"]) is not int or journal["schema_version"] != 1 or journal["account_id"] != token(account_id):
        raise PaperError("paper account identity mismatch")
    bootstrap = journal["bootstrap"]
    source_raw = bootstrap["source_json"].encode("utf-8")
    if sha256_bytes(source_raw) != bootstrap["source_sha256"] or strict_json(source_raw) != bootstrap["source_snapshot"]:
        raise PaperError("bootstrap source snapshot/hash mismatch")
    selected = date(bootstrap["date"])
    current = imported_state(bootstrap["source_snapshot"], account_id, selected)
    verify_closes(bootstrap["price_receipts"], selected, bootstrap["closes"])
    if current != bootstrap["state"]:
        raise PaperError("bootstrap state inconsistent with source snapshot")
    initial = valuation(current, selected, bootstrap["closes"], {}, bootstrap["initial_nav"], "0.00")
    if initial != bootstrap["valuation"] or initial["status"] != "COMPLETE" or initial["unit_nav"] != "1":
        raise PaperError("bootstrap valuation inconsistent")
    if decimal(bootstrap["initial_nav"]) <= ZERO:
        raise PaperError("initial NAV must be positive")
    states = {0: (current, "0.00", False)}
    for key, plan in journal["plans"].items():
        if token(key) != plan["cycle_id"] or plan["account_id"] != account_id:
            raise PaperError("plan identity mismatch")
        validate_plan(plan)
    for key, event in journal["events"].items():
        validate_event(event)
        if key != event["event_id"]:
            raise PaperError("event identity mismatch")
    for key, cancellation in journal["cancelled_cycles"].items():
        if key not in journal["plans"] or set(cancellation) != {"reason", "cancelled_at"} or not cancellation["reason"].strip():
            raise PaperError("invalid cancelled cycle")
    settled, consumed, total_external = set(), set(), ZERO
    replay = dict(journal, records=[])
    for record in journal["records"]:
        result = record["result"]
        if result["prior_digest"] != state(current).digest:
            raise PaperError("record predecessor mismatch")
        events = result["events"]
        event_ids = [e["event_id"] for e in events]
        if len(set(event_ids)) != len(event_ids) or consumed.intersection(event_ids):
            raise PaperError("event applied more than once")
        for event in events:
            if journal["events"].get(event["event_id"]) != event or event["booking_date"] < state(current).as_of_date:
                raise PaperError("event source/date mismatch")
        if record["kind"] == "SETTLEMENT":
            cycle = result["cycle_id"]
            plan = journal["plans"].get(cycle)
            if plan is None or cycle in settled or cycle in journal["cancelled_cycles"] or plan["prior"] != current:
                raise PaperError("settlement plan/predecessor mismatch")
            if plan["predecessor_record_count"] != len(replay["records"]):
                raise PaperError("settlement predecessor record count mismatch")
            if any(e["booking_date"] > plan["trade_date"] for e in events):
                raise PaperError("future event used as execution cash")
            calendar = [date(d) for d in record["calendar"]]
            if calendar != sorted(set(calendar)) or plan["trade_date"] not in calendar:
                raise PaperError("settlement calendar does not cover selected date")
            index = calendar.index(plan["trade_date"])
            previous_date = calendar[index - 1] if index else None
            requested = {p["instrument"] for p in current["positions"]} | {p["instrument"] for p in plan["buys"]}
            if set(result["quotes"]) != requested:
                raise PaperError("settlement quotes do not cover the exact request")
            for instrument, quote in result["quotes"].items():
                if quote["previous_date"] != previous_date:
                    raise PaperError("settlement used a different previous trading session")
                if quote["open_evidence"]["calendar_position"] != index:
                    raise PaperError("execution word uses a different calendar position")
                opening, factor = verify_price_row(quote["open_evidence"], instrument, plan["trade_date"], "NEXT_OPEN")
                if (opening, factor) != (quote["open"], quote["factor"]):
                    raise PaperError("settlement open differs from observed cash price")
                if previous_date is not None:
                    if quote["previous_close_evidence"]["calendar_position"] != index - 1:
                        raise PaperError("previous close uses a different calendar position")
                    closing, factor = verify_price_row(quote["previous_close_evidence"], instrument, previous_date, "CASH_CLOSE")
                    if (closing, factor) != (quote["previous_close"], quote["previous_factor"]):
                        raise PaperError("settlement previous close/factor differs from observation")
            expected = settle_plan(plan, current, result["quotes"], result["rules"], events, buy_locks(replay))
            settled.add(cycle)
        elif record["kind"] == "EVENTS":
            after, external = apply_events(current, events)
            expected = {"prior_digest": state(current).digest, "after": after,
                        "events": events, "external_cashflow": external}
        else:
            raise PaperError("unknown paper record kind")
        if expected != result:
            raise PaperError("record accounting is inconsistent")
        consumed.update(event_ids)
        current = result["after"]
        total_external += decimal(result["external_cashflow"])
        replay["records"].append(record)
        states[len(replay["records"])] = (current, amount(total_external), has_external_funding(replay))
    pending = [p for key, p in journal["plans"].items() if key not in settled and key not in journal["cancelled_cycles"]]
    if len(pending) > 1 or any(p["prior"] != current or p["predecessor_record_count"] != len(journal["records"]) for p in pending):
        raise PaperError("pending plan has wrong predecessor")
    for plan in journal["plans"].values():
        predecessor = states.get(plan["predecessor_record_count"])
        if predecessor is None or predecessor[0] != plan["prior"]:
            raise PaperError("plan predecessor is not in this account")
    cache = {r["instrument"]: {"price": r["price"], "price_date": r["price_date"]}
             for r in initial["positions"]}
    last_date = selected
    for item in journal["valuations"]:
        result = item["result"] if "result" in item else item
        # The journal stores observed close inputs beside each derived valuation.
        if type(item["record_count"]) is not int:
            raise PaperError("valuation record count must be an integer")
        prior_row = states.get(item["record_count"])
        if prior_row is None or result["date"] < last_date:
            raise PaperError("valuation state/date mismatch")
        if state(prior_row[0]).digest != result["state_digest"]:
            raise PaperError("valuation state digest mismatch")
        verify_closes(item["price_receipts"], result["date"], item["observed_closes"])
        expected = valuation(prior_row[0], result["date"], item["observed_closes"], cache,
                             bootstrap["initial_nav"], prior_row[1], prior_row[2])
        if result != expected:
            raise PaperError("valuation is inconsistent")
        for row in result["positions"]:
            if row["price"] is not None and not row["stale"]:
                cache[row["instrument"]] = {"price": row["price"], "price_date": row["price_date"]}
        last_date = result["date"]


class PaperStore:
    def __init__(self, ctx, account_id):
        self.ctx, self.account_id = ctx, token(account_id)
        self.root = ctx.data_path("paper")
        self.path = self.root / "account.json"
        self.lock_path = self.root / "account.lock"

    def _contained(self, path):
        try:
            path.resolve().relative_to(self.ctx.root.resolve())
        except ValueError as exc:
            raise PaperError("paper path escapes selected workspace") from exc
        return path

    @contextmanager
    def locked(self):
        self._contained(self.root)
        self._contained(self.lock_path)
        lease = TrainingExecutionLease(self.lock_path)
        lease.acquire(run_id="paper-" + self.account_id)
        try:
            yield self
        finally:
            lease.release()

    def read(self):
        self._contained(self.path)
        if not self.path.exists():
            return None
        try:
            document = strict_json(self.path.read_bytes())
            checksum = document.pop("digest")
            if checksum != fingerprint_value(document):
                raise PaperError("paper journal digest mismatch")
            validate_journal(document, self.account_id)
            return document
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperError("invalid paper journal: %s" % exc) from exc

    def write(self, journal):
        validate_journal(journal, self.account_id)
        document = dict(journal, digest=fingerprint_value(journal))
        payload = json.dumps(document, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        self._contained(self.path)
        atomic_write_json_bytes(self.path, payload)
        # Check the canonical path after replace, not an open pre-replace descriptor.
        if self.read() != journal:
            raise PaperError("paper commit post-state is uncertain; inspect before retry")
