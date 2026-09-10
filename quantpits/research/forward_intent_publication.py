"""Create-only first intent pair, with offline accounting inputs and local time evidence.

Digest order: logical eight members -> request body -> linked physical members
-> manifest -> completion. Reading never confers fresh planning/time authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Any

from dateutil.tz import gettz
from quantpits.evidence.contracts import canonical_json_bytes as canonical
from quantpits.research import forward_intent_preparation as c3
from quantpits.research import historical_cycle as prices_module
from quantpits.research import intents as intent_module
from quantpits.research.accounting import ShadowPortfolioState, ShadowIntentBatch, ExecutionAssumption
from quantpits.research.forward_definitions import compile_shadow_forward_definitions

MEMBERS = ("request.json", "definitions.json", "priors.json", "champion_ranking.csv",
           "challenger_ranking.csv", "plans.json", "anchor_prices.json",
           "calendar_day.txt", "calendar_future.txt")
MEMBER_LIMIT = 4 * 1024 * 1024
TOTAL_LIMIT = 32 * 1024 * 1024
RECORD_LIMIT = 1024 * 1024
TIME_POLICY = "LOCAL_PREOPEN_OBSERVATION_V1"
OPENING_POLICY = "MATCHED_BOOTSTRAP_UNTRADED_V1"
_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class _Invalid(Exception):
    def __init__(self, code):
        self.code = code


def _need(value, code="BUNDLE_INVALID"):
    if not value:
        raise _Invalid(code)


def _raw(data):
    return hashlib.sha256(data).hexdigest()


def _hash(value):
    return _raw(canonical(value))


def _keys(value, names):
    _need(type(value) is dict and set(value) == set(names))
    return value


def _digest(value):
    _need(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "DIGEST_INVALID")
    return value


def _json(data):
    value = c3.surface._strict_json(data, "bundle")
    _need(canonical(value) == data)
    numeric_fields = {"schema_version", "size", "size_bytes", "calendar_position", "budget_seconds",
                      "clock_tolerance_seconds", "quantity", "topk", "n_drop", "production_buy_lot_size",
                      "target_buy_count", "buy_intent_shortage", "order_count", "buy_count", "sell_count",
                      "shortage_count", "pending_count", "instrument_count", "eligible", "scored", "missing",
                      "unscored", "requested", "raw_sources", "observed", "invalid"}
    def check_types(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key in numeric_fields:
                    _need(type(child) is not bool)
                if key == "schema_version":
                    _need(type(child) is int)
                check_types(child)
        elif isinstance(item, list):
            for child in item:
                check_types(child)
    check_types(value)
    return value


def _utc(value):
    _need(type(value) is str, "TIME_POLICY_INVALID")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _need(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0, "TIME_POLICY_INVALID")
    return parsed.astimezone(timezone.utc)


def _stamp(value):
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _policy(deadline, next_open, zone, opening, *, continuing=False):
    deadline, next_open = _utc(deadline), _utc(next_open)
    _need(type(zone) is str and gettz(zone) is not None and not zone.startswith("/"), "TIME_POLICY_INVALID")
    _need(deadline <= next_open and opening == ("VERIFIED_PREDECESSOR_AFTER_STATE_V1" if continuing else OPENING_POLICY), "TIME_POLICY_INVALID")
    return {"policy": TIME_POLICY, "decision_deadline_utc": _stamp(deadline),
            "next_open_utc": _stamp(next_open), "market_timezone": zone,
            "opening_policy": opening, "budget_seconds": 600, "clock_tolerance_seconds": 2}


def _clock():
    return datetime.now(timezone.utc), time.monotonic()


class _TimeGate:
    def __init__(self, policy):
        self.policy = policy
        self.first = None
        self.previous = None
        self.observations = []

    def check(self, event):
        wall, mono = _clock()
        _need(isinstance(wall, datetime) and wall.tzinfo is not None
              and wall.utcoffset().total_seconds() == 0 and math.isfinite(mono), "CLOCK_INVALID")
        if self.first is None:
            self.first = (wall, mono)
        if self.previous is not None:
            _need(wall >= self.previous[0] and mono >= self.previous[1], "CLOCK_INVALID")
        elapsed = mono - self.first[1]
        _need(abs((wall - self.first[0]).total_seconds() - elapsed) <= 2, "CLOCK_INVALID")
        _need(0 <= elapsed <= 600 and wall < _utc(self.policy["decision_deadline_utc"]), "DEADLINE_EXCEEDED")
        self.previous = (wall, mono)
        self.observations.append({"event": event, "at_utc": _stamp(wall), "elapsed_seconds": elapsed})
        return _stamp(wall)


class FirstIntentPublicationPlan:
    """Immutable safe observation, never a publication input."""
    __slots__ = ("_summary", "_inputs", "_metadata", "_report_data")

    def __init__(self, *args, **kwargs):
        raise TypeError("use publication APIs")

    def __setattr__(self, name, value):
        raise TypeError("observations are immutable")

    @property
    def status(self):
        return self.to_safe_summary_dict()["status"]

    @property
    def request_digest(self):
        return self.to_safe_summary_dict()["request_digest"]

    @property
    def d1_inputs(self):
        return self._inputs

    @property
    def d1_metadata(self):
        """Copies of verified persisted metadata; no publication authority."""
        return None if self._metadata is None else json.loads(self._metadata)

    @property
    def verified_report_data(self):
        """Defensive copy of same-read verified members, without authority."""
        return None if self._report_data is None else json.loads(self._report_data)

    def to_safe_summary_dict(self):
        return json.loads(self._summary)


class FirstIntentPublicationResult(FirstIntentPublicationPlan):
    __slots__ = ()


class FirstIntentBundleObservation(FirstIntentPublicationPlan):
    __slots__ = ()


class FirstIntentAccountingInputs(NamedTuple):
    role: str
    prior: Any
    intents: Any
    execution_assumption: Any


def _summary(status="PRECONDITION_BLOCKED", **changes):
    value = dict(schema_version=1, status=status, reason_codes=[], request_digest=None,
                 manifest_digest=None, current_cycle_id=None, trade_date=None,
                 did_write=False, write_state="NOT_WRITTEN", bundle_verified=False,
                 completion_record_verified=False, recorded_time_claim=None,
                 d1_readable=False, prospective_claim=False, epoch_started=False,
                 promotion_capability=False, time_observations=[], operation_id=None,
                 target_binding_digest=None, order_counts=None)
    value.update(changes)
    return value


def _result(cls, summary, inputs=None, metadata=None, report_data=None):
    value = object.__new__(cls)
    object.__setattr__(value, "_summary", canonical(summary))
    object.__setattr__(value, "_inputs", inputs)
    object.__setattr__(value, "_metadata", None if metadata is None else canonical(metadata))
    object.__setattr__(value, "_report_data", None if report_data is None else canonical(
        {name: raw.decode("utf-8") for name, raw in report_data.items()}))
    return value


def _implementation():
    return {"domain": "FIRST_INTENT_PUBLICATION_IMPLEMENTATION_V1", "members": [
        {"path": "research/forward_intent_publication.py", "sha256": _raw(Path(__file__).read_bytes())},
        {"path": "scripts/publish_forward_intent.py", "sha256": _raw(
            (Path(__file__).parents[1] / "scripts/publish_forward_intent.py").read_bytes())}]}


def _inventory(members):
    return [{"name": name, "size": len(data), "sha256": _raw(data), "schema_version": 1}
            for name, data in members.items()]


def _budget(members):
    _need(tuple(members) == MEMBERS and all(len(v) <= MEMBER_LIMIT for v in members.values())
          and sum(map(len, members.values())) <= TOTAL_LIMIT, "BUNDLE_BUDGET_EXCEEDED")


def _build(prepared, epoch, policy, *, continuing=False):
    _need(prepared.intent_pair_prepared, "PREPARATION_REQUIRED")
    pair, summary = prepared.pair, prepared.to_safe_summary_dict()
    definitions = {"schema_version": 1, "definitions": pair.definitions._current_raw()}
    members = {"definitions.json": canonical(definitions),
               "priors.json": canonical({"schema_version": 1, "opening_policy": policy["opening_policy"],
                   "bootstrap_manifest": json.loads(pair.bootstrap_bytes),
                   "roles": [{"role": role, "state": state.to_dict()} for role, state in zip(c3.ROLES, pair.priors)]}),
               "champion_ranking.csv": pair.rankings[0].to_csv_bytes(),
               "challenger_ranking.csv": pair.rankings[1].to_csv_bytes(),
               "plans.json": canonical({"schema_version": 1, "roles": [
                   {"role": role, "plan": plan.to_dict()} for role, plan in zip(c3.ROLES, pair.plans)]}),
               "anchor_prices.json": canonical({"schema_version": 1, "receipt": pair.price_receipt.to_dict()}),
               "calendar_day.txt": pair.calendar_bytes[0], "calendar_future.txt": pair.calendar_bytes[1]}
    body = {"schema_version": 1, "domain": "FIRST_FORWARD_INTENT_PAIR_REQUEST_V1", "epoch_id": epoch,
            "current_cycle_id": summary["current_cycle_id"], "trade_date": summary["trade_date"],
            "time_policy": policy, "implementation": _implementation(),
            "input_provenance": json.loads(pair.input_provenance), "input_digest": summary["input_digest"],
            "preparation_digest": summary["preparation_digest"], "roles": summary["roles"],
            "semantic_members": _inventory(members)}
    if continuing:
        body.update(schema_version=2, domain="CONTINUING_FORWARD_INTENT_PAIR_REQUEST_V1",
                    continuation=pair.continuation)
        _need(pair.continuation is not None, "PREDECESSOR_REQUIRED")
        body["implementation"]["domain"] = "CONTINUING_INTENT_PUBLICATION_IMPLEMENTATION_V1"
        body["implementation"]["members"].extend(_continuing_implementation())
    digest = _hash(body)
    definitions["request_digest"] = digest
    members["definitions.json"] = canonical(definitions)
    members["request.json"] = canonical({"body": body, "request_digest": digest})
    members = {name: members[name] for name in MEMBERS}
    _budget(members)
    manifest = {"schema_version": 1, "domain": "FIRST_FORWARD_INTENT_PAIR_BUNDLE_V1",
                "epoch_id": epoch, "current_cycle_id": body["current_cycle_id"],
                "request_digest": digest, "members": _inventory(members)}
    if continuing:
        manifest.update(schema_version=2, domain="CONTINUING_FORWARD_INTENT_PAIR_BUNDLE_V1")
    _validate(members, manifest, epoch, digest, continuing=continuing)
    return members, canonical(manifest), digest


def _validate_words(receipt, day, anchor, *, kind="CASH_CLOSE"):
    _need(kind in ("CASH_CLOSE", "NEXT_OPEN"))
    _keys(receipt, ("observation_kind", "observation_date", "calendar_position", "requested_instruments",
                    "rows", "observed_instruments", "missing_instruments", "invalid_instruments", "counts", "digest"))
    _need(receipt["digest"] == prices_module._digest_payload({k: v for k, v in receipt.items() if k != "digest"}))
    position = day.index(anchor)
    _need(receipt["observation_kind"] == kind and receipt["observation_date"] == anchor
          and type(receipt["calendar_position"]) is int and receipt["calendar_position"] == position)
    requested = receipt["requested_instruments"]
    _need(requested and requested == sorted(set(requested))
          and [r["instrument"] for r in receipt["rows"]] == requested)
    for row in receipt["rows"]:
        _keys(row, ("instrument", "observation_date", "calendar_position", "derived_field", "status", "reason_code",
                    "numerator", "denominator", "derivation_rule", "cash_price"))
        _need(row["observation_date"] == anchor and row["calendar_position"] == position
              and row["derived_field"] == kind and row["derivation_rule"] == prices_module.DERIVATION_RULE)
        for key, field in (("numerator", "close" if kind == "CASH_CLOSE" else "open"), ("denominator", "factor")):
            word = row[key]
            _keys(word, ("field", "logical_path", "status", "reason_code", "raw_file_digest", "feature_float32_bits", "feature_text"))
            _need(word["field"] == field and word["logical_path"] == "features/%s/%s.day.bin" % (row["instrument"].lower(), field))
            status, reason, bits, txt = (word[k] for k in ("status", "reason_code", "feature_float32_bits", "feature_text"))
            allowed = {"OBSERVED": ("OBSERVED",), "MISSING": ("SOURCE_FILE_MISSING", "POSITION_BEFORE_START", "MISSING_TAIL"),
                       "INVALID": ("MALFORMED_BIN", "INVALID_START_POSITION", "NONFINITE_WORD")}
            _need(status in allowed and reason in allowed[status])
            if reason == "SOURCE_FILE_MISSING":
                _need(word["raw_file_digest"] is None)
            else:
                ref = word["raw_file_digest"]
                _keys(ref, ("algorithm", "domain", "value", "size_bytes"))
                _need(type(ref["size_bytes"]) is int and ref["size_bytes"] >= 0)
                _need(ref["algorithm"] == "sha256" and ref["domain"] == "raw_bytes")
                _digest(ref["value"])
            if reason in ("OBSERVED", "INVALID_START_POSITION", "NONFINITE_WORD"):
                _need(type(bits) is str and len(bits) == 8 and bytes.fromhex(bits).hex() == bits)
                value = struct.unpack("<f", bytes.fromhex(bits))[0]
                _need(txt == format(value, ".17g"))
                if reason == "OBSERVED":
                    _need(math.isfinite(value))
                elif reason == "NONFINITE_WORD":
                    _need(not math.isfinite(value))
                else:
                    _need(not math.isfinite(value) or int(value) != value)
            else:
                _need(bits is None and txt is None)
        num, den = row["numerator"], row["denominator"]
        cash = None
        if "MISSING" in (num["status"], den["status"]):
            status, reason = "MISSING", "NUMERATOR_OR_FACTOR_MISSING"
        elif (num["status"], den["status"]) != ("OBSERVED", "OBSERVED"):
            status, reason = "INVALID", "NUMERATOR_OR_FACTOR_INVALID"
        else:
            factor, number = float(den["feature_text"]), float(num["feature_text"])
            derived = number / factor if factor > 0 else float("nan")
            if factor <= 0 or not math.isfinite(derived) or derived <= 0:
                status, reason = "INVALID", "INVALID_FACTOR_OR_DERIVED_PRICE"
            else:
                status, reason, cash = "OBSERVED", "OBSERVED", format(derived, ".17g")
        _need((row["status"], row["reason_code"], row["cash_price"]) == (status, reason, cash))
    counts = {"requested": len(requested), "raw_sources": 2 * len(requested)}
    for status in ("OBSERVED", "MISSING", "INVALID"):
        names = [r["instrument"] for r in receipt["rows"] if r["status"] == status]
        _need(receipt[status.lower() + "_instruments"] == names)
        counts[status.lower()] = len(names)
    _need(receipt["counts"] == counts)


def _validate_plan(report, prior, ranking, snapshot, definition, anchor, trade, market):
    fields = ("schema_version evidence_class prospective_claim promotion_capability warnings request definition definition_digest "
              "production_primitive_id ranking_digest ranking_counts universe prior_state_digest prices price_partitions "
              "ranking_partitions holding_classifications production_sell_proposal production_buy_suggestions "
              "raw_buy_candidate_instruments target_buy_count selected_buy_instruments excluded_buy_instruments "
              "buy_intent_shortage intents parity_checked selection_checked status result_digest").split()
    _keys(report, fields)
    _need(report["result_digest"] == intent_module._digest({k: v for k, v in report.items() if k != "result_digest"}))
    _need(report["schema_version"] == 1 and report["status"] == "COMPLETE"
          and report["parity_checked"] is True and report["selection_checked"] is True
          and report["prospective_claim"] is False and report["promotion_capability"] is False
          and report["evidence_class"] == "RETROSPECTIVE_TECHNICAL_REPLAY"
          and report["warnings"] == list(intent_module._WARNINGS))
    _need(report["request"] == dict(portfolio_id=prior.portfolio_id, cycle_id=anchor,
                                    anchor_date=anchor, trade_date=trade, market=market))
    _need(report["prior_state_digest"] == prior.digest and report["ranking_digest"] == _raw(ranking.to_csv_bytes())
          and report["prices"] == snapshot.to_dict() and report["definition"] == definition.to_dict()
          and report["definition_digest"] == definition.digest
          and report["production_primitive_id"] == definition.production_primitive_id)
    _need(report["ranking_counts"] == dict(eligible=ranking.eligible_count, scored=ranking.scored_count, unscored=ranking.missing_count))
    universe = intent_module.UniverseSnapshot.observe(market, anchor, tuple(r["instrument"] for r in ranking.rows))
    _need(report["universe"] == dict(market=market, anchor_date=anchor, instruments=list(universe.instruments),
          instrument_count=universe.instrument_count, fingerprint_algorithm=universe.fingerprint_algorithm,
          fingerprint=universe.fingerprint, membership_check_status="ENABLED" if definition.sell_out_of_universe else "DISABLED_BY_DEFINITION"))
    _need(report["price_partitions"] == dict(requested=list(snapshot.requested_instruments),
          observed=list(snapshot.observed_instruments), missing=list(snapshot.missing_instruments)))
    ids = [r["instrument"] for r in ranking.rows if r["scored"]]
    usable = set(snapshot.observed_instruments)
    _need(report["ranking_partitions"] == dict(scored_with_usable_price=[i for i in ids if i in usable],
          scored_with_missing_price=[i for i in ids if i not in usable], unscored_eligible=[r["instrument"] for r in ranking.rows if not r["scored"]]))
    holdings = {p.instrument: p.quantity for p in prior.positions}
    groups = report["holding_classifications"]
    _keys(groups, ("continuing", "normal_sell", "forced_executable", "forced_pending", "eligible_unscored_retained"))
    flat = [i for v in groups.values() for i in v]
    _need(len(flat) == len(set(flat)) and set(flat) == set(holdings))
    _need(all(i in usable for i in groups["normal_sell"] + groups["forced_executable"]))
    _need(all(i not in usable and i not in ids for i in groups["forced_pending"]))
    sells, buys = report["production_sell_proposal"], report["production_buy_suggestions"]
    for proposal in sells + buys:
        _keys(proposal, ("instrument", "datetime", "quantity", "estimated_amount", "score", "current_close"))
        raw_proposal = dict(proposal)
        raw_proposal["value"] = raw_proposal.pop("quantity")
        _need(intent_module._serialize_order(raw_proposal, "sealed proposal", trade) == proposal)
    _need([p["instrument"] for p in sells] == groups["forced_executable"] + groups["normal_sell"])
    _need(all(p["quantity"] == holdings[p["instrument"]] for p in sells))
    target = report["target_buy_count"]
    _need(type(target) is int and target >= 0)
    candidates = report["raw_buy_candidate_instruments"]
    _need(len(candidates) == len(set(candidates)) and all(i in usable and i in ids and i not in holdings for i in candidates))
    buy_ids = [p["instrument"] for p in buys]
    _need(len(set(buy_ids)) == len(buy_ids) and [i for i in candidates if i in buy_ids] == buy_ids)
    _need(report["selected_buy_instruments"] == buy_ids[:target]
          and report["excluded_buy_instruments"] == buy_ids[target:]
          and report["buy_intent_shortage"] == max(target - len(buys[:target]), 0))
    raw = report["intents"]
    _keys(raw, ("portfolio_id", "cycle_id", "trade_date", "intents", "digest"))
    rows = []
    for side, proposals in (("SELL", sells), ("BUY", buys[:target])):
        for ordinal, proposal in enumerate(proposals):
            instrument, quantity = proposal["instrument"], proposal["quantity"]
            rows.append(dict(intent_id=intent_module._intent_id(portfolio_id=prior.portfolio_id, cycle_id=anchor,
                trade_date=trade, definition_digest=definition.digest, ranking_digest=report["ranking_digest"],
                prior_digest=prior.digest, prices_digest=snapshot.digest, side=side, ordinal=ordinal,
                instrument=instrument, quantity=quantity), portfolio_id=prior.portfolio_id, cycle_id=anchor,
                trade_date=trade, side=side, instrument=instrument, quantity=quantity))
    batch = ShadowIntentBatch.from_iterable(portfolio_id=prior.portfolio_id, cycle_id=anchor, trade_date=trade, rows=rows)
    _need(batch.to_dict() == raw)
    return batch


def _continuing_implementation():
    return [{"path": name, "sha256": _raw((Path(__file__).parents[1] / name).read_bytes())}
            for name in ("research/forward_continuation.py", "scripts/continue_forward.py")]


def _validate(members, manifest, epoch, expected, *, continuing=False):
    _budget(members)
    _keys(manifest, ("schema_version", "domain", "epoch_id", "current_cycle_id", "request_digest", "members"))
    _need(manifest["schema_version"] == (2 if continuing else 1) and manifest["domain"] == ("CONTINUING_FORWARD_INTENT_PAIR_BUNDLE_V1" if continuing else "FIRST_FORWARD_INTENT_PAIR_BUNDLE_V1")
          and manifest["epoch_id"] == epoch and manifest["request_digest"] == expected
          and manifest["members"] == _inventory(members))
    request = _keys(_json(members["request.json"]), ("body", "request_digest"))
    body = _keys(request["body"], ("schema_version domain epoch_id current_cycle_id trade_date time_policy implementation "
                  "input_provenance input_digest preparation_digest roles semantic_members").split() + (["continuation"] if continuing else []))
    _need(request["request_digest"] == expected == _hash(body) and body["epoch_id"] == epoch
          and body["schema_version"] == (2 if continuing else 1) and body["domain"] == ("CONTINUING_FORWARD_INTENT_PAIR_REQUEST_V1" if continuing else "FIRST_FORWARD_INTENT_PAIR_REQUEST_V1"))
    implementation = _keys(body["implementation"], ("domain", "members"))
    _need(implementation["domain"] == ("CONTINUING_INTENT_PUBLICATION_IMPLEMENTATION_V1" if continuing else "FIRST_INTENT_PUBLICATION_IMPLEMENTATION_V1"))
    _need([r["path"] for r in implementation["members"]] == ["research/forward_intent_publication.py", "scripts/publish_forward_intent.py"] + (["research/forward_continuation.py", "scripts/continue_forward.py"] if continuing else []))
    for row in implementation["members"]:
        _keys(row, ("path", "sha256"))
        _digest(row["sha256"])
    logical = {name: data for name, data in members.items() if name != "request.json"}
    definitions = _keys(_json(logical["definitions.json"]), ("schema_version", "definitions", "request_digest"))
    _need(definitions["request_digest"] == expected and definitions["schema_version"] == 1)
    logical["definitions.json"] = canonical({k: v for k, v in definitions.items() if k != "request_digest"})
    _need(body["semantic_members"] == _inventory(logical))
    compiled = compile_shadow_forward_definitions(definitions["definitions"])
    provenance = body["input_provenance"]
    provenance_keys = ("selectors definition bootstrap cycle_seal signal model rankings calendar future_calendar price "
                       "implementation engine_commit engine_tree execution_assumption").split()
    if "model_copy_continuity" in provenance or "model_copy_observed_inputs" in provenance:
        provenance_keys += ["model_copy_continuity", "model_copy_observed_inputs"]
    _keys(provenance, provenance_keys)
    _need(body["input_digest"] == _hash(provenance)
          and body["preparation_digest"] == _hash({"input_digest": body["input_digest"], "roles": body["roles"]}))
    _need(provenance["definition"] == dict(compiled.request_digest)
          and provenance["execution_assumption"] == compiled.execution_assumption.to_dict())
    selectors = provenance["selectors"]
    _keys(selectors, ("definition_set_id", "definition_evidence_cycle_id", "bootstrap_set_id", "bootstrap_source_cycle_id",
                      "current_cycle_id", "signal_capsule_id", "model_capsule_id"))
    for value in selectors.values():
        c3.surface._identifier(value, "selector")
    anchor, trade = c3.surface._date(body["current_cycle_id"], "anchor"), c3.surface._date(body["trade_date"], "trade")
    _need(manifest["current_cycle_id"] == anchor == selectors["current_cycle_id"]
          and selectors["definition_set_id"] == compiled.definition_set_id)
    day, future = c3._calendar(members["calendar_day.txt"]), c3._calendar(members["calendar_future.txt"])
    _need(anchor in day and anchor in future and [d for d in day if d <= anchor] == [d for d in future if d <= anchor]
          and [d for d in future if d > anchor][0] == trade
          and provenance["calendar"] == _raw(members["calendar_day.txt"])
          and provenance["future_calendar"] == _raw(members["calendar_future.txt"]))
    policy = body["time_policy"]
    _need(policy == _policy(policy["decision_deadline_utc"], policy["next_open_utc"], policy["market_timezone"], policy["opening_policy"], continuing=continuing))
    _need(_utc(policy["next_open_utc"]).astimezone(gettz(policy["market_timezone"])).date().isoformat() == trade, "NEXT_OPEN_DATE_MISMATCH")
    prior_doc = _keys(_json(members["priors.json"]), ("schema_version", "opening_policy", "bootstrap_manifest", "roles"))
    plan_doc = _keys(_json(members["plans.json"]), ("schema_version", "roles"))
    price_doc = _keys(_json(members["anchor_prices.json"]), ("schema_version", "receipt"))
    _need(prior_doc["schema_version"] == plan_doc["schema_version"] == price_doc["schema_version"] == 1
          and prior_doc["opening_policy"] == policy["opening_policy"])
    bootstrap = prior_doc["bootstrap_manifest"]
    from quantpits.research import forward_bootstrap as bootstrap_module
    _keys(bootstrap, ("schema_version bootstrap_kind storage_claim bootstrap_set_id definition_set_id "
        "definition_evidence_cycle_id bootstrap_source_cycle_id definition_request_digest "
        "definition_evidence_request_digest definition_evidence_manifest_digest definition_evidence_operation_id "
        "phase37a_seal_digest phase37a_manifest_digest source_portfolio_raw_digest source_portfolio_semantic_digest "
        "source_state_interpretation economic_state_digest roles member_count members matched_economics_checked "
        "portfolio_bootstrap_claim intent_claim epoch_started prospective_claim promotion_capability").split())
    _need(bootstrap["schema_version"] == 1 and bootstrap["bootstrap_kind"] == bootstrap_module.BOOTSTRAP_KIND
          and bootstrap["storage_claim"] == bootstrap_module.STORAGE_CLAIM
          and bootstrap["bootstrap_set_id"] == selectors["bootstrap_set_id"]
          and bootstrap["definition_set_id"] == compiled.definition_set_id
          and bootstrap["definition_request_digest"] == dict(compiled.request_digest)
          and bootstrap["member_count"] == 3 and len(bootstrap["members"]) == 3
          and bootstrap["source_state_interpretation"] == "SEALED_PRODUCTION_PORTFOLIO_AT_BOOTSTRAP_SOURCE_CYCLE_V1"
          and bootstrap["matched_economics_checked"] is True and bootstrap["portfolio_bootstrap_claim"] is True
          and all(bootstrap[k] is False for k in ("intent_claim", "epoch_started", "prospective_claim", "promotion_capability")))
    _need(provenance["bootstrap"] == c3.surface._digest(bootstrap)
          and bootstrap["bootstrap_source_cycle_id"] == selectors["bootstrap_source_cycle_id"]
          and bootstrap["definition_evidence_cycle_id"] == selectors["definition_evidence_cycle_id"])
    receipt = price_doc["receipt"]
    _validate_words(receipt, day, anchor)
    _need(provenance["price"] == receipt["digest"])
    for rows in (prior_doc["roles"], plan_doc["roles"], body["roles"], bootstrap["roles"]):
        _need(type(rows) is list and [r["role"] for r in rows] == list(c3.ROLES))
    priors, batches, rankings = [], [], []
    execution = ExecutionAssumption.from_dict({k: compiled.execution_assumption.to_dict()[k] for k in ExecutionAssumption._FIELDS})
    definition = intent_module.CurrentRuleIntentDefinition.from_dict(compiled.protocol.to_dict()["intent_definition"])
    market = plan_doc["roles"][0]["plan"]["request"]["market"]
    for index, role in enumerate(c3.ROLES):
        state = _keys(prior_doc["roles"][index], ("role", "state"))["state"]
        prior = ShadowPortfolioState.from_dict({k: v for k, v in state.items() if k != "digest"})
        if continuing:
            from quantpits.research.forward_continuation import _validate_prior
            _validate_prior(body["continuation"], index, role, prior, state, compiled, selectors, anchor)
            _need(prior.portfolio_id == bootstrap["roles"][index]["portfolio_id"], "PREDECESSOR_ROLE_INVALID")
        else:
            _need(prior.to_dict() == state and prior.as_of_date == selectors["bootstrap_source_cycle_id"] < anchor)
            strategy = (compiled.champion, compiled.challenger)[index].to_dict()
            _need(strategy["data_cutoff"] <= selectors["definition_evidence_cycle_id"] <= prior.as_of_date
                  and anchor >= strategy["effective_cycle"])
            portfolio_id = "portfolio." + _hash(dict(domain="MATCHED_FORWARD_PORTFOLIO_ID_V1", role=role,
                bootstrap_source_cycle_id=prior.as_of_date, source_portfolio_raw_digest=bootstrap["source_portfolio_raw_digest"],
                definition_set_id=compiled.definition_set_id, strategy_id=strategy["strategy_id"]))
            _need(prior.portfolio_id == portfolio_id and bootstrap["roles"][index] == dict(role=role,
                strategy_id=strategy["strategy_id"], portfolio_id=portfolio_id, state_digest=c3.surface._digest(state)))
            from quantpits.research.forward_bootstrap import _economic_payload
            _need(c3.surface._digest(_economic_payload(prior)) == bootstrap["economic_state_digest"])
        ranking_data = members[role.lower() + "_ranking.csv"]
        ranking = c3.replay._ranking_from_csv(ranking_data)
        _need(ranking.to_csv_bytes() == ranking_data and ranking.complete and ranking.scored_count > 0 and ranking.missing_count == 0)
        requested = sorted({r["instrument"] for r in ranking.rows} | {p.instrument for p in prior.positions})
        snapshot = intent_module.AnchorPriceSnapshot.from_iterable(anchor_date=anchor, requested_instruments=requested,
            rows=[dict(instrument=r["instrument"], anchor_date=anchor,
                       status="OBSERVED" if r["status"] == "OBSERVED" else "MISSING",
                       cash_close=r["cash_price"] if r["status"] == "OBSERVED" else None)
                  for r in receipt["rows"] if r["instrument"] in requested])
        report = _keys(plan_doc["roles"][index], ("role", "plan"))["plan"]
        batch = _validate_plan(report, prior, ranking, snapshot, definition, anchor, trade, market)
        expected_role = dict(role=role, status="COMPLETE", reason=None,
            coverage_counts=dict(eligible=ranking.eligible_count, scored=ranking.scored_count, missing=ranking.missing_count),
            order_count=len(batch.intents), buy_count=len(report["selected_buy_instruments"]),
            sell_count=len(report["production_sell_proposal"]), shortage_count=report["buy_intent_shortage"],
            pending_count=len(report["holding_classifications"]["forced_pending"]), ranking_digest=_raw(ranking_data),
            prior_digest=prior.digest, planning_result_digest=report["result_digest"], intent_batch_digest=batch.digest,
            price_digest=snapshot.digest)
        _need(body["roles"][index] == expected_role and provenance["rankings"][index] == _raw(ranking_data))
        priors.append(prior)
        batches.append(batch)
        rankings.append(ranking)
    _need(priors[0].portfolio_id != priors[1].portfolio_id
          and {r["instrument"] for r in rankings[0].rows} == {r["instrument"] for r in rankings[1].rows}
          and receipt["requested_instruments"] == sorted({r["instrument"] for r in rankings[0].rows}
              | {p.instrument for prior in priors for p in prior.positions}))
    if continuing:
        from quantpits.research.forward_continuation import _validate_continuation, _schedule
        _validate_continuation(body["continuation"], epoch, compiled, selectors)
        _need(body["continuation"]["schedule"]["market_timezone"] == policy["market_timezone"], "SCHEDULE_INVALID")
        _schedule(members["calendar_day.txt"], members["calendar_future.txt"], body["continuation"], anchor, trade)
    return body, tuple(FirstIntentAccountingInputs(role, prior, batch, execution)
                       for role, prior, batch in zip(c3.ROLES, priors, batches))


def _directory(path):
    c3.surface._physical_path(path, "target", directory=True)
    info = os.lstat(str(path))
    _need(stat.S_IMODE(info.st_mode) == 0o700, "TARGET_PERMISSIONS_INVALID")
    return info.st_dev, info.st_ino, info.st_mode


def _target(root, epoch):
    c3.surface._identifier(epoch, "epoch")
    root = c3.surface._physical_path(root, "intent_store_root", directory=True)
    return root, root / epoch, _directory(root)


def _binding(root, identity, epoch, *, store_bindings=None):
    value = {"root": str(root), "identity": list(identity), "epoch_id": epoch}
    if store_bindings is not None:
        value["store_bindings"] = store_bindings
    return _hash(value)


def _read_bundle(root, target, identity, epoch, expected, *, completion=True, cycle_index=None):
    _need(_directory(root) == identity, "TARGET_REPLACED")
    _need(os.path.lexists(str(target)), "INCOMPLETE")
    target_identity = _directory(target)
    names = set(MEMBERS) | {"manifest.json"}
    if completion:
        names.add("completion.json")
    actual = {p.name for p in target.iterdir()}
    _need(not actual - names, "CONFLICT")
    _need(actual == names, "INCOMPLETE")
    data = {}
    fingerprints = {}
    for name in MEMBERS + ("manifest.json",) + (("completion.json",) if completion else ()):
        data[name], fingerprints[name] = c3.surface._read_regular(target / name,
            maximum=MEMBER_LIMIT if name in MEMBERS else RECORD_LIMIT, private=True)
    members = {name: data[name] for name in MEMBERS}
    manifest = _json(data["manifest.json"])
    _need(manifest.get("request_digest") == expected, "CONFLICT")
    body, inputs = _validate(members, manifest, epoch, expected, continuing=cycle_index is not None)
    if cycle_index is not None:
        _need(body["continuation"]["cycle_index"] == cycle_index, "CYCLE_INDEX_INVALID")
    record = None
    if completion:
        record = _keys(_json(data["completion.json"]), ("schema_version", "event", "operation_id", "epoch_id",
            "current_cycle_id", "request_digest", "manifest_digest", "time_policy", "bundle_verified_at_utc",
            "preparation_started_at_utc", "target_binding_digest") + (("store_bindings",) if cycle_index is not None else ()))
        _need(record["schema_version"] == 1 and record["event"] == "DATA_BUNDLE_VERIFIED_BEFORE_COMPLETION_CREATION"
              and record["epoch_id"] == epoch and record["current_cycle_id"] == body["current_cycle_id"]
              and record["request_digest"] == expected and record["manifest_digest"] == _raw(data["manifest.json"])
              and record["time_policy"] == body["time_policy"]
              and record["target_binding_digest"] == _binding(root, identity, target.name,
                  store_bindings=record["store_bindings"] if cycle_index is not None else None))
        if cycle_index is not None:
            _keys(record["store_bindings"], ("intent", "settlement"))
            for value in record["store_bindings"].values():
                _digest(value)
            _need(record["store_bindings"]["intent"] == _binding(root, identity, "CONTINUING_INTENT_ROOT"), "STORE_BINDING_INVALID")
        _need(str(uuid.UUID(record["operation_id"])) == record["operation_id"])
        started, verified = _utc(record["preparation_started_at_utc"]), _utc(record["bundle_verified_at_utc"])
        _need(started <= verified < _utc(body["time_policy"]["decision_deadline_utc"])
              and (verified - started).total_seconds() <= 602)
    # Reopen every canonical member after schema checks; no old-fd success.
    for name, original in data.items():
        observed, fingerprint = c3.surface._read_regular(target / name,
            maximum=MEMBER_LIMIT if name in MEMBERS else RECORD_LIMIT, private=True)
        _need(observed == original and fingerprint == fingerprints[name], "TARGET_REPLACED")
    _need(_directory(root) == identity and _directory(target) == target_identity
          and {p.name for p in target.iterdir()} == names, "TARGET_REPLACED")
    summary = _summary("VERIFIED", request_digest=expected, manifest_digest=_raw(data["manifest.json"]),
        current_cycle_id=body["current_cycle_id"], trade_date=body["trade_date"],
        bundle_verified=True, completion_record_verified=completion, d1_readable=completion,
        target_binding_digest=_binding(root, identity, target.name,
            store_bindings=record["store_bindings"] if cycle_index is not None and record else None),
        order_counts=[len(i.intents.intents) for i in inputs])
    if cycle_index is not None:
        summary.update(cycle_index=cycle_index, cycle_intent_published=False,
                       predecessor_join_observed=False, whole_chain_verified=False, planning_roles=body["roles"])
    if record:
        summary.update(operation_id=record["operation_id"], recorded_time_claim={
            "policy": TIME_POLICY, "bundle_verified_at_utc": record["bundle_verified_at_utc"],
            "scope": "DATA_BUNDLE_ONLY", "original_success_record_required": True})
    return summary, inputs, data


def inspect_first_forward_intent_pair(intent_store_root, epoch_id, *, expected_request_digest):
    summary = _summary()
    try:
        _digest(expected_request_digest)
        root, target, identity = _target(intent_store_root, epoch_id)
        summary, inputs, data = _read_bundle(root, target, identity, epoch_id, expected_request_digest)
        metadata = {name: data[name].decode("utf-8") for name in
                    ("request.json", "manifest.json", "completion.json", "calendar_day.txt")}
        return _result(FirstIntentBundleObservation, summary, inputs, metadata, data)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        code = exc.code if isinstance(exc, (_Invalid, c3._Blocked)) else "BUNDLE_INVALID"
        status = code if code in ("INCOMPLETE", "CONFLICT") else "UNCERTAIN"
        if isinstance(exc, _Invalid) and code == "BUNDLE_INVALID":
            status = "CONFLICT"
        return _result(FirstIntentBundleObservation, _summary(status, reason_codes=[code]))


def _write_member(descriptor, name, data):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                 0o600, dir_fd=descriptor)
    try:
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            _need(count > 0, "WRITE_FAILED")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_bundle(root, target, identity, epoch, members, manifest, digest, gate, owner, *, cycle_index=None, store_bindings=None):
    summary = _summary(request_digest=digest, target_binding_digest=_binding(root, identity, target.name, store_bindings=store_bindings))
    if cycle_index is not None:
        summary.update(cycle_index=cycle_index, cycle_intent_published=False,
                       predecessor_join_observed=False, whole_chain_verified=False, planning_roles=None)
    root_fd = target_fd = None
    attempted = False
    try:
        gate.check("BEFORE_WRITE")
        owner.check()
        _need(_directory(root) == identity, "TARGET_REPLACED")
        root_fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        info = os.fstat(root_fd)
        _need((info.st_dev, info.st_ino, info.st_mode) == identity, "TARGET_REPLACED")
        try:
            attempted = True
            os.mkdir(target.name, mode=0o700, dir_fd=root_fd)
            summary.update(did_write=True, write_state="TARGET_CREATED")
        except FileExistsError:
            attempted = False
            if cycle_index is None:
                observed = inspect_first_forward_intent_pair(root, epoch, expected_request_digest=digest)
            else:
                from quantpits.research.forward_continuation import inspect_next_forward_intent_pair
                observed = inspect_next_forward_intent_pair(root.parent, epoch, cycle_index,
                                                           expected_request_digest=digest)
            summary = observed.to_safe_summary_dict()
            if observed.status == "VERIFIED":
                summary["status"] = "ADOPTED"
            return summary, observed.d1_inputs
        target_fd = os.open(target.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
        info = os.fstat(target_fd)
        target_identity = (info.st_dev, info.st_ino, info.st_mode)
        _need(_directory(target) == target_identity, "TARGET_REPLACED")
        for name, data in list(members.items()) + [("manifest.json", manifest)]:
            _need(_directory(root) == identity and _directory(target) == target_identity, "TARGET_REPLACED")
            _write_member(target_fd, name, data)
        os.fsync(target_fd)
        os.fsync(root_fd)
        _, _, stored = _read_bundle(root, target, identity, epoch, digest, completion=False, cycle_index=cycle_index)
        _need(all(stored[name] == data for name, data in members.items()) and stored["manifest.json"] == manifest)
        owner.check()
        verified = gate.check("DATA_BUNDLE_VERIFIED")
        operation = str(uuid.uuid4())
        summary.update(operation_id=operation, manifest_digest=_raw(manifest))
        record_doc = dict(schema_version=1, event="DATA_BUNDLE_VERIFIED_BEFORE_COMPLETION_CREATION",
            operation_id=operation, epoch_id=epoch, current_cycle_id=_json(members["request.json"])["body"]["current_cycle_id"],
            request_digest=digest, manifest_digest=_raw(manifest), time_policy=gate.policy,
            bundle_verified_at_utc=verified, preparation_started_at_utc=gate.observations[0]["at_utc"],
            target_binding_digest=_binding(root, identity, target.name, store_bindings=store_bindings))
        if cycle_index is not None:
            record_doc["store_bindings"] = store_bindings
        record = canonical(record_doc)
        _need(len(record) <= RECORD_LIMIT)
        _write_member(target_fd, "completion.json", record)
        os.fsync(target_fd)
        os.fsync(root_fd)
        observed, inputs, final_data = _read_bundle(root, target, identity, epoch, digest, cycle_index=cycle_index)
        _need(final_data["completion.json"] == record and _directory(target) == target_identity, "TARGET_REPLACED")
        owner.check()
        owner.close()
        gate.check("FINAL_PUBLICATION_VERIFIED")
        summary = observed
        observed.update(status="COMMITTED", did_write=True, write_state="COMMITTED",
                        prospective_claim=True, epoch_started=cycle_index is None, time_observations=gate.observations)
        if cycle_index is not None:
            observed.update(cycle_intent_published=True, predecessor_join_observed=True)
        return observed, inputs
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        code = exc.code if isinstance(exc, (_Invalid, c3._Blocked)) else "PUBLICATION_FAILED"
        if summary["did_write"] or attempted:
            summary.update(status="UNCERTAIN", did_write=True if summary["did_write"] else None,
                           write_state="WRITTEN_UNVERIFIED" if summary["did_write"] else "UNKNOWN")
        summary.update(reason_codes=[code], time_observations=gate.observations)
        return summary, None
    finally:
        active = sys.exc_info()[1]
        for fd in (target_fd, root_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except _PROCESS_CONTROL:
                    if not isinstance(active, _PROCESS_CONTROL):
                        raise
                except Exception:
                    summary.update(status="UNCERTAIN", reason_codes=["DESCRIPTOR_CLOSE_FAILED"],
                                   prospective_claim=False, epoch_started=False)
                    if cycle_index is not None:
                        summary.update(cycle_intent_published=False, predecessor_join_observed=False)


def _run(args, root_value, epoch, deadline, next_open, zone, opening, expected=None):
    cls = FirstIntentPublicationPlan if expected is None else FirstIntentPublicationResult
    owner = c3._PreparationGuards()
    summary = _summary()
    inputs = None
    try:
        root, target, identity = _target(root_value, epoch)
        if expected is not None:
            _digest(expected)
            if os.path.lexists(str(target)):
                observed = inspect_first_forward_intent_pair(root, epoch, expected_request_digest=expected)
                summary, inputs = observed.to_safe_summary_dict(), observed.d1_inputs
                if observed.status == "VERIFIED":
                    summary["status"] = "ADOPTED"
                return _result(cls, summary, inputs)
        policy = _policy(deadline, next_open, zone, opening)
        gate = _TimeGate(policy)
        gate.check("BEFORE_PREPARATION")
        prepared = c3._prepare_first_forward_intent(*args, _guard_owner=owner)
        if not prepared.intent_pair_prepared:
            summary.update(status=prepared.status, reason_codes=prepared.to_safe_summary_dict()["reason_codes"])
        else:
            members, manifest, digest = _build(prepared, epoch, policy)
            owner.check()
            gate.check("PREPARATION_VERIFIED")
            summary.update(status="READY", request_digest=digest, current_cycle_id=args[4],
                trade_date=prepared.to_safe_summary_dict()["trade_date"],
                target_binding_digest=_binding(root, identity, epoch), time_observations=gate.observations,
                order_counts=[len(p.intents.intents) for p in prepared.pair.plans])
            if expected is not None:
                if expected != digest:
                    summary.update(status="REQUEST_MISMATCH", reason_codes=["REQUEST_MISMATCH"])
                else:
                    summary, inputs = _publish_bundle(root, target, identity, epoch, members, manifest, digest, gate, owner)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        summary.update(status="UNCERTAIN" if summary["did_write"] else "PRECONDITION_BLOCKED",
                       reason_codes=[exc.code if isinstance(exc, (_Invalid, c3._Blocked)) else "INPUT_INVALID"])
    finally:
        active = sys.exc_info()[1]
        try:
            owner.close()
        except _PROCESS_CONTROL:
            if not isinstance(active, _PROCESS_CONTROL):
                raise
        except Exception:
            summary.update(status="UNCERTAIN" if summary["did_write"] else "PRECONDITION_BLOCKED",
                           reason_codes=["INPUT_OBSERVER_CLOSE_FAILED"], prospective_claim=False, epoch_started=False)
    return _result(cls, summary, inputs)


def prepare_first_forward_intent_publication(
    production_workspace_root, research_workspace_root, engine_root,
    qlib_provider_root, current_cycle_id, activation_path,
    definition_store_root, evidence_store_root, bootstrap_store_root, bootstrap_set_id,
    signal_capsule_store_root, signal_capsule_id, model_capsule_store_root, model_capsule_id,
    *, intent_store_root, epoch_id, decision_deadline_utc, next_open_utc, market_timezone, opening_policy,
) -> FirstIntentPublicationPlan:
    return _run((production_workspace_root, research_workspace_root, engine_root, qlib_provider_root,
        current_cycle_id, activation_path, definition_store_root, evidence_store_root, bootstrap_store_root,
        bootstrap_set_id, signal_capsule_store_root, signal_capsule_id, model_capsule_store_root, model_capsule_id),
        intent_store_root, epoch_id, decision_deadline_utc, next_open_utc, market_timezone, opening_policy)


def publish_first_forward_intent_pair(
    production_workspace_root, research_workspace_root, engine_root,
    qlib_provider_root, current_cycle_id, activation_path,
    definition_store_root, evidence_store_root, bootstrap_store_root, bootstrap_set_id,
    signal_capsule_store_root, signal_capsule_id, model_capsule_store_root, model_capsule_id,
    *, intent_store_root, epoch_id, decision_deadline_utc, next_open_utc, market_timezone, opening_policy,
    expected_request_digest,
) -> FirstIntentPublicationResult:
    if expected_request_digest is None:
        return _result(FirstIntentPublicationResult, _summary(reason_codes=["DIGEST_INVALID"]))
    return _run((production_workspace_root, research_workspace_root, engine_root, qlib_provider_root,
        current_cycle_id, activation_path, definition_store_root, evidence_store_root, bootstrap_store_root,
        bootstrap_set_id, signal_capsule_store_root, signal_capsule_id, model_capsule_store_root, model_capsule_id),
        intent_store_root, epoch_id, decision_deadline_utc, next_open_utc, market_timezone, opening_policy,
        expected_request_digest)
