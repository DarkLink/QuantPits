"""First-pair settlement: observed inputs, B0 accounting, create-only offline evidence.

Source metadata records the publisher's local observations, not trusted timestamps.
No persisted readiness flag is accepted as execution authority.
"""
from __future__ import annotations

import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Any

from quantpits.research import forward_intent_publication as c4
from quantpits.research import historical_cycle as prices
from quantpits.research.accounting import (
    ShadowPortfolioState, ShadowIntentBatch, ExecutionAssumption,
    ShadowQuoteSnapshot, ShadowPortfolioTransition,
)

canonical = c4.canonical
_raw, _hash, _keys, _need, _digest = c4._raw, c4._hash, c4._keys, c4._need, c4._digest
ROLES = ("CHAMPION", "CHALLENGER")
ARRIVAL_RULE = "OBSERVED_NEXT_OPEN_IN_REQUESTED_UNION_V1"
MEMBERS = ("request.json", "source_intent.json", "publication_success.json",
           "calendar_day.txt", "next_open_prices.json", "settlements.json")
MEMBER_LIMIT, TOTAL_LIMIT, RECORD_LIMIT = c4.MEMBER_LIMIT, c4.TOTAL_LIMIT, c4.RECORD_LIMIT
_PROCESS_CONTROL = c4._PROCESS_CONTROL


class SettlementAfterState(NamedTuple):
    role: str
    after_state: Any
    prior_digest: str
    intent_batch_digest: str
    source_request_digest: str
    source_manifest_digest: str
    source_operation_id: str
    settlement_request_digest: str


class FirstForwardSettlementObservation:
    __slots__ = ("_summary", "_states", "_metadata")

    def __init__(self, *args, **kwargs):
        raise TypeError("use settlement APIs")

    def __setattr__(self, name, value):
        raise TypeError("observations are immutable")

    @property
    def status(self):
        return self.to_safe_summary_dict()["status"]

    @property
    def request_digest(self):
        return self.to_safe_summary_dict()["request_digest"]

    @property
    def after_states(self):
        return self._states

    @property
    def continuation_metadata(self):
        return None if self._metadata is None else json.loads(self._metadata)

    def to_safe_summary_dict(self):
        return json.loads(self._summary)


def _summary(status="PRECONDITION_BLOCKED", **changes):
    value = dict(schema_version=1, status=status, reason_codes=[], request_digest=None,
        manifest_digest=None, current_cycle_id=None, trade_date=None, operation_id=None,
        target_binding_digest=None, did_write=False, write_state="NOT_WRITTEN",
        bundle_verified=False, completion_record_verified=False,
        accounting_pair_complete=False, state_chain_ready=False,
        source_forward_record_linked=False, prospective_claim=False, epoch_started=False,
        promotion_capability=False, roles=[dict(role=r, status="NOT_RUN", reason_codes=[]) for r in ROLES])
    value.update(changes)
    return value


def _result(summary, states=None, metadata=None):
    value = object.__new__(FirstForwardSettlementObservation)
    object.__setattr__(value, "_summary", canonical(summary))
    object.__setattr__(value, "_states", states if summary["state_chain_ready"] else None)
    object.__setattr__(value, "_metadata", None if metadata is None else canonical(metadata))
    return value


def _json(data):
    value = c4._json(data)
    def schemas(item):
        if isinstance(item, dict):
            if "schema_version" in item:
                _need(type(item["schema_version"]) is int)
            for child in item.values():
                schemas(child)
        elif isinstance(item, list):
            for child in item:
                schemas(child)
    schemas(value)
    return value


def _same(left, right):
    return canonical(left) == canonical(right)


def _clock():
    return datetime.now(timezone.utc)


IMPLEMENTATION_PATHS = ("research/forward_settlement.py", "scripts/settle_forward_intent.py",
                        "research/accounting.py", "research/historical_cycle.py",
                        "research/forward_intent_publication.py")


def _implementation(*, continuing=False):
    return dict(domain="CONTINUING_FORWARD_SETTLEMENT_IMPLEMENTATION_V1" if continuing else "FIRST_FORWARD_SETTLEMENT_IMPLEMENTATION_V1", members=[
        dict(path=name, sha256=_raw((Path(__file__).parents[1] / name).read_bytes()))
        for name in IMPLEMENTATION_PATHS + (("research/forward_continuation.py", "scripts/continue_forward.py") if continuing else ())])


def _budget(members):
    _need(tuple(members) == MEMBERS and all(len(v) <= MEMBER_LIMIT for v in members.values())
          and sum(map(len, members.values())) <= TOTAL_LIMIT, "BUNDLE_BUDGET_EXCEEDED")


def _source(source, success_bytes, epoch, *, continuing=False):
    """Check saved references and canonical inputs; never re-adopt external sources."""
    _keys(source, ("schema_version", "metadata", "roles"))
    _need(source["schema_version"] == (2 if continuing else 1))
    from quantpits.research.forward_continuation import SOURCE_METADATA
    meta = _keys(source["metadata"], SOURCE_METADATA if continuing else ("request.json", "manifest.json", "completion.json", "calendar_day.txt"))
    request, manifest, completion = (_json(meta[n].encode("utf-8")) for n in
                                   ("request.json", "manifest.json", "completion.json"))
    _keys(request, ("body", "request_digest"))
    body = _keys(request["body"], ("schema_version domain epoch_id current_cycle_id trade_date time_policy implementation "
        "input_provenance input_digest preparation_digest roles semantic_members").split() + (["continuation"] if continuing else []))
    expected = _digest(request["request_digest"])
    _need(_hash(body) == expected and body["epoch_id"] == epoch
          and body["domain"] == ("CONTINUING_FORWARD_INTENT_PAIR_REQUEST_V1" if continuing else "FIRST_FORWARD_INTENT_PAIR_REQUEST_V1") and body["schema_version"] == (2 if continuing else 1))
    _keys(manifest, ("schema_version", "domain", "epoch_id", "current_cycle_id", "request_digest", "members"))
    anchor, trade = body["current_cycle_id"], body["trade_date"]
    c4.c3.surface._date(anchor, "anchor")
    c4.c3.surface._date(trade, "trade")
    _need(anchor < trade and manifest["schema_version"] == (2 if continuing else 1)
          and manifest["domain"] == ("CONTINUING_FORWARD_INTENT_PAIR_BUNDLE_V1" if continuing else "FIRST_FORWARD_INTENT_PAIR_BUNDLE_V1")
          and manifest["epoch_id"] == epoch and manifest["current_cycle_id"] == anchor
          and manifest["request_digest"] == expected)
    _need([r["name"] for r in manifest["members"]] == list(c4.MEMBERS))
    for row in manifest["members"]:
        _keys(row, ("name", "size", "sha256", "schema_version"))
        _digest(row["sha256"])
        _need(type(row["size"]) is int and 0 <= row["size"] <= MEMBER_LIMIT and row["schema_version"] == 1)
        if row["name"] in meta:
            data = meta[row["name"]].encode("utf-8")
            _need(row["sha256"] == _raw(data) and row["size"] == len(data))
    policy = body["time_policy"]
    _need(policy == c4._policy(policy["decision_deadline_utc"], policy["next_open_utc"],
                              policy["market_timezone"], policy["opening_policy"], continuing=continuing))
    _need(c4._utc(policy["next_open_utc"]).astimezone(c4.gettz(policy["market_timezone"])).date().isoformat() == trade)
    _keys(completion, ("schema_version event operation_id epoch_id current_cycle_id request_digest manifest_digest "
          "time_policy bundle_verified_at_utc preparation_started_at_utc target_binding_digest").split() + (["store_bindings"] if continuing else []))
    manifest_digest = _raw(meta["manifest.json"].encode("utf-8"))
    _need(completion["schema_version"] == 1
          and completion["event"] == "DATA_BUNDLE_VERIFIED_BEFORE_COMPLETION_CREATION"
          and completion["epoch_id"] == epoch and completion["current_cycle_id"] == anchor
          and completion["request_digest"] == expected and completion["manifest_digest"] == manifest_digest
          and completion["time_policy"] == policy)
    _need(str(uuid.UUID(completion["operation_id"])) == completion["operation_id"])
    _digest(completion["target_binding_digest"])
    if continuing:
        _keys(completion["store_bindings"], ("intent", "settlement"))
        for value in completion["store_bindings"].values():
            _digest(value)
    # Original safe record may be pretty printed; preserve exact bytes and reject duplicate keys.
    success = _json(canonical(prices._strict_json(success_bytes, "success")))
    _keys(success, list(c4._summary()) + (["cycle_index", "cycle_intent_published", "predecessor_join_observed", "whole_chain_verified", "planning_roles"] if continuing else []))
    _need(success["schema_version"] == 1 and success["status"] == "COMMITTED"
          and success["write_state"] == "COMMITTED" and success["reason_codes"] == []
          and all(success[k] is True for k in ("did_write", "bundle_verified", "completion_record_verified",
                                               "d1_readable", "prospective_claim"))
          and success["epoch_started"] is (not continuing)
          and success["promotion_capability"] is False, "ORIGINAL_SUCCESS_REQUIRED")
    if continuing:
        _need(success["cycle_index"] == body["continuation"]["cycle_index"]
              and type(success["cycle_index"]) is int
              and success["cycle_intent_published"] is True
              and success["predecessor_join_observed"] is True
              and success["whole_chain_verified"] is False
              and success["planning_roles"] == body["roles"], "ORIGINAL_SUCCESS_MISMATCH")
    for key, value in dict(request_digest=expected, manifest_digest=manifest_digest,
                           operation_id=completion["operation_id"], target_binding_digest=completion["target_binding_digest"],
                           current_cycle_id=anchor, trade_date=trade).items():
        _need(success[key] == value, "ORIGINAL_SUCCESS_MISMATCH")
    _need(success["recorded_time_claim"] == dict(policy=c4.TIME_POLICY,
        bundle_verified_at_utc=completion["bundle_verified_at_utc"], scope="DATA_BUNDLE_ONLY",
        original_success_record_required=True), "ORIGINAL_SUCCESS_MISMATCH")
    events = success["time_observations"]
    _need(type(events) is list and [r["event"] for r in events] == ["BEFORE_PREPARATION", "PREPARATION_VERIFIED",
          "BEFORE_WRITE", "DATA_BUNDLE_VERIFIED", "FINAL_PUBLICATION_VERIFIED"], "TIME_EVIDENCE_INVALID")
    first = previous = c4._utc(completion["preparation_started_at_utc"])
    elapsed_previous = 0
    for index, row in enumerate(events):
        _keys(row, ("event", "at_utc", "elapsed_seconds"))
        wall, elapsed = c4._utc(row["at_utc"]), row["elapsed_seconds"]
        _need(type(elapsed) in (int, float) and math.isfinite(elapsed)
              and elapsed_previous <= elapsed <= 600 and previous <= wall < c4._utc(policy["decision_deadline_utc"])
              and abs((wall - first).total_seconds() - elapsed) <= 2, "TIME_EVIDENCE_INVALID")
        if index == 0:
            _need(wall == first and elapsed == 0, "TIME_EVIDENCE_INVALID")
        previous, elapsed_previous = wall, elapsed
    _need(events[3]["at_utc"] == completion["bundle_verified_at_utc"], "TIME_EVIDENCE_INVALID")
    day_bytes = meta["calendar_day.txt"].encode("utf-8")
    day = c4.c3._calendar(day_bytes)
    provenance = body["input_provenance"]
    _need(anchor in day and provenance["calendar"] == _raw(day_bytes)
          and body["input_digest"] == _hash(provenance)
          and body["preparation_digest"] == _hash(dict(input_digest=body["input_digest"], roles=body["roles"])))
    _need(type(source["roles"]) is list and [r["role"] for r in source["roles"]] == list(ROLES)
          and [r["role"] for r in body["roles"]] == list(ROLES))
    inputs = []
    for row, ref in zip(source["roles"], body["roles"]):
        _keys(row, ("role", "prior", "intents", "execution_assumption"))
        prior = ShadowPortfolioState.from_dict({k: v for k, v in row["prior"].items() if k != "digest"})
        batch_doc = row["intents"]
        batch = ShadowIntentBatch.from_iterable(portfolio_id=batch_doc["portfolio_id"], cycle_id=batch_doc["cycle_id"],
            trade_date=batch_doc["trade_date"], rows=batch_doc["intents"])
        assumption = ExecutionAssumption.from_dict({k: v for k, v in row["execution_assumption"].items() if k != "digest"})
        _need(prior.to_dict() == row["prior"] and batch.to_dict() == batch_doc
              and assumption.to_dict() == row["execution_assumption"]
              and assumption == ExecutionAssumption.from_dict({k: provenance["execution_assumption"][k]
                                                               for k in ExecutionAssumption._FIELDS})
              and prior.digest == ref["prior_digest"] and batch.digest == ref["intent_batch_digest"]
              and len(batch.intents) == ref["order_count"] and ref["status"] == "COMPLETE"
              and batch.trade_date == trade and batch.cycle_id == anchor and prior.as_of_date <= anchor
              and prior.portfolio_id == batch.portfolio_id)
        inputs.append(c4.FirstIntentAccountingInputs(row["role"], prior, batch, assumption))
    _need(inputs[0].prior.portfolio_id != inputs[1].prior.portfolio_id
          and success["order_counts"] == [len(i.intents.intents) for i in inputs])
    if continuing:
        from quantpits.research.forward_continuation import _validate_continuation, _validate_prior, _schedule
        _need(len(success_bytes) <= RECORD_LIMIT, "BUNDLE_BUDGET_EXCEEDED")
        definitions = _keys(_json(meta["definitions.json"].encode("utf-8")),
                            ("schema_version", "definitions", "request_digest"))
        _need(definitions["schema_version"] == 1 and definitions["request_digest"] == expected)
        compiled = c4.compile_shadow_forward_definitions(definitions["definitions"])
        _need(provenance["definition"] == dict(compiled.request_digest)
              and provenance["execution_assumption"] == compiled.execution_assumption.to_dict())
        _validate_continuation(body["continuation"], epoch, compiled, provenance["selectors"])
        _need(body["continuation"]["schedule"]["market_timezone"] == policy["market_timezone"], "SCHEDULE_INVALID")
        for index, item in enumerate(inputs):
            _validate_prior(body["continuation"], index, item.role, item.prior, item.prior.to_dict(),
                            compiled, provenance["selectors"], anchor)
        future_bytes = meta["calendar_future.txt"].encode("utf-8")
        _need(provenance["future_calendar"] == _raw(future_bytes))
        _schedule(day_bytes, future_bytes, body["continuation"], anchor, trade)
    else:
        _need(all(i.prior.as_of_date < anchor for i in inputs))
    return body, completion, tuple(inputs)


def _requested(item):
    return sorted({p.instrument for p in item.prior.positions} | {i.instrument for i in item.intents.intents})


def _calendar(day_bytes, body, source):
    day = c4.c3._calendar(day_bytes)
    old = c4.c3._calendar(source["metadata"]["calendar_day.txt"].encode("utf-8"))
    anchor, trade = body["current_cycle_id"], body["trade_date"]
    _need([d for d in day if d <= anchor] == [d for d in old if d <= anchor], "CALENDAR_PREFIX_CHANGED")
    _need(trade in day, "WAITING_FOR_DATA")
    _need([d for d in day if d > anchor][0] == trade, "NEXT_SESSION_CHANGED")
    return day


def _quotes(item, rows):
    requested = _requested(item)
    return ShadowQuoteSnapshot.from_iterable(portfolio_id=item.prior.portfolio_id,
        cycle_id=item.intents.cycle_id, trade_date=item.intents.trade_date, requested_instruments=requested,
        rows=[dict(instrument=r["instrument"], trade_date=item.intents.trade_date, field="NEXT_OPEN",
                   status="OBSERVED" if r["status"] == "OBSERVED" else "MISSING",
                   price=r["cash_price"] if r["status"] == "OBSERVED" else None)
              for r in rows if r["instrument"] in requested])


def _prices(doc, day, trade, inputs):
    _keys(doc, ("schema_version", "arrival_rule", "receipt"))
    _need(doc["schema_version"] == 1 and doc["arrival_rule"] == ARRIVAL_RULE)
    union = sorted({i for item in inputs for i in _requested(item)})
    receipt = doc["receipt"]
    if not union:
        _need(receipt == dict(status="NOT_REQUIRED_EMPTY_EXPOSURE", trade_date=trade, requested_instruments=[], rows=[]))
    else:
        c4._validate_words(receipt, day, trade, kind="NEXT_OPEN")
        _need(receipt["requested_instruments"] == union)
        _need(all(type(count) is int for count in receipt["counts"].values()))
        for row in receipt["rows"]:
            _need(type(row["calendar_position"]) is int)
            for key in ("numerator", "denominator"):
                word = row[key]
                _need(word["reason_code"] not in ("MALFORMED_BIN", "INVALID_START_POSITION"),
                      "PRICE_SOURCE_SCHEMA_INVALID")
                if word["raw_file_digest"] is not None:
                    size = word["raw_file_digest"]["size_bytes"]
                    _need(size >= 8 and size % 4 == 0, "PRICE_SOURCE_SCHEMA_INVALID")
        _need(bool(receipt["observed_instruments"]), "WAITING_FOR_DATA")
    return receipt["rows"]


def _calculate(inputs, rows):
    transitions, diagnostics = [], []
    for item in inputs:
        diagnostic = dict(role=item.role, status="FAILED", reason_codes=["ACCOUNTING_FAILED"],
                          order_count=len(item.intents.intents))
        try:
            result = ShadowPortfolioTransition.apply(prior=item.prior, intents=item.intents,
                quotes=_quotes(item, rows), assumption=item.execution_assumption)
            _need(result.transition_status == "COMPLETE", "ACCOUNTING_FAILED")
            diagnostic.update(status="COMPLETE", reason_codes=[], filled_count=result.filled_count,
                no_fill_count=result.no_fill_count, valuation_status=result.valuation_status,
                nav_available=result.nav_after is not None, missing_valuation_count=len(result.missing_valuation_instruments),
                corporate_action_mode=result.corporate_action_mode, warnings=list(result.warnings))
            transitions.append(result)
        except _PROCESS_CONTROL:
            raise
        except Exception:
            transitions.append(None)
        diagnostics.append(diagnostic)
    return transitions, diagnostics


def _request_body(epoch, body, completion, inputs, logical, *, implementation):
    result = dict(schema_version=1, domain="FIRST_FORWARD_SETTLEMENT_REQUEST_V1", epoch_id=epoch,
        current_cycle_id=body["current_cycle_id"], trade_date=body["trade_date"],
        source_request_digest=completion["request_digest"], source_manifest_digest=completion["manifest_digest"],
        source_operation_id=completion["operation_id"], publication_success_digest=_raw(logical["publication_success.json"]),
        roles=[dict(role=i.role, prior_digest=i.prior.digest, intent_batch_digest=i.intents.digest,
                    execution_assumption_digest=i.execution_assumption.digest) for i in inputs],
        arrival_rule=ARRIVAL_RULE, implementation=implementation, semantic_members=c4._inventory(logical))
    if body["domain"] == "CONTINUING_FORWARD_INTENT_PAIR_REQUEST_V1":
        result.update(schema_version=2, domain="CONTINUING_FORWARD_SETTLEMENT_REQUEST_V1",
                      continuation=body["continuation"])
    return result


def _validate(members, manifest, epoch, expected, *, cycle_index=None):
    _budget(members)
    _keys(manifest, ("schema_version", "domain", "epoch_id", "current_cycle_id", "request_digest", "members"))
    request = _keys(_json(members["request.json"]), ("body", "request_digest"))
    body = request["body"]
    _need(request["request_digest"] == expected == _hash(body))
    source = _json(members["source_intent.json"])
    original, completion, inputs = _source(source, members["publication_success.json"], epoch, continuing=cycle_index is not None)
    if cycle_index is not None:
        _need(original["continuation"]["cycle_index"] == cycle_index, "CYCLE_INDEX_INVALID")
    day = _calendar(members["calendar_day.txt"], original, source)
    rows = _prices(_json(members["next_open_prices.json"]), day, original["trade_date"], inputs)
    transitions, diagnostics = _calculate(inputs, rows)
    _need(all(t is not None for t in transitions), "ACCOUNTING_FAILED")
    saved = _json(members["settlements.json"])
    _need(_same(saved, dict(schema_version=1, roles=[dict(role=r, transition=t.to_dict()) for r, t in zip(ROLES, transitions)])),
          "TRANSITION_RECOMPUTATION_MISMATCH")
    logical = {k: v for k, v in members.items() if k != "request.json"}
    rebuilt = _request_body(epoch, original, completion, inputs, logical, implementation=body["implementation"])
    # Fingerprint is recorded evidence, not a requirement to keep the current engine installed.
    implementation = _keys(body["implementation"], ("domain", "members"))
    _need(implementation["domain"] == ("CONTINUING_FORWARD_SETTLEMENT_IMPLEMENTATION_V1" if cycle_index is not None else "FIRST_FORWARD_SETTLEMENT_IMPLEMENTATION_V1")
          and [r["path"] for r in implementation["members"]] == list(IMPLEMENTATION_PATHS) + (["research/forward_continuation.py", "scripts/continue_forward.py"] if cycle_index is not None else []))
    for ref in implementation["members"]:
        _keys(ref, ("path", "sha256"))
        _digest(ref["sha256"])
    rebuilt["implementation"] = implementation
    _need(_same(body, rebuilt) and _same(manifest, dict(schema_version=2 if cycle_index is not None else 1, domain="CONTINUING_FORWARD_SETTLEMENT_BUNDLE_V1" if cycle_index is not None else "FIRST_FORWARD_SETTLEMENT_BUNDLE_V1",
        epoch_id=epoch, current_cycle_id=body["current_cycle_id"], request_digest=expected, members=c4._inventory(members))))
    states = tuple(SettlementAfterState(i.role, t.after_state, i.prior.digest, i.intents.digest,
        completion["request_digest"], completion["manifest_digest"], completion["operation_id"], expected)
        for i, t in zip(inputs, transitions))
    return body, states, diagnostics


def _read_bundle(root, target, identity, epoch, expected, *, completion=True, cycle_index=None):
    _need(c4._directory(root) == identity, "TARGET_REPLACED")
    _need(os.path.lexists(str(target)), "INCOMPLETE")
    target_identity = c4._directory(target)
    names = MEMBERS + ("manifest.json",) + (("completion.json",) if completion else ())
    actual = {p.name for p in target.iterdir()}
    _need(not actual - set(names), "CONFLICT")
    _need(actual == set(names), "INCOMPLETE")
    data, fingerprints = {}, {}
    for name in names:
        data[name], fingerprints[name] = c4.c3.surface._read_regular(target / name,
            maximum=MEMBER_LIMIT if name in MEMBERS else RECORD_LIMIT, private=True)
    manifest = _json(data["manifest.json"])
    _need(manifest.get("request_digest") == expected, "CONFLICT")
    body, states, diagnostics = _validate({n: data[n] for n in MEMBERS}, manifest, epoch, expected, cycle_index=cycle_index)
    record = None
    if completion:
        record = _keys(_json(data["completion.json"]), ("schema_version event operation_id epoch_id current_cycle_id "
            "request_digest manifest_digest target_binding_digest settled_at_utc").split())
        _need(record["schema_version"] == 1 and record["event"] == "SETTLEMENT_DATA_VERIFIED_BEFORE_COMPLETION"
              and record["epoch_id"] == epoch and record["current_cycle_id"] == body["current_cycle_id"]
              and record["request_digest"] == expected and record["manifest_digest"] == _raw(data["manifest.json"])
              and record["target_binding_digest"] == c4._binding(root, identity, target.name))
        _need(str(uuid.UUID(record["operation_id"])) == record["operation_id"])
        source = _json(data["source_intent.json"])
        original = _json(source["metadata"]["request.json"].encode("utf-8"))["body"]
        _need(c4._utc(record["settled_at_utc"]) >= c4._utc(original["time_policy"]["next_open_utc"]))
    for name in names:
        observed, fingerprint = c4.c3.surface._read_regular(target / name,
            maximum=MEMBER_LIMIT if name in MEMBERS else RECORD_LIMIT, private=True)
        _need(observed == data[name] and fingerprint == fingerprints[name], "TARGET_REPLACED")
    _need(c4._directory(root) == identity and c4._directory(target) == target_identity
          and {p.name for p in target.iterdir()} == set(names), "TARGET_REPLACED")
    summary = _summary("VERIFIED", request_digest=expected, manifest_digest=_raw(data["manifest.json"]),
        current_cycle_id=body["current_cycle_id"], trade_date=body["trade_date"],
        target_binding_digest=c4._binding(root, identity, target.name), bundle_verified=True,
        completion_record_verified=completion, accounting_pair_complete=True, state_chain_ready=completion,
        source_forward_record_linked=True, roles=diagnostics, operation_id=record["operation_id"] if record else None)
    if cycle_index is not None:
        meta = _metadata(data)
        _need(meta["source_completion"]["store_bindings"]["settlement"] ==
              c4._binding(root, identity, "CONTINUING_SETTLEMENT_ROOT"), "STORE_BINDING_INVALID")
        summary.update(cycle_index=cycle_index, whole_chain_verified=False)
    return summary, states, data


def _metadata(data):
    source = _json(data["source_intent.json"])
    return {"request": _json(data["request.json"])["body"],
            "source_completion": _json(source["metadata"]["completion.json"].encode("utf-8")),
            "source_request": _json(source["metadata"]["request.json"].encode("utf-8"))["body"]}


def inspect_first_forward_settlement(settlement_store_root, epoch_id, *, expected_request_digest):
    try:
        _digest(expected_request_digest)
        root, target, identity = c4._target(settlement_store_root, epoch_id)
        summary, states, data = _read_bundle(root, target, identity, epoch_id, expected_request_digest)
        return _result(summary, states, _metadata(data))
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        code = getattr(exc, "code", "BUNDLE_INVALID")
        status = code if code in ("INCOMPLETE", "CONFLICT") else "CONFLICT" if code in (
            "BUNDLE_INVALID", "TRANSITION_RECOMPUTATION_MISMATCH") else "UNCERTAIN"
        return _result(_summary(status, reason_codes=[code]))


def _guard(owner, root, paths):
    guard = c4.c3.surface.SourceMutationObserver(root, ())
    owner.guards.append(guard)
    guard.add_paths(paths)
    _need(guard.supported, "INPUT_OBSERVATION_UNSUPPORTED")
    return guard


def _build(intent_root, epoch, expected, success_path, provider_root, owner, summary, *, cycle_index=None):
    if cycle_index is None:
        root, target, _ = c4._target(intent_root, epoch)
        _guard(owner, root, (epoch,))
        observed = c4.inspect_first_forward_intent_pair(root, epoch, expected_request_digest=expected)
    else:
        from quantpits.research.forward_continuation import _target, inspect_next_forward_intent_pair
        root, target, _ = _target(intent_root, epoch, cycle_index)
        _guard(owner, root, (target.name,))
        observed = inspect_next_forward_intent_pair(intent_root, epoch, cycle_index, expected_request_digest=expected)
    _need(observed.status == "VERIFIED" and observed.d1_inputs is not None, "SOURCE_INTENT_INVALID")
    source = dict(schema_version=2 if cycle_index is not None else 1, metadata=observed.d1_metadata, roles=[dict(role=i.role,
        prior=i.prior.to_dict(), intents=i.intents.to_dict(), execution_assumption=i.execution_assumption.to_dict())
        for i in observed.d1_inputs])
    success_path = c4.c3.surface._physical_path(success_path, "success", directory=False)
    _guard(owner, success_path.parent, (success_path.name,))
    success_bytes, _ = c4.c3.surface._read_regular(success_path, maximum=MEMBER_LIMIT, private=True)
    body, completion, inputs = _source(source, success_bytes, epoch, continuing=cycle_index is not None)
    # Join the original physical completion to the same validated C4 observation.
    _need(completion["target_binding_digest"] == observed.to_safe_summary_dict()["target_binding_digest"])
    summary.update(current_cycle_id=body["current_cycle_id"], trade_date=body["trade_date"])
    now = _clock()
    _need(isinstance(now, datetime) and now.tzinfo is not None and now.utcoffset().total_seconds() == 0, "CLOCK_INVALID")
    _need(now >= c4._utc(body["time_policy"]["next_open_utc"]), "WAITING_FOR_DATA")
    provider = c4.c3.surface._physical_path(provider_root, "provider", directory=True)
    guard = _guard(owner, provider, ("calendars/day.txt",))
    day_bytes = prices._stable_read(provider, "calendars/day.txt")
    day = _calendar(day_bytes, body, source)
    union = sorted({instrument for item in inputs for instrument in _requested(item)})
    if union:
        guard.add_paths(["features/%s/%s.day.bin" % (i.lower(), field) for i in union for field in ("open", "factor")])
        _need(guard.supported, "INPUT_OBSERVATION_UNSUPPORTED")
        receipt = prices.QlibCashPriceObserver(provider, day, prices._digest_bytes(day_bytes)).observe_next_open(
            trade_date=body["trade_date"], requested_instruments=union).to_dict()
    else:
        receipt = dict(status="NOT_REQUIRED_EMPTY_EXPOSURE", trade_date=body["trade_date"], requested_instruments=[], rows=[])
    price_doc = dict(schema_version=1, arrival_rule=ARRIVAL_RULE, receipt=receipt)
    rows = _prices(price_doc, day, body["trade_date"], inputs)
    transitions, diagnostics = _calculate(inputs, rows)
    summary["roles"] = diagnostics
    _need(all(t is not None for t in transitions), "ACCOUNTING_FAILED")
    logical = {"source_intent.json": canonical(source), "publication_success.json": success_bytes,
        "calendar_day.txt": day_bytes, "next_open_prices.json": canonical(price_doc),
        "settlements.json": canonical(dict(schema_version=1, roles=[dict(role=r, transition=t.to_dict())
                                                                 for r, t in zip(ROLES, transitions)]))}
    request_body = _request_body(epoch, body, completion, inputs, logical, implementation=_implementation(continuing=cycle_index is not None))
    digest = _hash(request_body)
    members = {"request.json": canonical(dict(body=request_body, request_digest=digest)), **logical}
    manifest = dict(schema_version=2 if cycle_index is not None else 1, domain="CONTINUING_FORWARD_SETTLEMENT_BUNDLE_V1" if cycle_index is not None else "FIRST_FORWARD_SETTLEMENT_BUNDLE_V1", epoch_id=epoch,
        current_cycle_id=body["current_cycle_id"], request_digest=digest, members=c4._inventory(members))
    _need(len(canonical(manifest)) <= RECORD_LIMIT, "BUNDLE_BUDGET_EXCEEDED")
    _validate(members, manifest, epoch, digest, cycle_index=cycle_index)
    owner.check()
    return members, canonical(manifest), digest


# Kept as a local seam for write-boundary fault injection.
_write_member = c4._write_member


def _publish(root, target, identity, epoch, members, manifest, digest, owner, summary, *, cycle_index=None):
    root_fd = target_fd = None
    attempted = False
    states = None
    try:
        owner.check()
        _need(c4._directory(root) == identity, "TARGET_REPLACED")
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
                result = inspect_first_forward_settlement(root, epoch, expected_request_digest=digest)
            else:
                from quantpits.research.forward_continuation import inspect_next_forward_settlement
                result = inspect_next_forward_settlement(root.parent, epoch, cycle_index, expected_request_digest=digest)
            summary.update(result.to_safe_summary_dict())
            if result.status == "VERIFIED":
                summary["status"] = "ADOPTED"
            return result.after_states
        target_fd = os.open(target.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
        info = os.fstat(target_fd)
        target_identity = (info.st_dev, info.st_ino, info.st_mode)
        for name, data in list(members.items()) + [("manifest.json", manifest)]:
            _need(c4._directory(root) == identity and c4._directory(target) == target_identity, "TARGET_REPLACED")
            _need(len(data) <= (MEMBER_LIMIT if name in MEMBERS else RECORD_LIMIT), "BUNDLE_BUDGET_EXCEEDED")
            _write_member(target_fd, name, data)
        os.fsync(target_fd)
        os.fsync(root_fd)
        _, _, stored = _read_bundle(root, target, identity, epoch, digest, completion=False, cycle_index=cycle_index)
        _need(all(stored[n] == d for n, d in members.items()) and stored["manifest.json"] == manifest)
        owner.check()
        now = _clock()
        _need(isinstance(now, datetime) and now.tzinfo is not None and now.utcoffset().total_seconds() == 0, "CLOCK_INVALID")
        record = canonical(dict(schema_version=1, event="SETTLEMENT_DATA_VERIFIED_BEFORE_COMPLETION",
            operation_id=str(uuid.uuid4()), epoch_id=epoch, current_cycle_id=summary["current_cycle_id"],
            request_digest=digest, manifest_digest=_raw(manifest), target_binding_digest=c4._binding(root, identity, target.name),
            settled_at_utc=c4._stamp(now)))
        _need(len(record) <= RECORD_LIMIT, "BUNDLE_BUDGET_EXCEEDED")
        _write_member(target_fd, "completion.json", record)
        os.fsync(target_fd)
        os.fsync(root_fd)
        observed, states, stored = _read_bundle(root, target, identity, epoch, digest, cycle_index=cycle_index)
        _need(stored["completion.json"] == record and c4._directory(target) == target_identity, "TARGET_REPLACED")
        owner.check()
        summary.update(observed)
        summary.update(status="COMMITTED", did_write=True, write_state="COMMITTED")
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        if summary["did_write"] or attempted:
            summary.update(status="UNCERTAIN", did_write=True if summary["did_write"] else None,
                write_state="WRITTEN_UNVERIFIED" if summary["did_write"] else "UNKNOWN")
        else:
            summary["status"] = "PRECONDITION_BLOCKED"
        summary.update(reason_codes=[getattr(exc, "code", "PUBLICATION_FAILED")], state_chain_ready=False)
    finally:
        active = sys.exc_info()[1]
        interrupted = None
        for fd in (target_fd, root_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except _PROCESS_CONTROL as exc:
                    if not isinstance(active, _PROCESS_CONTROL) and interrupted is None:
                        interrupted = exc
                except Exception:
                    summary.update(status="UNCERTAIN", reason_codes=["DESCRIPTOR_CLOSE_FAILED"], state_chain_ready=False)
        if interrupted is not None:
            raise interrupted
    return states


def _run(intent_root, epoch, expected_intent, success_path, provider, settlement_root, expected=None):
    summary, states = _summary(), None
    owner = c4.c3._PreparationGuards()
    try:
        root, target, identity = c4._target(settlement_root, epoch)
        if expected is not None:
            _digest(expected)
            if os.path.lexists(str(target)):
                observed = inspect_first_forward_settlement(root, epoch, expected_request_digest=expected)
                summary.update(observed.to_safe_summary_dict())
                states = observed.after_states
                if observed.status == "VERIFIED":
                    summary["status"] = "ADOPTED"
            else:
                members, manifest, digest = _build(intent_root, epoch, expected_intent, success_path, provider, owner, summary)
                summary.update(request_digest=digest, target_binding_digest=c4._binding(root, identity, epoch))
                if digest != expected:
                    summary.update(status="REQUEST_MISMATCH", reason_codes=["REQUEST_MISMATCH"])
                else:
                    states = _publish(root, target, identity, epoch, members, manifest, digest, owner, summary)
        else:
            members, manifest, digest = _build(intent_root, epoch, expected_intent, success_path, provider, owner, summary)
            summary.update(status="READY", request_digest=digest, target_binding_digest=c4._binding(root, identity, epoch),
                           accounting_pair_complete=True, source_forward_record_linked=True)
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        code = getattr(exc, "code", "INPUT_INVALID")
        summary.update(status="UNCERTAIN" if summary["did_write"] is not False else
                       "WAITING_FOR_DATA" if code == "WAITING_FOR_DATA" else "PRECONDITION_BLOCKED",
                       reason_codes=[code], state_chain_ready=False)
    finally:
        active = sys.exc_info()[1]
        try:
            owner.close()
        except _PROCESS_CONTROL:
            if not isinstance(active, _PROCESS_CONTROL):
                raise
        except Exception:
            summary.update(status="UNCERTAIN" if summary["did_write"] is not False else "PRECONDITION_BLOCKED",
                           reason_codes=["INPUT_OBSERVER_CLOSE_FAILED"], state_chain_ready=False)
    return _result(summary, states)


def prepare_first_forward_settlement(intent_store_root, epoch_id, expected_intent_request_digest,
                                     publication_success_record_path, qlib_provider_root, *, settlement_store_root):
    return _run(intent_store_root, epoch_id, expected_intent_request_digest, publication_success_record_path,
                qlib_provider_root, settlement_store_root)


def publish_first_forward_settlement(intent_store_root, epoch_id, expected_intent_request_digest,
                                     publication_success_record_path, qlib_provider_root, *, settlement_store_root,
                                     expected_request_digest):
    if expected_request_digest is None:
        return _result(_summary(reason_codes=["DIGEST_INVALID"]))
    return _run(intent_store_root, epoch_id, expected_intent_request_digest, publication_success_record_path,
                qlib_provider_root, settlement_store_root, expected_request_digest)
