"""Explicit weekly continuation of a verified two-arm forward epoch.

Only a fresh predecessor read grants a planning join. Offline observations verify
this bundle, never the unread history. Store bindings are physical completion
metadata and are deliberately excluded from semantic requests.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

from quantpits.research import forward_intent_preparation as c3
from quantpits.research import forward_intent_publication as c4
from quantpits.research import forward_settlement as d1

_need, _keys, _digest = c4._need, c4._keys, c4._digest
RULE = "WEEKLY_LAST_SESSION_V1"
OPENING_POLICY = "VERIFIED_PREDECESSOR_AFTER_STATE_V1"
SOURCE_METADATA = ("request.json", "manifest.json", "completion.json", "calendar_day.txt",
                   "definitions.json", "calendar_future.txt")


def _index(value, minimum=2):
    _need(type(value) is int and value >= minimum, "CYCLE_INDEX_INVALID")
    return value


def _target(root, epoch, index):
    _index(index)
    c3.surface._identifier(epoch, "epoch")
    root = c3.surface._physical_path(root, "continuing_store", directory=True)
    c4._directory(root)
    parent = root / epoch
    return parent, parent / str(index), c4._directory(parent)


def _bindings(intent_root, settlement_root, epoch, index):
    result = {}
    for name, value in (("intent", intent_root), ("settlement", settlement_root)):
        root, _, identity = _target(value, epoch, index)
        result[name] = c4._binding(root, identity, "CONTINUING_%s_ROOT" % name.upper())
    _need(result["intent"] != result["settlement"], "STORE_BINDING_INVALID")
    _need(c3.surface._physical_path(intent_root, "intent", directory=True) !=
          c3.surface._physical_path(settlement_root, "settlement", directory=True), "STORE_BINDING_INVALID")
    return result


def _week(day):
    parsed = date.fromisoformat(day)
    return parsed - timedelta(days=parsed.weekday())


def _schedule(day_bytes, future_bytes, continuation, anchor, trade):
    day, future = c3._calendar(day_bytes), c3._calendar(future_bytes)
    _need(future[-1] >= day[-1], "SCHEDULE_UNAVAILABLE")
    _need([d for d in day if d <= min(day[-1], future[-1])] ==
          [d for d in future if d <= min(day[-1], future[-1])], "CALENDAR_JOIN_INVALID")
    _need(continuation["schedule"]["rule"] == RULE, "SCHEDULE_INVALID")
    first = continuation["schedule"]["first_anchor"]
    predecessor = continuation["predecessor"]["current_cycle_id"]
    _need(first <= predecessor < anchor, "GAP_DETECTED")
    for previous in (first, predecessor):
        _need(previous in future, "SCHEDULE_UNAVAILABLE")
        week = _week(previous)
        _need(any(_week(d) > week for d in future), "SCHEDULE_UNAVAILABLE")
        _need(max(d for d in future if _week(d) == week) == previous, "ANCHOR_NOT_WEEK_END")
    later = [d for d in future if _week(d) > _week(predecessor)]
    _need(bool(later), "SCHEDULE_UNAVAILABLE")
    target_week = _week(later[0])
    _need(any(_week(d) > target_week for d in later), "SCHEDULE_UNAVAILABLE")
    expected = max(d for d in later if _week(d) == target_week)
    _need(anchor == expected, "GAP_DETECTED")
    _need(anchor in day and anchor in future, "SCHEDULE_UNAVAILABLE")
    _need(trade == next(d for d in future if d > anchor), "SCHEDULE_INVALID")


def _reference(observation):
    safe = observation.to_safe_summary_dict()
    return {key: safe[key] for key in ("request_digest", "manifest_digest", "operation_id")}


def _validate_continuation(value, epoch, compiled, selectors):
    _keys(value, ("cycle_index", "epoch_id", "first_intent", "frozen_selectors", "definition_request_digest",
                  "predecessor", "schedule"))
    index = _index(value["cycle_index"])
    _need(value["epoch_id"] == epoch, "EPOCH_JOIN_INVALID")
    _keys(value["first_intent"], ("request_digest", "manifest_digest", "operation_id"))
    _validate_reference(value["first_intent"])
    frozen = {k: v for k, v in selectors.items() if k not in ("current_cycle_id", "signal_capsule_id")}
    _need(value["frozen_selectors"] == frozen and
          value["definition_request_digest"] == dict(compiled.request_digest), "FROZEN_DEFINITION_MISMATCH")
    schedule = _keys(value["schedule"], ("rule", "first_anchor", "market_timezone"))
    _need(schedule["rule"] == RULE and type(schedule["market_timezone"]) is str
          and c4.gettz(schedule["market_timezone"]) is not None, "SCHEDULE_INVALID")
    c3.surface._date(schedule["first_anchor"], "first_anchor")
    predecessor = _keys(value["predecessor"], ("cycle_index", "kind", "current_cycle_id", "trade_date",
                                               "intent", "settlement", "roles"))
    _index(predecessor["cycle_index"], 1)
    _need(predecessor["cycle_index"] + 1 == index, "PREDECESSOR_INDEX_INVALID")
    _need(predecessor["kind"] == ("FIRST" if index == 2 else "CONTINUING"), "PREDECESSOR_KIND_INVALID")
    for key in ("intent", "settlement"):
        _keys(predecessor[key], ("request_digest", "manifest_digest", "operation_id"))
        _validate_reference(predecessor[key])
    for key in ("current_cycle_id", "trade_date"):
        c3.surface._date(predecessor[key], key)
    _need(schedule["first_anchor"] <= predecessor["current_cycle_id"] < predecessor["trade_date"], "PREDECESSOR_DATE_INVALID")
    if index == 2:
        _need(predecessor["intent"] == value["first_intent"] and
              predecessor["current_cycle_id"] == schedule["first_anchor"], "FIRST_INTENT_JOIN_INVALID")
    else:
        _need(predecessor["current_cycle_id"] > schedule["first_anchor"], "PREDECESSOR_DATE_INVALID")
    rows = predecessor["roles"]
    _need(type(rows) is list and [r["role"] for r in rows] == list(c3.ROLES), "PREDECESSOR_ROLE_INVALID")
    for row in rows:
        _keys(row, ("role", "portfolio_id", "after_state_digest"))
        c3.surface._identifier(row["portfolio_id"], "portfolio_id")
        _digest(row["after_state_digest"])
    _need(rows[0]["portfolio_id"] != rows[1]["portfolio_id"], "PREDECESSOR_ROLE_INVALID")


def _validate_reference(value):
    import uuid
    _digest(value["request_digest"])
    _digest(value["manifest_digest"])
    _need(str(uuid.UUID(value["operation_id"])) == value["operation_id"], "PREDECESSOR_REFERENCE_INVALID")


def _validate_prior(continuation, index, role, prior, state, compiled, selectors, anchor):
    predecessor = continuation["predecessor"]
    _need(prior.to_dict() == state and prior.as_of_date == predecessor["trade_date"] <= anchor,
          "PREDECESSOR_STATE_INVALID")
    _need(predecessor["roles"][index] == dict(role=role, portfolio_id=prior.portfolio_id,
                                           after_state_digest=prior.digest), "PREDECESSOR_STATE_INVALID")
    strategy = (compiled.champion, compiled.challenger)[index].to_dict()
    _need(strategy["data_cutoff"] <= selectors["definition_evidence_cycle_id"] <=
          selectors["bootstrap_source_cycle_id"] < anchor and anchor >= strategy["effective_cycle"],
          "FROZEN_DEFINITION_MISMATCH")


def _watch(guards, root, paths):
    guard = c3.surface.SourceMutationObserver(root, paths)
    guards.append(guard)
    _need(guard.supported, "INPUT_OBSERVATION_UNSUPPORTED")


def _observe_predecessor(selection, guards, compiled, bootstrap_id, model_id, source_cycle, evidence_cycle, anchor):
    epoch, index = selection["epoch_id"], _index(selection["cycle_index"])
    previous_index = _index(selection["predecessor_cycle_index"], 1)
    _need(previous_index + 1 == index, "PREDECESSOR_INDEX_INVALID")
    first_root, _, _ = c4._target(selection["first_intent_store_root"], epoch)
    _watch(guards, first_root, (epoch,))
    first = c4.inspect_first_forward_intent_pair(first_root, epoch,
        expected_request_digest=selection["expected_first_intent_request_digest"])
    _need(first.status == "VERIFIED" and first.d1_inputs is not None, "FIRST_INTENT_INVALID")
    first_body = c4._json(first.d1_metadata["request.json"].encode())["body"]
    frozen = {k: v for k, v in first_body["input_provenance"]["selectors"].items()
              if k not in ("current_cycle_id", "signal_capsule_id")}
    _need(frozen == dict(definition_set_id=compiled.definition_set_id, definition_evidence_cycle_id=evidence_cycle,
                        bootstrap_set_id=bootstrap_id, bootstrap_source_cycle_id=source_cycle, model_capsule_id=model_id)
          and first_body["input_provenance"]["definition"] == dict(compiled.request_digest), "FROZEN_DEFINITION_MISMATCH")
    bindings = _bindings(selection["intent_store_root"], selection["settlement_store_root"], epoch, index)
    for key in ("intent_store_root", "settlement_store_root"):
        root, _, _ = _target(selection[key], epoch, index)
        # Root identities also remain stable during preparation and publication.
        _watch(guards, root.parent, ())
        _watch(guards, root, ())
    predecessor_root = selection["predecessor_settlement_store_root"]
    if previous_index == 1:
        root, _, _ = c4._target(predecessor_root, epoch)
        _watch(guards, root, (epoch,))
        previous = d1.inspect_first_forward_settlement(root, epoch,
            expected_request_digest=selection["expected_predecessor_request_digest"])
    else:
        root, target, _ = _target(predecessor_root, epoch, previous_index)
        _watch(guards, root, (target.name,))
        _need(root == _target(selection["settlement_store_root"], epoch, index)[0], "STORE_BINDING_INVALID")
        previous = inspect_next_forward_settlement(predecessor_root, epoch, previous_index,
            expected_request_digest=selection["expected_predecessor_request_digest"])
    _need(previous.status == "VERIFIED" and previous.to_safe_summary_dict()["state_chain_ready"]
          and previous.after_states is not None, "PREDECESSOR_SETTLEMENT_INVALID")
    meta = previous.continuation_metadata
    previous_body = meta["source_request"]
    _need(previous_body["epoch_id"] == epoch, "EPOCH_JOIN_INVALID")
    source_ref = dict(request_digest=meta["request"]["source_request_digest"],
                      manifest_digest=meta["request"]["source_manifest_digest"],
                      operation_id=meta["request"]["source_operation_id"])
    if previous_index == 1:
        _need(source_ref == _reference(first) and previous_body == first_body, "FIRST_INTENT_JOIN_INVALID")
    else:
        saved = previous_body["continuation"]
        _need(saved["first_intent"] == _reference(first) and saved["frozen_selectors"] == frozen
              and saved["cycle_index"] == previous_index
              and saved["schedule"] == dict(rule=RULE, first_anchor=first_body["current_cycle_id"], market_timezone=first_body["time_policy"]["market_timezone"]), "PREDECESSOR_JOIN_INVALID")
        _need(meta["source_completion"]["store_bindings"] == bindings, "STORE_BINDING_INVALID")
    states = previous.after_states
    _need([s.role for s in states] == list(c3.ROLES), "PREDECESSOR_ROLE_INVALID")
    for old, initial in zip(states, first.d1_inputs):
        _need(old.role == initial.role and old.after_state.portfolio_id == initial.prior.portfolio_id
              and old.source_request_digest == source_ref["request_digest"]
              and old.source_manifest_digest == source_ref["manifest_digest"]
              and old.source_operation_id == source_ref["operation_id"]
              and old.settlement_request_digest == previous.request_digest, "PREDECESSOR_STATE_INVALID")
    value = dict(cycle_index=index, epoch_id=epoch, first_intent=_reference(first), frozen_selectors=frozen,
        definition_request_digest=dict(compiled.request_digest), schedule=dict(rule=RULE, first_anchor=first_body["current_cycle_id"], market_timezone=first_body["time_policy"]["market_timezone"]),
        predecessor=dict(cycle_index=previous_index, kind="FIRST" if previous_index == 1 else "CONTINUING",
            current_cycle_id=previous_body["current_cycle_id"], trade_date=previous_body["trade_date"],
            intent=source_ref, settlement=_reference(previous), roles=[dict(role=s.role,
                portfolio_id=s.after_state.portfolio_id, after_state_digest=s.after_state.digest) for s in states]))
    _validate_continuation(value, epoch, compiled, dict(frozen, current_cycle_id=anchor, signal_capsule_id="current"))
    _need(selection["market_timezone"] == first_body["time_policy"]["market_timezone"], "SCHEDULE_INVALID")
    return tuple(s.after_state for s in states), value


def _intent_summary(index, **changes):
    return c4._summary(cycle_index=index, cycle_intent_published=False,
                       predecessor_join_observed=False, whole_chain_verified=False, planning_roles=None, **changes)


def inspect_next_forward_intent_pair(intent_store_root, epoch_id, cycle_index, *, expected_request_digest):
    try:
        _digest(expected_request_digest)
        root, target, identity = _target(intent_store_root, epoch_id, cycle_index)
        safe, inputs, data = c4._read_bundle(root, target, identity, epoch_id, expected_request_digest, cycle_index=cycle_index)
        return c4._result(c4.FirstIntentBundleObservation, safe, inputs, {n: data[n].decode("utf-8") for n in SOURCE_METADATA}, data)
    except c4._PROCESS_CONTROL:
        raise
    except Exception as exc:
        code = getattr(exc, "code", "BUNDLE_INVALID")
        status = code if code in ("INCOMPLETE", "CONFLICT") else "CONFLICT" if code == "BUNDLE_INVALID" else "UNCERTAIN"
        return c4._result(c4.FirstIntentBundleObservation, _intent_summary(None, status=status, reason_codes=[code]))


def _intent_run(args, selection, deadline, next_open, expected=None):
    index, epoch = selection["cycle_index"], selection["epoch_id"]
    safe, inputs = _intent_summary(None), None
    owner = c3._PreparationGuards()
    try:
        root, target, identity = _target(selection["intent_store_root"], epoch, index)
        safe["cycle_index"] = index
        if expected is not None:
            _digest(expected)
            if os.path.lexists(str(target)):
                observed = inspect_next_forward_intent_pair(selection["intent_store_root"], epoch, index,
                                                            expected_request_digest=expected)
                safe, inputs = observed.to_safe_summary_dict(), observed.d1_inputs
                if observed.status == "VERIFIED":
                    safe["status"] = "ADOPTED"
                return c4._result(c4.FirstIntentPublicationResult, safe, inputs)
        bindings = _bindings(selection["intent_store_root"], selection["settlement_store_root"], epoch, index)
        policy = c4._policy(deadline, next_open, selection["market_timezone"], OPENING_POLICY, continuing=True)
        gate = c4._TimeGate(policy)
        gate.check("BEFORE_PREPARATION")
        prepared = c3._prepare_forward_intent(*args, _guard_owner=owner, _continuation=selection)
        safe["planning_roles"] = prepared.to_safe_summary_dict()["roles"]
        if not prepared.intent_pair_prepared:
            safe.update(status=prepared.status, reason_codes=prepared.to_safe_summary_dict()["reason_codes"])
        else:
            members, manifest, digest = c4._build(prepared, epoch, policy, continuing=True)
            owner.check()
            _need(bindings == _bindings(selection["intent_store_root"], selection["settlement_store_root"], epoch, index), "STORE_BINDING_INVALID")
            gate.check("PREPARATION_VERIFIED")
            safe.update(status="READY", request_digest=digest, current_cycle_id=args[4],
                trade_date=prepared.to_safe_summary_dict()["trade_date"], predecessor_join_observed=True,
                target_binding_digest=c4._binding(root, identity, target.name, store_bindings=bindings), time_observations=gate.observations,
                order_counts=[len(p.intents.intents) for p in prepared.pair.plans])
            if os.path.lexists(str(target)):
                safe.update(status="CONFLICT", reason_codes=["CONFLICT"])
            elif expected is not None:
                if digest != expected:
                    safe.update(status="REQUEST_MISMATCH", reason_codes=["REQUEST_MISMATCH"])
                else:
                    safe, inputs = c4._publish_bundle(root, target, identity, epoch, members, manifest, digest,
                        gate, owner, cycle_index=index, store_bindings=bindings)
    except c4._PROCESS_CONTROL:
        raise
    except Exception as exc:
        safe.update(status="UNCERTAIN" if safe["did_write"] is not False else "PRECONDITION_BLOCKED",
                    reason_codes=[getattr(exc, "code", "INPUT_INVALID")], prospective_claim=False,
                    cycle_intent_published=False, predecessor_join_observed=False)
    finally:
        active = sys.exc_info()[1]
        try:
            owner.close()
        except c4._PROCESS_CONTROL:
            if not isinstance(active, c4._PROCESS_CONTROL):
                raise
        except Exception:
            safe.update(status="UNCERTAIN" if safe["did_write"] is not False else "PRECONDITION_BLOCKED",
                        reason_codes=["INPUT_OBSERVER_CLOSE_FAILED"], prospective_claim=False,
                        cycle_intent_published=False, predecessor_join_observed=False)
    return c4._result(c4.FirstIntentPublicationResult, safe, inputs)


def prepare_next_forward_intent_publication(
    production_workspace_root, research_workspace_root, engine_root, qlib_provider_root, current_cycle_id,
    activation_path, definition_store_root, evidence_store_root, bootstrap_store_root, bootstrap_set_id,
    signal_capsule_store_root, signal_capsule_id, model_capsule_store_root, model_capsule_id, *,
    intent_store_root, settlement_store_root, epoch_id, cycle_index, first_intent_store_root,
    expected_first_intent_request_digest, predecessor_settlement_store_root, predecessor_cycle_index,
    expected_predecessor_request_digest, decision_deadline_utc, next_open_utc, market_timezone,
):
    return _intent_api(locals())


def publish_next_forward_intent_pair(
    production_workspace_root, research_workspace_root, engine_root, qlib_provider_root, current_cycle_id,
    activation_path, definition_store_root, evidence_store_root, bootstrap_store_root, bootstrap_set_id,
    signal_capsule_store_root, signal_capsule_id, model_capsule_store_root, model_capsule_id, *,
    intent_store_root, settlement_store_root, epoch_id, cycle_index, first_intent_store_root,
    expected_first_intent_request_digest, predecessor_settlement_store_root, predecessor_cycle_index,
    expected_predecessor_request_digest, decision_deadline_utc, next_open_utc, market_timezone, expected_request_digest,
):
    if expected_request_digest is None:
        return c4._result(c4.FirstIntentPublicationResult, _intent_summary(None, reason_codes=["DIGEST_INVALID"]))
    return _intent_api(locals())


_CURRENT_ARGUMENTS = ("production_workspace_root", "research_workspace_root", "engine_root", "qlib_provider_root",
    "current_cycle_id", "activation_path", "definition_store_root", "evidence_store_root", "bootstrap_store_root",
    "bootstrap_set_id", "signal_capsule_store_root", "signal_capsule_id", "model_capsule_store_root", "model_capsule_id")


def _intent_api(values):
    args = tuple(values.pop(name) for name in _CURRENT_ARGUMENTS)
    deadline, next_open = values.pop("decision_deadline_utc"), values.pop("next_open_utc")
    expected = values.pop("expected_request_digest", None)
    return _intent_run(args, values, deadline, next_open, expected)


def inspect_next_forward_settlement(settlement_store_root, epoch_id, cycle_index, *, expected_request_digest):
    try:
        _digest(expected_request_digest)
        root, target, identity = _target(settlement_store_root, epoch_id, cycle_index)
        safe, states, data = d1._read_bundle(root, target, identity, epoch_id, expected_request_digest, cycle_index=cycle_index)
        meta = d1._metadata(data)
        _need(meta["source_completion"]["store_bindings"]["settlement"] ==
              c4._binding(root, identity, "CONTINUING_SETTLEMENT_ROOT"), "STORE_BINDING_INVALID")
        return d1._result(safe, states, meta, data)
    except c4._PROCESS_CONTROL:
        raise
    except Exception as exc:
        code = getattr(exc, "code", "BUNDLE_INVALID")
        status = code if code in ("INCOMPLETE", "CONFLICT") else "CONFLICT" if code in (
            "BUNDLE_INVALID", "TRANSITION_RECOMPUTATION_MISMATCH") else "UNCERTAIN"
        return d1._result(d1._summary(status, cycle_index=None, whole_chain_verified=False, reason_codes=[code]))


def _settlement_run(intent_root, epoch, index, expected_intent, success, provider, settlement_root, expected=None):
    safe, states = d1._summary(cycle_index=None, whole_chain_verified=False), None
    owner = c3._PreparationGuards()
    try:
        root, target, identity = _target(settlement_root, epoch, index)
        safe["cycle_index"] = index
        if expected is not None:
            _digest(expected)
            if os.path.lexists(str(target)):
                observed = inspect_next_forward_settlement(settlement_root, epoch, index, expected_request_digest=expected)
                safe, states = observed.to_safe_summary_dict(), observed.after_states
                if observed.status == "VERIFIED":
                    safe["status"] = "ADOPTED"
                return d1._result(safe, states)
        members, manifest, digest = d1._build(intent_root, epoch, expected_intent, success, provider, owner, safe, cycle_index=index)
        meta = d1._metadata(members)
        _need(meta["source_completion"]["store_bindings"] == _bindings(intent_root, settlement_root, epoch, index), "STORE_BINDING_INVALID")
        safe.update(status="READY", request_digest=digest, target_binding_digest=c4._binding(root, identity, target.name),
                    accounting_pair_complete=True, source_forward_record_linked=True)
        if os.path.lexists(str(target)):
            safe.update(status="CONFLICT", reason_codes=["CONFLICT"])
        elif expected is not None:
            if expected != digest:
                safe.update(status="REQUEST_MISMATCH", reason_codes=["REQUEST_MISMATCH"])
            else:
                states = d1._publish(root, target, identity, epoch, members, manifest, digest, owner, safe, cycle_index=index)
    except c4._PROCESS_CONTROL:
        raise
    except Exception as exc:
        code = getattr(exc, "code", "INPUT_INVALID")
        safe.update(status="UNCERTAIN" if safe["did_write"] is not False else
                    "WAITING_FOR_DATA" if code == "WAITING_FOR_DATA" else "PRECONDITION_BLOCKED",
                    reason_codes=[code], state_chain_ready=False)
    finally:
        active = sys.exc_info()[1]
        try:
            owner.close()
        except c4._PROCESS_CONTROL:
            if not isinstance(active, c4._PROCESS_CONTROL):
                raise
        except Exception:
            safe.update(status="UNCERTAIN" if safe["did_write"] is not False else "PRECONDITION_BLOCKED",
                        reason_codes=["INPUT_OBSERVER_CLOSE_FAILED"], state_chain_ready=False)
    return d1._result(safe, states)


def prepare_next_forward_settlement(intent_store_root, epoch_id, cycle_index, expected_intent_request_digest,
                                    publication_success_record_path, qlib_provider_root, *, settlement_store_root):
    return _settlement_run(intent_store_root, epoch_id, cycle_index, expected_intent_request_digest,
                           publication_success_record_path, qlib_provider_root, settlement_store_root)


def publish_next_forward_settlement(intent_store_root, epoch_id, cycle_index, expected_intent_request_digest,
                                    publication_success_record_path, qlib_provider_root, *, settlement_store_root,
                                    expected_request_digest):
    if expected_request_digest is None:
        return d1._result(d1._summary(reason_codes=["DIGEST_INVALID"], cycle_index=None, whole_chain_verified=False))
    return _settlement_run(intent_store_root, epoch_id, cycle_index, expected_intent_request_digest,
                           publication_success_record_path, qlib_provider_root, settlement_store_root, expected_request_digest)
