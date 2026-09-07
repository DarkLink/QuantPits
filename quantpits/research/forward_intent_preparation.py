"""Read-only preparation of the first matched Champion/Challenger intent pair."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Any, Tuple

from quantpits.evidence.contracts import canonical_json_bytes
from quantpits.research import decision_surface as surface
from quantpits.research import replay
from quantpits.research.accounting import ShadowPortfolioState
from quantpits.research.historical_cycle import QlibCashPriceObserver
from quantpits.research.intents import AnchorPriceSnapshot, CurrentRuleIntentDefinition, CurrentRuleShadowIntentPlanner

ROLES = ("CHAMPION", "CHALLENGER")
_PROCESS_CONTROL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class _Blocked(Exception):
    def __init__(self, code):
        self.code = code


def _require(condition, code):
    if not condition:
        raise _Blocked(code)


def _hash(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw(data):
    return hashlib.sha256(data).hexdigest()


class _PreparedPair(NamedTuple):
    priors: Tuple[Any, Any]
    rankings: Tuple[Any, Any]
    prices: Tuple[Any, Any]
    plans: Tuple[Any, Any]
    definitions: Any
    price_receipt: Any
    input_provenance: bytes
    calendar_bytes: Tuple[bytes, bytes]
    bootstrap_bytes: bytes


class FirstForwardIntentPreparation:
    """An in-memory observation; its summary cannot be used to restore a pair."""
    __slots__ = ("_summary", "_pair")

    def __setattr__(self, name, value):
        raise TypeError("preparations are immutable")

    def __init__(self, *args, **kwargs):
        raise TypeError("preparations are produced by prepare_first_forward_intent")

    @property
    def status(self):
        return json.loads(self._summary)["status"]

    @property
    def intent_pair_prepared(self):
        return self.status == "PREPARED"

    @property
    def pair(self):
        return self._pair

    def to_safe_summary_dict(self):
        return json.loads(self._summary)


def _result(summary, pair=None):
    value = object.__new__(FirstForwardIntentPreparation)
    object.__setattr__(value, "_summary", canonical_json_bytes(summary))
    object.__setattr__(value, "_pair", pair if summary["status"] == "PREPARED" else None)
    return value


def _summary():
    return {
        "schema_version": 1, "status": "PRECONDITION_BLOCKED", "reason_codes": [],
        "current_cycle_id": None, "bootstrap_source_cycle_id": None, "trade_date": None,
        "input_digest": None, "preparation_digest": None,
        "roles": [{"role": role, "status": "NOT_RUN", "reason": None,
                   "coverage_counts": None, "order_count": None, "buy_count": None,
                   "sell_count": None, "shortage_count": None, "pending_count": None,
                   "ranking_digest": None, "prior_digest": None, "planning_result_digest": None,
                   "intent_batch_digest": None, "price_digest": None} for role in ROLES],
        "sealed_ranking_digest": None, "recomputed_ranking_digest": None,
        "calendar_match_status": None, "price_provenance": None,
        "historical_materialization_relation": "unverified", "price_counts": None,
        "comparison": None, "engine_commit": None, "engine_tree": None,
        "implementation_digest": None,
        "intent_pair_prepared": False, "intent_publication_capability": False,
        "epoch_started": False, "prospective_claim": False,
        "promotion_capability": False, "did_write": False,
    }


def _metadata(paths):
    """Continuity of selected inputs whose contents are checked by their readers."""
    rows = []
    for selected in paths:
        children = [selected]
        if selected.is_dir() and not selected.is_symlink():
            children += sorted(selected.rglob("*"))
        for path in children:
            try:
                info = os.lstat(str(path))
            except FileNotFoundError:
                rows.append((str(path), None))
                continue
            _require(not stat.S_ISLNK(info.st_mode), "INPUT_PATH_INVALID")
            rows.append((str(path), info.st_dev, info.st_ino, info.st_mode,
                         info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink))
    return tuple(rows)


def _read_json(path):
    return surface._strict_json(surface._read_regular(path, private=True)[0], "selected_input")


def _bootstrap(production, store, identifier, candidate, evidence, anchor):
    source_cycle, source = surface._bootstrap_authority(production, store, identifier, candidate, evidence)
    target = store / identifier
    manifest = _read_json(target / "bootstrap_manifest.json")
    compiled = candidate.compiled_definitions
    strategies = (compiled.champion.to_dict(), compiled.challenger.to_dict())
    _require(all(strategy["data_cutoff"] <= evidence.evidence_cycle_id <= source_cycle < anchor
                 and anchor >= strategy["effective_cycle"] for strategy in strategies), "CYCLE_JOIN_INVALID")
    for field in ("definition_evidence_cycle_id", "bootstrap_source_cycle_id",
                  "definition_evidence_request_digest", "definition_evidence_manifest_digest",
                  "definition_evidence_operation_id", "source_portfolio_raw_digest",
                  "source_portfolio_semantic_digest"):
        _require(source[field] == manifest[field], "BOOTSTRAP_SOURCE_JOIN_INVALID")
    _require(manifest["definition_evidence_cycle_id"] == evidence.evidence_cycle_id
             and manifest["definition_evidence_operation_id"] == evidence.evidence_receipt.operation_id,
             "BOOTSTRAP_DEFINITION_JOIN_INVALID")
    from quantpits.research.forward_bootstrap import _economic_payload
    states = []
    for role in ROLES:
        raw = _read_json(target / (role.lower() + "_state.json"))
        state = ShadowPortfolioState.from_dict({key: value for key, value in raw.items() if key != "digest"})
        _require(state.to_dict() == raw, "BOOTSTRAP_STATE_DIGEST_INVALID")
        states.append(state)
    states = tuple(states)
    _require(states[0].portfolio_id != states[1].portfolio_id, "BOOTSTRAP_PORTFOLIO_ID_INVALID")
    expected_roles = []
    for role, strategy, state in zip(ROLES, strategies, states):
        _require(state.as_of_date == source_cycle, "BOOTSTRAP_STATE_DATE_INVALID")
        _require(surface._digest(_economic_payload(state)) == manifest["economic_state_digest"],
                 "BOOTSTRAP_ECONOMICS_INVALID")
        portfolio_payload = {
            "domain": "MATCHED_FORWARD_PORTFOLIO_ID_V1", "role": role,
            "bootstrap_source_cycle_id": source_cycle,
            "source_portfolio_raw_digest": source["source_portfolio_raw_digest"],
            "definition_set_id": candidate.definition_set_id, "strategy_id": strategy["strategy_id"],
        }
        _require(state.portfolio_id == "portfolio." + _hash(portfolio_payload), "BOOTSTRAP_PORTFOLIO_ID_INVALID")
        expected_roles.append({"role": role, "strategy_id": strategy["strategy_id"],
                               "portfolio_id": state.portfolio_id, "state_digest": surface._digest(state.to_dict())})
    _require(manifest["roles"] == expected_roles, "BOOTSTRAP_ROLE_JOIN_INVALID")
    from quantpits.research.forward_portfolio_source import observe_forward_portfolio_source
    from quantpits.research.forward_bootstrap import _build_states, _source_receipt
    observed_source = observe_forward_portfolio_source(production, source_cycle)
    _require(observed_source.bootstrap_source_capability, "BOOTSTRAP_SOURCE_UNVERIFIED")
    _require(source == _source_receipt(evidence, observed_source, evidence.evidence_cycle_id, source_cycle),
             "BOOTSTRAP_SOURCE_RECEIPT_INVALID")
    expected_states = _build_states(observed_source, candidate)[:2]
    _require(tuple(state.to_dict() for state in states) == tuple(state.to_dict() for state in expected_states),
             "BOOTSTRAP_SOURCE_ECONOMICS_INVALID")
    _require(manifest["source_state_interpretation"]
             == "SEALED_PRODUCTION_PORTFOLIO_AT_BOOTSTRAP_SOURCE_CYCLE_V1", "BOOTSTRAP_STATE_INTERPRETATION_INVALID")
    return source_cycle, states, manifest


def _calendar(data):
    dates = tuple(surface._date(line, "calendar") for line in data.decode("utf-8").splitlines())
    _require(bool(dates) and dates == tuple(sorted(set(dates))), "CALENDAR_INVALID")
    return dates


def _parity(sealed, rebuilt):
    _require((sealed.eligible_count, sealed.scored_count, sealed.missing_count)
             == (rebuilt.eligible_count, rebuilt.scored_count, rebuilt.missing_count), "CHAMPION_PARITY_COVERAGE")
    for left, right in zip(sealed.rows, rebuilt.rows):
        _require(all(left[key] == right[key] for key in ("instrument", "eligible", "scored", "rank", "coverage_status")),
                 "CHAMPION_PARITY_ORDER")
        _require(abs(float(left["raw_score"]) - float(right["raw_score"])) <= 1e-12,
                 "CHAMPION_PARITY_SCORE")


def _engine(engine, manifest):
    expected = surface._git_blob_projection(engine, surface._engine_commit(manifest))
    for row in expected["members"]:
        data = surface._read_regular(engine / row["path"])[0]
        _require(surface._digest(data, "raw_bytes") == row["digest"], "RUNTIME_CODE_MISMATCH")
        # Python imports must consume the repository whose code was observed.
        loaded = Path(__file__).resolve().parents[1] / Path(row["path"]).relative_to("quantpits")
        _require(loaded.read_bytes() == data, "LOADED_CODE_MISMATCH")
    identities = []
    for revision in ("HEAD", "HEAD^{tree}"):
        result = subprocess.run(["git", "rev-parse", revision], cwd=str(engine),
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=True)
        identity = result.stdout.decode().strip()
        _require(len(identity) == 40 and all(c in "0123456789abcdef" for c in identity), "ENGINE_IDENTITY_INVALID")
        identities.append(identity)
    files = ("forward_intent_preparation.py", "replay.py", "intents.py", "accounting.py",
             "historical_cycle.py", "decision_surface.py", "forward_definition_evidence.py",
             "forward_observation.py", "forward_definitions.py", "definition_store.py",
             "forward_bootstrap.py", "forward_portfolio_source.py", "model_artifact_capsule.py", "signal_input_capsule.py")
    implementation = [{"path": name, "digest": _raw(Path(__file__).with_name(name).read_bytes())} for name in files]
    return identities[0], identities[1], _hash(implementation)


def _plan_pair(priors, rankings, snapshots, definition, anchor, trade, market, summary):
    plans = []
    for index, (prior, ranking, prices) in enumerate(zip(priors, rankings, snapshots)):
        role = summary["roles"][index]
        try:
            # The legacy primitive prints private holdings and arithmetic.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                plan = CurrentRuleShadowIntentPlanner.plan(
                    portfolio_id=prior.portfolio_id, cycle_id=anchor, market=market,
                    anchor_date=anchor, trade_date=trade, definition=definition,
                    ranking=ranking, prior=prior, prices=prices,
                )
            _require(plan.status == "COMPLETE" and plan.parity_checked and plan.selection_checked,
                     "PLANNER_INCOMPLETE")
            _require(dict(plan.request) == {"portfolio_id": prior.portfolio_id, "cycle_id": anchor,
                     "market": market, "anchor_date": anchor, "trade_date": trade}
                     and plan.prior_state_digest == prior.digest
                     and plan.ranking_digest == _raw(ranking.to_csv_bytes())
                     and plan.prices.digest == prices.digest and plan.definition_digest == definition.digest,
                     "PLANNER_JOIN_INVALID")
            role.update(status="COMPLETE", order_count=len(plan.intents.intents),
                        buy_count=len(plan.selected_buy_instruments), sell_count=len(plan.production_sell_proposal),
                        shortage_count=plan.buy_intent_shortage,
                        pending_count=len(plan.holding_classifications["forced_pending"]),
                        planning_result_digest=plan.result_digest, intent_batch_digest=plan.intents.digest)
            plans.append(plan)
        except _PROCESS_CONTROL:
            raise
        except Exception:
            role.update(status="FAILED", reason="ARM_PLANNING_FAILED")
            plans.append(None)
    _require(all(plan is not None for plan in plans), "ARM_PLANNING_FAILED")
    for plan, prior, role in zip(plans, priors, summary["roles"]):
        batch = plan.intents._validated_copy()
        _require(plan.result_digest == role["planning_result_digest"]
                 and batch.digest == role["intent_batch_digest"]
                 and (batch.portfolio_id, batch.cycle_id, batch.trade_date)
                 == (prior.portfolio_id, anchor, trade), "FINAL_PAIR_JOIN_INVALID")
    return tuple(plans)


def prepare_first_forward_intent(
    production_workspace_root, research_workspace_root, engine_root,
    qlib_provider_root, current_cycle_id, activation_path,
    definition_store_root, evidence_store_root, bootstrap_store_root, bootstrap_set_id,
    signal_capsule_store_root, signal_capsule_id, model_capsule_store_root, model_capsule_id,
) -> FirstForwardIntentPreparation:
    summary = _summary()
    guards = []
    stage = "INPUT_INVALID"
    try:
        anchor = surface._date(current_cycle_id, "current_cycle_id")
        summary["current_cycle_id"] = anchor
        values = (production_workspace_root, research_workspace_root, engine_root, qlib_provider_root,
                  activation_path, definition_store_root, evidence_store_root, bootstrap_store_root,
                  signal_capsule_store_root, model_capsule_store_root)
        paths = tuple(surface._physical_path(value, "input", directory=index != 4) for index, value in enumerate(values))
        production, research, engine, provider, activation, definitions, evidence_root, bootstraps, signals, models = paths
        for identifier in (bootstrap_set_id, signal_capsule_id, model_capsule_id):
            surface._identifier(identifier, "selector")
        selected = (activation, definitions / activation.stem, evidence_root / activation.stem,
                    bootstraps / bootstrap_set_id, signals / signal_capsule_id, models / model_capsule_id,
                    production / "config" / "strategy_config.yaml",
                    provider / "calendars" / "day.txt", provider / "calendars" / "day_future.txt")
        # Cycle guards begin before activation/bootstrap selectors are read.
        for root, logical in ((production, ("data/evidence/v1/cycles", "config/strategy_config.yaml")),
                              (research, tuple(path.relative_to(research).as_posix() for path in selected[:6])),
                              (engine, ("quantpits", ".git/HEAD", ".git/refs", ".git/index")),
                              (provider, ("calendars",))):
            guard = surface.SourceMutationObserver(root, logical)
            guards.append(guard)
            _require(guard.supported, "INPUT_OBSERVATION_UNSUPPORTED")
        before = _metadata(selected)
        activation_raw = _read_json(activation)
        evidence_cycle = surface._date(activation_raw["evidence_cycle_id"], "evidence_cycle")
        stage = "SURFACE_INCOMPARABLE"
        observed = surface.observe_production_decision_surface(
            research, production, engine, anchor, activation, definitions, evidence_root,
            bootstraps, bootstrap_set_id, reference_source="production",
        )
        if observed.status == "VERSION_BREAK":
            summary.update(status="VERSION_BREAK", reason_codes=["VERSION_BREAK"])
            return _result(summary)
        _require(observed.same_champion_segment, "SURFACE_INCOMPARABLE")
        stage = "DEFINITION_ADOPTION_FAILED"
        from quantpits.research.forward_definition_evidence import adopt_fresh_champion_segment_definition_evidence
        from quantpits.research.forward_observation import observe_fresh_champion_segment_candidate
        evidence = adopt_fresh_champion_segment_definition_evidence(production, research, evidence_cycle, activation, definitions, evidence_root)
        _require(evidence.status == "ADOPTED" and evidence.definition_evidence_complete
                 and not evidence.evidence_receipt.did_write, "DEFINITION_ADOPTION_FAILED")
        candidate = observe_fresh_champion_segment_candidate(production, research, evidence_cycle, activation)
        compiled = candidate.compiled_definitions
        request = compiled.to_store_request()
        stage = "BOOTSTRAP_INVALID"
        source_cycle, priors, bootstrap = _bootstrap(production, bootstraps, bootstrap_set_id, candidate, evidence, anchor)
        summary["bootstrap_source_cycle_id"] = source_cycle
        stage = "CURRENT_CYCLE_INVALID"
        cycle_path, manifest, seal = surface._cycle_authority(production, anchor)
        _require(surface._source_matches_definition(surface._source_projection(manifest), candidate), "SIGNAL_DEFINITION_JOIN_INVALID")
        stage = "RUNTIME_CODE_MISMATCH"
        commit, tree, implementation = _engine(engine, manifest)
        summary.update(engine_commit=commit, engine_tree=tree, implementation_digest=implementation)
        stage = "MODEL_ADOPTION_FAILED"
        from quantpits.research.model_artifact_capsule import adopt_definition_bound_model_artifact_capsule
        model = adopt_definition_bound_model_artifact_capsule(production, research, evidence_cycle, activation, definitions, evidence_root, models, model_capsule_id)
        _require(model.status == "ADOPTED" and model.model_artifact_retention_complete and not model.did_write, stage)
        stage = "SIGNAL_ADOPTION_FAILED"
        from quantpits.research.signal_input_capsule import adopt_signal_input_capsule, MEMBER_NAMES
        signal = adopt_signal_input_capsule(production, research, anchor, signals, signal_capsule_id)
        _require(signal.status == "ADOPTED" and signal.critical_signal_retention_complete and not signal.did_write, stage)
        capsule_manifest = _read_json(signals / signal_capsule_id / "capsule_manifest.json")
        expected = {row["logical_path"]: row["raw_digest"] for row in capsule_manifest["members"]}
        signal_bytes = tuple(replay._stable_read(signals / signal_capsule_id, name, expected[name]) for name in MEMBER_NAMES)
        stage = "RANKING_INPUT_INVALID"
        sealed_bytes = replay._stable_read(cycle_path, "ranking.csv", seal["named_file_digests"]["ranking.csv"])
        champion = replay._ranking_from_csv(sealed_bytes)
        summary["sealed_ranking_digest"] = _raw(sealed_bytes)
        summary["roles"][0]["coverage_counts"] = {"eligible": champion.eligible_count, "scored": champion.scored_count, "missing": champion.missing_count}
        _require(champion.complete and champion.missing_count == 0 and champion.scored_count > 0, "SEALED_RANKING_INCOMPLETE")
        material = manifest["data_identity"]["qlib_materialization_identity"]
        market = surface._market_projection(manifest)["market"]
        universe_path = "instruments/%s.txt" % material["universe_name"]
        universe_guard = surface.SourceMutationObserver(provider, (universe_path,))
        guards.append(universe_guard)
        _require(universe_guard.supported, "INPUT_OBSERVATION_UNSUPPORTED")
        universe_data = replay._stable_read(provider, universe_path, material["universe_digest"])
        universe = replay._universe_at(replay._universe_intervals(universe_data), anchor)
        _require(set(universe) == {row["instrument"] for row in champion.rows}, "UNIVERSE_JOIN_INVALID")
        members = tuple(row["source_id"] for row in compiled.champion.to_dict()["source_members"])
        subset = tuple(row["source_id"] for row in compiled.challenger.to_dict()["source_members"])
        _require(len(members) == 4 and len(subset) == 3 and tuple(name for name in members if name in subset) == subset, "MEMBER_ORDER_INVALID")
        inputs = dict(zip(members, signal_bytes[:4]))
        rebuilt = replay.rank_complete_anchor(prediction_bytes_by_member=inputs, member_order=members, anchor_date=anchor, eligible_instruments=universe)
        summary["recomputed_ranking_digest"] = _raw(rebuilt.to_csv_bytes())
        _parity(champion, rebuilt)
        challenger = replay.rank_complete_anchor(prediction_bytes_by_member=inputs, member_order=subset, anchor_date=anchor, eligible_instruments=universe)
        rankings = (champion, challenger)
        stage = "CALENDAR_INVALID"
        day_data = replay._stable_read(provider, "calendars/day.txt")
        future_data = replay._stable_read(provider, "calendars/day_future.txt")
        day, future = _calendar(day_data), _calendar(future_data)
        _require(anchor in day and anchor in future and tuple(d for d in day if d <= anchor) == tuple(d for d in future if d <= anchor), stage)
        sessions = tuple(d for d in future if d > anchor)
        _require(bool(sessions), "NEXT_SESSION_UNAVAILABLE")
        trade = sessions[0]
        summary.update(trade_date=trade, calendar_match_status="MATCH" if surface._digest(day_data, "raw_bytes") == material["calendar_digest"] else "DIFFERENT",
                       price_provenance="CURRENT_PROVIDER_ANCHOR_OBSERVATION")
        requested = tuple(sorted(set(universe) | {position.instrument for prior in priors for position in prior.positions}))
        price_paths = tuple(provider / "features" / instrument.lower() / (field + ".day.bin") for instrument in requested for field in ("close", "factor"))
        price_selected = price_paths + (provider / universe_path,)
        price_before = _metadata(price_selected)
        guard = surface.SourceMutationObserver(provider, tuple(path.relative_to(provider).as_posix() for path in price_selected))
        guards.append(guard)
        _require(guard.supported, "INPUT_OBSERVATION_UNSUPPORTED")
        stage = "PRICE_OBSERVATION_FAILED"
        receipt = QlibCashPriceObserver(provider, day).observe_anchor_close(anchor_date=anchor, requested_instruments=requested)
        receipt._validated_copy()
        _require(receipt.observation_kind == "CASH_CLOSE" and receipt.observation_date == anchor
                 and receipt.requested_instruments == requested
                 and receipt.calendar_position == day.index(anchor), "PRICE_RECEIPT_JOIN_INVALID")
        summary["price_counts"] = receipt.to_dict()["counts"]
        snapshots = tuple(AnchorPriceSnapshot.from_iterable(
            anchor_date=anchor, requested_instruments=tuple(sorted(set(universe) | {p.instrument for p in prior.positions})),
            rows=[{"instrument": row["instrument"], "anchor_date": anchor,
                   "status": "OBSERVED" if row["status"] == "OBSERVED" else "MISSING",
                   "cash_close": row["cash_price"] if row["status"] == "OBSERVED" else None}
                  for row in receipt.rows if row["instrument"] in set(universe) | {p.instrument for p in prior.positions}],
        ) for prior in priors)
        definition = CurrentRuleIntentDefinition.from_dict(compiled.protocol.to_dict()["intent_definition"])
        for role, ranking, prior, snapshot in zip(summary["roles"], rankings, priors, snapshots):
            role.update(coverage_counts={"eligible": ranking.eligible_count, "scored": ranking.scored_count, "missing": ranking.missing_count},
                        ranking_digest=_raw(ranking.to_csv_bytes()), prior_digest=prior.digest, price_digest=snapshot.digest)
        input_payload = {"selectors": {
                             "definition_set_id": candidate.definition_set_id,
                             "definition_evidence_cycle_id": evidence_cycle,
                             "bootstrap_set_id": bootstrap_set_id, "bootstrap_source_cycle_id": source_cycle,
                             "current_cycle_id": anchor, "signal_capsule_id": signal_capsule_id,
                             "model_capsule_id": model_capsule_id,
                         }, "definition": dict(request.request_digest), "bootstrap": surface._digest(bootstrap),
                         "cycle_seal": surface._digest(seal), "signal": dict(signal.manifest_digest),
                         "model": dict(model.manifest_digest), "rankings": [r["ranking_digest"] for r in summary["roles"]],
                         "calendar": _raw(day_data), "future_calendar": _raw(future_data), "price": dict(receipt.digest),
                         "implementation": implementation, "engine_commit": commit, "engine_tree": tree,
                         "execution_assumption": compiled.execution_assumption.to_dict()}
        stage = "ARM_PLANNING_FAILED"
        plans = _plan_pair(priors, rankings, snapshots, definition, anchor, trade, market, summary)
        stage = "INPUT_STABILITY_LOST"
        _require(before == _metadata(selected) and price_before == _metadata(price_selected)
                 and not any(guard.mutated() for guard in guards), stage)
        _require(_engine(engine, manifest) == (commit, tree, implementation), stage)
        # Close observations before granting a result; cleanup failure is not PREPARED.
        while guards:
            guard = guards[-1]
            _require(not guard.mutated(), stage)
            guard.close()
            guards.pop()
        stage = "PAIR_FINALIZATION_FAILED"
        top = [set(replay._top(ranking, definition.topk)) for ranking in rankings]
        buys = [set(plan.selected_buy_instruments) for plan in plans]
        sells = [{row["instrument"] for row in plan.production_sell_proposal} for plan in plans]
        summary["comparison"] = {"top_k": definition.topk, "champion_top_count": len(top[0]), "challenger_top_count": len(top[1]),
                                 "top_overlap_count": len(top[0] & top[1]), "buy_overlap_count": len(buys[0] & buys[1]), "sell_overlap_count": len(sells[0] & sells[1])}
        count = champion.scored_count
        if count >= 2:
            ranks = [{row["instrument"]: row["rank"] for row in ranking.rows} for ranking in rankings]
            squared = sum((ranks[0][key] - ranks[1][key]) ** 2 for key in ranks[0])
            summary["comparison"].update(spearman=1.0 - 6.0 * squared / (count * (count ** 2 - 1)), spearman_reason=None)
        else:
            summary["comparison"].update(spearman=None, spearman_reason="INSUFFICIENT_MEMBERS")
        summary["input_digest"] = _hash(input_payload)
        summary["preparation_digest"] = _hash({"input_digest": summary["input_digest"], "roles": summary["roles"]})
        summary.update(status="PREPARED", intent_pair_prepared=True)
        return _result(summary, _PreparedPair(
            priors, rankings, snapshots, plans, compiled, receipt,
            canonical_json_bytes(input_payload), (day_data, future_data), canonical_json_bytes(bootstrap),
        ))
    except _PROCESS_CONTROL:
        raise
    except Exception as exc:
        summary.update(status="PRECONDITION_BLOCKED", intent_pair_prepared=False,
                       input_digest=None, preparation_digest=None,
                       reason_codes=[exc.code if isinstance(exc, _Blocked) else stage])
        return _result(summary)
    finally:
        active = sys.exc_info()[1]
        close_failed = False
        for guard in reversed(guards):
            try:
                guard.close()
            except _PROCESS_CONTROL:
                if not isinstance(active, _PROCESS_CONTROL):
                    raise
            except Exception:
                close_failed = True
        if close_failed and active is None:
            summary.update(status="PRECONDITION_BLOCKED", intent_pair_prepared=False,
                           input_digest=None, preparation_digest=None,
                           reason_codes=["INPUT_OBSERVER_CLOSE_FAILED"])
            return _result(summary)
