"""Pure, read-only sequential historical shadow state-chain replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

from quantpits.research.accounting import ShadowPortfolioState
from quantpits.research.historical_cycle import (
    ARM_IDS,
    EVIDENCE_CLASS,
    WARNING_LINES,
    HistoricalCycleContractError,
    HistoricalCycleInputError,
    HistoricalShadowCycleReplay,
    ReplayProfileReceipt,
    _canonical_json,
    _decimal_text,
    _deterministic_id,
    _digest_payload,
    _environment_fingerprint,
    _source_fingerprint,
)
from quantpits.research.replay import (
    ReplayResult,
    SealedReplayInputs,
    revalidate_replay_result,
)


WINDOW_WARNINGS = WARNING_LINES + (
    "Sequential historical state chain; hindsight prices already exist.",
    "Not a prospective Challenger observation.",
    "Missing valuation is never forward-filled or treated as zero.",
)
_WINDOW_AUTHORITY = object()


class HistoricalWindowContractError(ValueError):
    """A representation violates the sequential-window contract."""


class HistoricalWindowInputError(RuntimeError):
    """A read-only source could not be observed continuously."""


def _checked_state(value: Any, field: str) -> ShadowPortfolioState:
    if type(value) is not ShadowPortfolioState:
        raise HistoricalWindowContractError("%s must be a canonical portfolio state" % field)
    try:
        copied = value._validated_copy()
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        raise HistoricalWindowContractError("%s failed canonical revalidation" % field) from exc
    if copied.digest != value.digest:
        raise HistoricalWindowContractError("%s digest changed during revalidation" % field)
    return copied


@dataclass(frozen=True, init=False)
class SequentialPriorStateSet:
    """Inspector-owned exact five-arm prior-state authority."""

    window_id: str
    source: str
    source_cycle_id: Optional[str]
    as_of_boundary: str
    states: Mapping[str, ShadowPortfolioState]
    aggregate_digest: Mapping[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop("_authority", None) is not _WINDOW_AUTHORITY or args:
            raise HistoricalWindowContractError("sequential prior-state sets are replay-owned")
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    @classmethod
    def _create(
        cls, authority: object = None, *, window_id: str, source: str,
        source_cycle_id: Optional[str],
        as_of_boundary: str, states: Mapping[str, ShadowPortfolioState],
    ) -> "SequentialPriorStateSet":
        if authority is not _WINDOW_AUTHORITY:
            raise HistoricalWindowContractError("sequential prior-state sets are replay-owned")
        if source not in ("INITIAL_BOOTSTRAP", "PRIOR_CYCLE_AFTER_STATE"):
            raise HistoricalWindowContractError("prior-state source is invalid")
        if (source == "INITIAL_BOOTSTRAP") != (source_cycle_id is None):
            raise HistoricalWindowContractError("prior-state source cycle is inconsistent")
        if not isinstance(states, Mapping) or tuple(states) != ARM_IDS:
            raise HistoricalWindowContractError("prior-state set must contain exact ordered five arms")
        copied = {arm: _checked_state(states[arm], "states.%s" % arm) for arm in ARM_IDS}
        if len({item.portfolio_id for item in copied.values()}) != len(ARM_IDS):
            raise HistoricalWindowContractError("prior-state portfolio identities must be distinct")
        if any(item.as_of_date != as_of_boundary for item in copied.values()):
            raise HistoricalWindowContractError("prior-state dates must equal the set boundary")
        payload = {
            "window_id": window_id, "source": source,
            "source_cycle_id": source_cycle_id, "as_of_boundary": as_of_boundary,
            "members": [
                {"arm_id": arm, "portfolio_id": copied[arm].portfolio_id,
                 "state_digest": copied[arm].digest}
                for arm in ARM_IDS
            ],
        }
        return cls(
            _authority=_WINDOW_AUTHORITY, window_id=window_id, source=source,
            source_cycle_id=source_cycle_id, as_of_boundary=as_of_boundary,
            states=MappingProxyType(copied),
            aggregate_digest=MappingProxyType(_digest_payload(payload)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id, "source": self.source,
            "source_cycle_id": self.source_cycle_id,
            "as_of_boundary": self.as_of_boundary,
            "members": [
                {"arm_id": arm, "portfolio_id": self.states[arm].portfolio_id,
                 "state_digest": self.states[arm].digest}
                for arm in ARM_IDS
            ],
            "aggregate_digest": dict(self.aggregate_digest),
        }


class HistoricalWindowResult(Mapping[str, Any]):
    """Immutable truth-owner envelope for one exact sequential window."""

    def __init__(self, payload: Mapping[str, Any], authority: object = None) -> None:
        if authority is not _WINDOW_AUTHORITY:
            raise HistoricalWindowContractError("historical-window results are replay-owned")
        fields = {
            "schema_version", "status", "evidence_class", "warnings",
            "prospective_claim", "promotion_capability", "window_identity",
            "stage_a_result_digest", "source_models", "universe_identity",
            "calendar_identity", "source_to_materialization_relation", "profile",
            "intent_definition_digest", "execution_assumption_digest",
            "engine_source_fingerprint", "environment_fingerprint",
            "requested_anchor_ids", "requested_arm_ids", "terminal_cycles",
            "counts", "chain_continuity_checked", "future_price_order_checked",
            "common_valuation_window_status", "arm_metrics", "champion_comparisons",
            "metric_capability", "result_digest",
        }
        if set(payload) != fields:
            raise HistoricalWindowContractError("historical-window result fields are not exact")
        anchors = tuple(payload["requested_anchor_ids"])
        cycles = payload["terminal_cycles"]
        if len(anchors) not in (4, 5, 6) or tuple(payload["requested_arm_ids"]) != ARM_IDS:
            raise HistoricalWindowContractError("historical-window requested sets are invalid")
        if len(cycles) != len(anchors) or tuple(item.get("anchor_date") for item in cycles) != anchors:
            raise HistoricalWindowContractError("historical-window terminal cycle identity is invalid")
        prior_portfolios: Dict[str, str] = {}
        prior_after: Dict[str, Mapping[str, Any]] = {}
        observed_cycle_ids = set()
        initial_economics = set()
        for index, cycle in enumerate(cycles):
            arms = cycle.get("terminal_arms", ())
            links = cycle.get("chain_links", ())
            if (
                cycle.get("cycle_index") != index
                or tuple(item.get("arm_id") for item in arms) != ARM_IDS
                or tuple(item.get("arm_id") for item in links) != ARM_IDS
            ):
                raise HistoricalWindowContractError("historical-window cycle grid identity is invalid")
            cell_complete = all(item.get("terminal_status") == "COMPLETE" for item in arms)
            links_complete = all(
                item.get("digest_equal_checked") is True
                and item.get("portfolio_id_equal_checked") is True
                and item.get("date_monotonic_checked") is True
                for item in links
            )
            should_cycle_complete = cell_complete and links_complete and len(cycle.get("comparisons", ())) == 4
            if cycle.get("cycle_status") != ("COMPLETE" if should_cycle_complete else "BLOCKED"):
                raise HistoricalWindowContractError("historical-window cycle capability is inconsistent")
            if index + 1 < len(cycles) and not cycle.get("trade_date", "") < anchors[index + 1]:
                raise HistoricalWindowContractError("historical-window cycle chronology is inconsistent")
            if not cell_complete:
                continue
            for arm, item, link in zip(ARM_IDS, arms, links):
                prior = item.get("prior_state")
                after = item.get("after_state")
                transition = item.get("transition_result")
                if not isinstance(prior, Mapping) or not isinstance(after, Mapping) or not isinstance(transition, Mapping):
                    raise HistoricalWindowContractError("complete window arm lacks state/transition authority")
                portfolio = prior.get("portfolio_id")
                cycle_id = transition.get("cycle_id")
                if (
                    not isinstance(portfolio, str) or not portfolio
                    or after.get("portfolio_id") != portfolio
                    or transition.get("portfolio_id") != portfolio
                    or not isinstance(cycle_id, str) or not cycle_id
                    or cycle_id in observed_cycle_ids
                ):
                    raise HistoricalWindowContractError("portfolio/cycle identity is inconsistent")
                observed_cycle_ids.add(cycle_id)
                if arm in prior_portfolios and prior_portfolios[arm] != portfolio:
                    raise HistoricalWindowContractError("portfolio identity changed across cycles")
                prior_portfolios[arm] = portfolio
                if index == 0:
                    initial_economics.add(_canonical_json({
                        "as_of_date": prior.get("as_of_date"), "cash": prior.get("cash"),
                        "positions": prior.get("positions"),
                    }))
                    if link.get("source") != "INITIAL_BOOTSTRAP" or link.get("predecessor_cycle_id") is not None:
                        raise HistoricalWindowContractError("initial chain link is inconsistent")
                else:
                    predecessor = prior_after.get(arm)
                    if (
                        predecessor is None or predecessor.get("digest") != prior.get("digest")
                        or predecessor.get("portfolio_id") != portfolio
                        or link.get("source") != "PRIOR_CYCLE_AFTER_STATE"
                        or link.get("prior_state_digest") != prior.get("digest")
                        or link.get("predecessor_after_state_digest") != predecessor.get("digest")
                    ):
                        raise HistoricalWindowContractError("serialized state chain is inconsistent")
                prior_after[arm] = after
        if observed_cycle_ids and (len(initial_economics) != 1 or len(set(prior_portfolios.values())) != 5):
            raise HistoricalWindowContractError("initial economics/portfolio independence is inconsistent")
        terminal = sum(len(item.get("terminal_arms", ())) for item in cycles)
        complete = sum(
            arm.get("terminal_status") == "COMPLETE"
            for cycle in cycles for arm in cycle.get("terminal_arms", ())
        )
        counts = {
            "requested_cycles": len(anchors), "terminal_cycles": len(cycles),
            "requested_cycle_arms": len(anchors) * len(ARM_IDS),
            "terminal_cycle_arms": terminal, "complete_cycle_arms": complete,
            "blocked_cycle_arms": terminal - complete,
        }
        if terminal != len(anchors) * len(ARM_IDS) or payload["counts"] != counts:
            raise HistoricalWindowContractError("historical-window grid/counts are inconsistent")
        complete_window = (
            complete == terminal
            and payload["chain_continuity_checked"] is True
            and all(item.get("cycle_status") == "COMPLETE" for item in cycles)
        )
        if (
            payload["schema_version"] != 1 or payload["evidence_class"] != EVIDENCE_CLASS
            or tuple(payload["warnings"]) != WINDOW_WARNINGS
            or payload["prospective_claim"] is not False
            or payload["promotion_capability"] is not False
            or payload["status"] != ("COMPLETE" if complete_window else "BLOCKED")
            or payload["future_price_order_checked"] is not True
        ):
            raise HistoricalWindowContractError("historical-window authority fields are inconsistent")
        metrics = payload["metric_capability"] is True
        if metrics != (complete_window and payload["common_valuation_window_status"] == "COMPLETE"):
            raise HistoricalWindowContractError("historical-window metric capability is inconsistent")
        if metrics:
            if tuple(item.get("arm_id") for item in payload["arm_metrics"]) != ARM_IDS:
                raise HistoricalWindowContractError("historical-window metrics are not exact")
            if len(payload["champion_comparisons"]) != 4:
                raise HistoricalWindowContractError("historical-window comparisons are not exact")
        elif payload["arm_metrics"] or payload["champion_comparisons"]:
            raise HistoricalWindowContractError("blocked metrics must not expose comparison capability")
        without = {key: value for key, value in payload.items() if key != "result_digest"}
        if payload["result_digest"] != _digest_payload(without):
            raise HistoricalWindowContractError("historical-window result digest is inconsistent")
        self._bytes = _canonical_json(payload)
        self._payload = json.loads(self._bytes.decode("utf-8"))

    def __getitem__(self, key: str) -> Any:
        return json.loads(json.dumps(self._payload[key]))

    def __iter__(self):
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(self._bytes.decode("utf-8"))

    def to_canonical_json_bytes(self) -> bytes:
        return self._bytes


def revalidate_historical_window_result(
    result: HistoricalWindowResult,
) -> HistoricalWindowResult:
    """Return an independent snapshot after checking the current payload."""
    if not isinstance(result, HistoricalWindowResult):
        raise HistoricalWindowContractError(
            "revalidation requires a replay-owned historical-window result"
        )
    try:
        snapshot = HistoricalWindowResult(result._payload, _WINDOW_AUTHORITY)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception as exc:
        raise HistoricalWindowContractError(
            "current historical-window result failed revalidation"
        ) from exc
    if snapshot.to_canonical_json_bytes() != result.to_canonical_json_bytes():
        raise HistoricalWindowContractError(
            "current historical-window result bytes are inconsistent"
        )
    return snapshot


def _safe_error(exc: BaseException, reason: str) -> Dict[str, str]:
    return {"type": type(exc).__name__, "reason_code": reason}


def _blocked_arm(arm: str, reason: str, exc: Optional[BaseException] = None) -> Dict[str, Any]:
    return {
        "arm_id": arm, "terminal_status": reason,
        "error": ({"type": "WindowBlocked", "reason_code": reason}
                  if exc is None else _safe_error(exc, reason)),
    }


def _sanitize_arm_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    if result.get("error") is not None:
        result["error"] = {
            "type": result["error"].get("type", "Error"),
            "reason_code": result.get("terminal_status", "BLOCKED_CYCLE"),
        }
    return result


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, Decimal("0")) / Decimal(len(values))


class HistoricalShadowWindowReplay:
    """Truth owner for an exact 4--6-cycle, five-arm state chain."""

    def __init__(
        self, *, stage_a_result: ReplayResult, stage_a_inputs: SealedReplayInputs,
        profile: ReplayProfileReceipt, provider_root: Path, market: str,
        window_size: int, engine_root: Optional[Path] = None,
    ) -> None:
        if not isinstance(stage_a_result, ReplayResult) or not isinstance(stage_a_inputs, SealedReplayInputs):
            raise HistoricalWindowContractError("Stage A inputs must be truth-owner contracts")
        if not isinstance(profile, ReplayProfileReceipt):
            raise HistoricalWindowContractError("profile must be an inspector-owned receipt")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size not in (4, 5, 6):
            raise HistoricalWindowContractError("window_size must be exactly 4, 5, or 6")
        try:
            stage_a = revalidate_replay_result(stage_a_result)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            raise HistoricalWindowContractError("Stage A failed result-digest revalidation") from exc
        inventory = stage_a["inventory"]
        anchors = tuple(inventory["selected_anchors"])
        if stage_a["status"] != "complete" or stage_a["parity"].get("status") != "passed":
            raise HistoricalWindowContractError("Stage A is not complete with parity")
        if len(anchors) != window_size or anchors != tuple(sorted(set(anchors))):
            raise HistoricalWindowContractError("Stage A selected anchors do not equal the requested window")
        weekly = tuple(inventory.get("weekly_candidates", ()))
        inventory_rows = tuple(inventory.get("dates", ()))
        if (
            inventory.get("status") != "complete"
            or tuple(item.get("anchor") for item in inventory_rows) != weekly
            or any(item.get("status") != "complete" for item in inventory_rows if item.get("anchor") in anchors)
        ):
            raise HistoricalWindowContractError("Stage A weekly inventory is not comparable")
        try:
            selected_positions = tuple(weekly.index(anchor) for anchor in anchors)
        except ValueError as exc:
            raise HistoricalWindowContractError("selected anchor is foreign to the weekly inventory") from exc
        if selected_positions != tuple(range(selected_positions[0], selected_positions[0] + window_size)):
            raise HistoricalWindowContractError("selected anchors are not one contiguous weekly slice")
        if (
            stage_a["calendar_identity"]["digest"] != dict(stage_a_inputs.calendar_digest)
            or stage_a["universe_identity"]["digest"] != dict(stage_a_inputs.universe_digest)
            or stage_a["source_models"] != [item.public_identity() for item in stage_a_inputs.sources]
        ):
            raise HistoricalWindowContractError("Stage A and sealed inputs have foreign identity")
        dates = stage_a_inputs.calendar_dates
        trade_dates = []
        for index, anchor in enumerate(anchors):
            try:
                position = dates.index(anchor)
            except ValueError as exc:
                raise HistoricalWindowContractError("selected anchor is absent from calendar") from exc
            if position + 1 >= len(dates):
                raise HistoricalWindowContractError("selected anchor has no next session")
            trade = dates[position + 1]
            if index + 1 < len(anchors) and not trade < anchors[index + 1]:
                raise HistoricalWindowContractError("window chronology overlaps or forks")
            trade_dates.append(trade)
            rankings = stage_a["rankings"].get(anchor)
            if not isinstance(rankings, Mapping) or tuple(rankings) != ARM_IDS:
                raise HistoricalWindowContractError("selected anchor lacks exact ordered five rankings")
        root = Path(engine_root) if engine_root is not None else Path(__file__).resolve().parents[2]
        self._engine_root = root.resolve(strict=True)
        self._stage_a = stage_a
        self._stage_a_inputs = stage_a_inputs
        self._profile = profile
        self._provider_root = Path(provider_root)
        self._market = market
        self._anchors = anchors
        self._trade_dates = tuple(trade_dates)
        self._cycle_template = HistoricalShadowCycleReplay(
            stage_a_result=stage_a, stage_a_inputs=stage_a_inputs, profile=profile,
            provider_root=self._provider_root, anchor_date=anchors[0], market=market,
            engine_root=self._engine_root,
        )
        self._observer = self._cycle_template._observer
        self._source = self._cycle_template._source
        self._environment = self._cycle_template._environment
        identity = {
            "stage_a_result_digest": stage_a["result_digest"],
            "profile_digest": dict(profile.canonical_digest),
            "intent_definition_digest": profile.definition.digest,
            "execution_assumption_digest": profile.assumption.digest,
            "requested_anchors": list(anchors), "market": market,
            "universe_identity": stage_a["universe_identity"],
            "calendar_identity": stage_a["calendar_identity"],
            "engine_source_fingerprint": dict(self._source),
        }
        self._window_id = _deterministic_id("B3A_WINDOW_V1", identity)
        self._window_identity = MappingProxyType({**identity, "window_id": self._window_id})
        self._initial_state_set = SequentialPriorStateSet._create(
            _WINDOW_AUTHORITY,
            window_id=self._window_id, source="INITIAL_BOOTSTRAP", source_cycle_id=None,
            as_of_boundary=profile.bootstrap["as_of_date"],
            states=MappingProxyType({arm: self._cycle_template._state(arm) for arm in ARM_IDS}),
        )

    def _cycle(self, anchor: str) -> HistoricalShadowCycleReplay:
        return self._cycle_template._window_step(anchor)

    @staticmethod
    def _chain_links(
        index: int, arms: Sequence[Mapping[str, Any]],
        predecessor: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> Tuple[Dict[str, Any], ...]:
        links = []
        for arm, item in zip(ARM_IDS, arms):
            prior = item.get("prior_state")
            if index == 0:
                initial = None if predecessor is None else predecessor.get(arm, {}).get("after_state")
                comparable = prior is not None and initial is not None
                links.append({
                    "arm_id": arm, "source": "INITIAL_BOOTSTRAP",
                    "prior_state_digest": None if prior is None else prior.digest,
                    "predecessor_cycle_id": None,
                    "predecessor_after_state_digest": None,
                    "digest_equal_checked": bool(comparable and prior.digest == initial.digest),
                    "portfolio_id_equal_checked": bool(
                        comparable and prior.portfolio_id == initial.portfolio_id
                    ),
                    "date_monotonic_checked": bool(
                        comparable and prior.as_of_date == initial.as_of_date
                    ),
                })
                continue
            previous = None if predecessor is None else predecessor.get(arm)
            after = None if previous is None else previous.get("after_state")
            transition = None if previous is None else previous.get("transition_result")
            comparable = prior is not None and after is not None and transition is not None
            links.append({
                "arm_id": arm, "source": "PRIOR_CYCLE_AFTER_STATE",
                "prior_state_digest": None if prior is None else prior.digest,
                "predecessor_cycle_id": None if transition is None else transition.cycle_id,
                "predecessor_after_state_digest": None if after is None else after.digest,
                "digest_equal_checked": bool(comparable and prior.digest == after.digest),
                "portfolio_id_equal_checked": bool(comparable and prior.portfolio_id == after.portfolio_id),
                "date_monotonic_checked": bool(comparable and after.as_of_date == prior.as_of_date),
            })
        return tuple(links)

    @staticmethod
    def _metrics(cycles: Sequence[Mapping[str, Any]]) -> Tuple[Tuple[Dict[str, Any], ...], Tuple[Dict[str, Any], ...]]:
        by_arm: Dict[str, List[Mapping[str, Decimal]]] = {arm: [] for arm in ARM_IDS}
        overlaps: Dict[str, List[Decimal]] = {arm: [] for arm in ARM_IDS[1:]}
        for cycle in cycles:
            raw_arms = cycle["_raw_arms"]
            champion_holdings = {
                (item.instrument, item.quantity)
                for item in raw_arms[0]["after_state"].positions
            }
            for arm, item in zip(ARM_IDS, raw_arms):
                transition = item["transition_result"]
                gross = sum((terminal.gross for terminal in transition.terminal_results
                             if terminal.status == "FILLED"), Decimal("0"))
                by_arm[arm].append({
                    "pre": transition.nav_before, "post": transition.nav_after,
                    "turnover": gross / transition.nav_before,
                    "cost": transition.total_fee + transition.slippage_cost,
                })
                if arm != "CHAMPION_4":
                    holdings = {(position.instrument, position.quantity)
                                for position in item["after_state"].positions}
                    union = champion_holdings | holdings
                    overlaps[arm].append(Decimal(len(champion_holdings & holdings)) / Decimal(len(union))
                                         if union else Decimal("1"))
        metrics = []
        normalized_by_arm: Dict[str, Tuple[Decimal, ...]] = {}
        returns_by_arm: Dict[str, Tuple[Decimal, ...]] = {}
        for arm in ARM_IDS:
            rows = by_arm[arm]
            base = rows[0]["pre"]
            normalized = tuple(row["post"] / base for row in rows)
            returns = tuple(
                row["post"] / (row["pre"] if index == 0 else rows[index - 1]["post"]) - Decimal("1")
                for index, row in enumerate(rows)
            )
            peak = Decimal("0")
            drawdowns = []
            for nav in normalized:
                peak = max(peak, nav)
                drawdowns.append(nav / peak - Decimal("1"))
            normalized_by_arm[arm] = normalized
            returns_by_arm[arm] = returns
            metrics.append({
                "arm_id": arm,
                "normalized_nav": [_decimal_text(item) for item in normalized],
                "cycle_return": [_decimal_text(item) for item in returns],
                "drawdown": [_decimal_text(item) for item in drawdowns],
                "turnover": [_decimal_text(row["turnover"]) for row in rows],
                "standardized_cost": [_decimal_text(row["cost"]) for row in rows],
                "terminal_normalized_nav": _decimal_text(normalized[-1]),
                "max_drawdown": _decimal_text(min(drawdowns)),
                "cumulative_cost_rate": _decimal_text(sum((row["cost"] for row in rows), Decimal("0")) / base),
                "mean_holding_overlap_ratio_with_champion": (
                    None if arm == "CHAMPION_4" else _decimal_text(_mean(overlaps[arm]))
                ),
            })
        comparisons = []
        champion_terminal = normalized_by_arm["CHAMPION_4"][-1]
        for arm in ARM_IDS[1:]:
            values = overlaps[arm]
            comparisons.append({
                "champion_arm_id": "CHAMPION_4", "challenger_arm_id": arm,
                "holding_overlap_ratio_mean": _decimal_text(_mean(values)),
                "holding_overlap_ratio_min": _decimal_text(min(values)),
                "holding_overlap_ratio_max": _decimal_text(max(values)),
                "cumulative_active_return": _decimal_text(
                    normalized_by_arm[arm][-1] / champion_terminal - Decimal("1")
                ),
                "cycle_active_return": [
                    _decimal_text(left - right)
                    for left, right in zip(returns_by_arm[arm], returns_by_arm["CHAMPION_4"])
                ],
            })
        return tuple(metrics), tuple(comparisons)

    def run(self) -> HistoricalWindowResult:
        preflight_error: Optional[BaseException] = None
        profile = self._profile
        try:
            if _source_fingerprint(self._engine_root) != self._source:
                raise HistoricalWindowInputError("engine source identity changed")
            if _environment_fingerprint() != self._environment:
                raise HistoricalWindowInputError("runtime environment changed")
            profile = self._profile.revalidate()
            if profile.canonical_digest != self._profile.canonical_digest:
                raise HistoricalWindowInputError("private profile content changed")
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except Exception as exc:
            preflight_error = exc
        cycles: List[Dict[str, Any]] = []
        previous_raw: Optional[Mapping[str, Mapping[str, Any]]] = MappingProxyType({
            arm: MappingProxyType({"after_state": self._initial_state_set.states[arm],
                                   "transition_result": None})
            for arm in ARM_IDS
        })
        prior_set: Optional[SequentialPriorStateSet] = None
        blocked = preflight_error is not None
        trace: List[Tuple[int, str]] = []
        for index, (anchor, trade) in enumerate(zip(self._anchors, self._trade_dates)):
            if blocked:
                reason = (
                    "BLOCKED_COMMON_SOURCE" if index == 0 and preflight_error is not None
                    else "BLOCKED_WINDOW_PREDECESSOR_INCOMPLETE"
                )
                cycles.append({
                    "cycle_index": index, "anchor_date": anchor, "trade_date": trade,
                    "cycle_status": "BLOCKED",
                    "terminal_arms": [
                        _blocked_arm(arm, reason, preflight_error if index == 0 else None)
                        for arm in ARM_IDS
                    ],
                    "chain_links": [
                        {"arm_id": arm, "source": (
                            "INITIAL_BOOTSTRAP" if index == 0 else "PRIOR_CYCLE_AFTER_STATE"
                         ),
                         "prior_state_digest": None, "predecessor_cycle_id": None,
                         "predecessor_after_state_digest": None,
                         "digest_equal_checked": False, "portfolio_id_equal_checked": False,
                         "date_monotonic_checked": False}
                        for arm in ARM_IDS
                    ],
                    "comparisons": [], "timeline": [], "_raw_arms": None,
                })
                continue
            try:
                runner = self._cycle(anchor)
                priors = None if prior_set is None else prior_set.states
                requested = runner.requested_anchor_instruments_for(priors)
                close = self._observer.observe_anchor_close(
                    anchor_date=anchor, requested_instruments=requested,
                )
                trace.append((index, "CLOSE"))
                prepared = runner.prepare_arms(close, prior_states=priors)
                trace.append((index, "PLAN"))
                if prepared.status != "COMPLETE":
                    raw_arms = prepared.terminal_arms
                    links = self._chain_links(index, raw_arms, previous_raw)
                    cycles.append({
                        "cycle_index": index, "anchor_date": anchor, "trade_date": trade,
                        "cycle_status": "BLOCKED",
                        "terminal_arms": [_sanitize_arm_payload(runner._arm_payload(item)) for item in raw_arms],
                        "chain_links": list(links), "comparisons": [],
                        "timeline": ["CLOSE", "PLAN"], "_raw_arms": raw_arms,
                    })
                    blocked = True
                    continue
                receipts = {}
                for item in prepared.terminal_arms:
                    requested_open = tuple(sorted(
                        {position.instrument for position in item["prior_state"].positions}
                        | {intent.instrument for intent in item["intent_planning_result"].intents.intents}
                    ))
                    receipts[item["arm_id"]] = self._observer.observe_next_open(
                        trade_date=trade, requested_instruments=requested_open,
                    )
                trace.append((index, "OPEN"))
                raw_arms = runner.settle_arms(prepared, receipts)
                trace.append((index, "SETTLE"))
                links = self._chain_links(index, raw_arms, previous_raw)
                complete = all(item["terminal_status"] == "COMPLETE" for item in raw_arms)
                link_ok = all(
                    item["digest_equal_checked"] and item["portfolio_id_equal_checked"]
                    and item["date_monotonic_checked"] for item in links
                )
                comparisons = tuple(
                    runner._comparison(raw_arms[0], item) for item in raw_arms[1:]
                    if raw_arms[0]["terminal_status"] == item["terminal_status"] == "COMPLETE"
                )
                cycles.append({
                    "cycle_index": index, "anchor_date": anchor, "trade_date": trade,
                    "cycle_status": "COMPLETE" if complete and link_ok else "BLOCKED",
                    "terminal_arms": [_sanitize_arm_payload(runner._arm_payload(item)) for item in raw_arms],
                    "chain_links": list(links),
                    "comparisons": [dict(item) for item in comparisons],
                    "timeline": ["CLOSE", "PLAN", "OPEN", "SETTLE"],
                    "_raw_arms": raw_arms,
                })
                if not complete or not link_ok:
                    blocked = True
                    continue
                states = MappingProxyType({arm: raw_arms[position]["after_state"]
                                           for position, arm in enumerate(ARM_IDS)})
                source_cycle_ids = {item["transition_result"].cycle_id for item in raw_arms}
                prior_set = SequentialPriorStateSet._create(
                    _WINDOW_AUTHORITY,
                    window_id=self._window_id,
                    source="PRIOR_CYCLE_AFTER_STATE",
                    source_cycle_id=_deterministic_id(
                        "B3A_PREDECESSOR_SET_V1", {"cycle_ids": sorted(source_cycle_ids)}
                    ),
                    as_of_boundary=trade, states=states,
                )
                previous_raw = MappingProxyType({arm: raw_arms[position]
                                                 for position, arm in enumerate(ARM_IDS)})
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception as exc:
                cycles.append({
                    "cycle_index": index, "anchor_date": anchor, "trade_date": trade,
                    "cycle_status": "BLOCKED",
                    "terminal_arms": [_blocked_arm(arm, "BLOCKED_COMMON_OBSERVATION", exc) for arm in ARM_IDS],
                    "chain_links": [
                        {"arm_id": arm, "source": "INITIAL_BOOTSTRAP" if index == 0 else "PRIOR_CYCLE_AFTER_STATE",
                         "prior_state_digest": None, "predecessor_cycle_id": None,
                         "predecessor_after_state_digest": None,
                         "digest_equal_checked": False, "portfolio_id_equal_checked": False,
                         "date_monotonic_checked": False}
                        for arm in ARM_IDS
                    ],
                    "comparisons": [], "timeline": [step for cycle_index, step in trace if cycle_index == index],
                    "_raw_arms": None,
                })
                blocked = True
        public_cycles = []
        for cycle in cycles:
            public_cycles.append({key: value for key, value in cycle.items() if key != "_raw_arms"})
        complete_cells = sum(
            arm["terminal_status"] == "COMPLETE"
            for cycle in public_cycles for arm in cycle["terminal_arms"]
        )
        all_complete = complete_cells == len(self._anchors) * len(ARM_IDS)
        chain_ok = all_complete and all(
            link["digest_equal_checked"] and link["portfolio_id_equal_checked"]
            and link["date_monotonic_checked"]
            for cycle in public_cycles for link in cycle["chain_links"]
        )
        valuation_complete = chain_ok and all(
            item["transition_result"].valuation_status == "COMPLETE"
            and item["transition_result"].nav_before is not None
            and item["transition_result"].nav_after is not None
            and item["transition_result"].nav_before > 0
            for cycle in cycles for item in (cycle["_raw_arms"] or ())
        )
        if valuation_complete:
            arm_metrics, comparisons = self._metrics(cycles)
        else:
            arm_metrics, comparisons = (), ()
        counts = {
            "requested_cycles": len(self._anchors), "terminal_cycles": len(public_cycles),
            "requested_cycle_arms": len(self._anchors) * len(ARM_IDS),
            "terminal_cycle_arms": sum(len(item["terminal_arms"]) for item in public_cycles),
            "complete_cycle_arms": complete_cells,
            "blocked_cycle_arms": len(self._anchors) * len(ARM_IDS) - complete_cells,
        }
        payload: Dict[str, Any] = {
            "schema_version": 1, "status": "COMPLETE" if chain_ok else "BLOCKED",
            "evidence_class": EVIDENCE_CLASS, "warnings": list(WINDOW_WARNINGS),
            "prospective_claim": False, "promotion_capability": False,
            "window_identity": dict(self._window_identity),
            "stage_a_result_digest": self._stage_a["result_digest"],
            "source_models": self._stage_a["source_models"],
            "universe_identity": self._stage_a["universe_identity"],
            "calendar_identity": self._stage_a["calendar_identity"],
            "source_to_materialization_relation": self._stage_a["source_to_materialization_relation"],
            "profile": profile.public_payload(),
            "intent_definition_digest": profile.definition.digest,
            "execution_assumption_digest": profile.assumption.digest,
            "engine_source_fingerprint": dict(self._source),
            "environment_fingerprint": dict(self._environment),
            "requested_anchor_ids": list(self._anchors), "requested_arm_ids": list(ARM_IDS),
            "terminal_cycles": public_cycles, "counts": counts,
            "chain_continuity_checked": chain_ok,
            "future_price_order_checked": all(
                cycle["timeline"] in (
                    ["CLOSE", "PLAN", "OPEN", "SETTLE"],
                    ["CLOSE", "PLAN"], ["CLOSE"], [],
                )
                for cycle in public_cycles
            ),
            "common_valuation_window_status": "COMPLETE" if valuation_complete else "NOT_COMPARABLE_MISSING_VALUATION",
            "arm_metrics": [dict(item) for item in arm_metrics],
            "champion_comparisons": [dict(item) for item in comparisons],
            "metric_capability": valuation_complete,
        }
        payload["result_digest"] = _digest_payload(payload)
        return HistoricalWindowResult(payload, _WINDOW_AUTHORITY)


def compact_window_summary(result: HistoricalWindowResult) -> Dict[str, Any]:
    if not isinstance(result, HistoricalWindowResult):
        raise HistoricalWindowContractError("summary requires a truth-owner result")
    metrics = {item["arm_id"]: item for item in result["arm_metrics"]}
    return {
        "status": result["status"].lower(), "evidence_class": result["evidence_class"],
        "warnings": result["warnings"], "prospective_claim": False,
        "promotion_capability": False,
        "window_digest": result["result_digest"],
        "requested_cycle_count": result["counts"]["requested_cycles"],
        "terminal_cycle_count": result["counts"]["terminal_cycles"],
        "complete_cycle_arm_count": result["counts"]["complete_cycle_arms"],
        "blocked_cycle_arm_count": result["counts"]["blocked_cycle_arms"],
        "chain_continuity_checked": result["chain_continuity_checked"],
        "metric_capability": result["metric_capability"],
        "arms": [
            {
                "arm_id": arm,
                "terminal_normalized_nav": metrics.get(arm, {}).get("terminal_normalized_nav"),
                "max_drawdown": metrics.get(arm, {}).get("max_drawdown"),
                "cumulative_cost_rate": metrics.get(arm, {}).get("cumulative_cost_rate"),
                "mean_holding_overlap_ratio_with_champion": metrics.get(arm, {}).get(
                    "mean_holding_overlap_ratio_with_champion"
                ),
            }
            for arm in ARM_IDS
        ],
    }


__all__ = [
    "HistoricalShadowWindowReplay", "HistoricalWindowContractError",
    "HistoricalWindowInputError", "HistoricalWindowResult",
    "SequentialPriorStateSet", "compact_window_summary",
    "revalidate_historical_window_result",
]
