"""Pure cashflow consumption for post-trade state commits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence, Tuple

from quantpits.post_trade.contracts import PostTradeCashflowError, PostTradeCashflowConflictError


@dataclass(frozen=True)
class CashflowCommit:
    consumed_dates: Tuple[str, ...]
    next_config: Mapping[str, object]
    changed: bool

    def to_bytes(self) -> bytes:
        return (json.dumps(dict(self.next_config), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise PostTradeCashflowError("Invalid cashflow date: %s" % value) from exc
    return value


def build_cashflow_commit(raw_cashflow_config: Mapping[str, object], processed_dates: Sequence[str]) -> CashflowCommit:
    if not isinstance(raw_cashflow_config, Mapping):
        raise PostTradeCashflowError("cashflow config must be an object")
    dates = tuple(dict.fromkeys(_date(str(value)) for value in processed_dates))
    result = dict(raw_cashflow_config)
    if "cashflows" in result or "processed" in result:
        active = result.get("cashflows", {})
        processed = result.get("processed", {})
        if not isinstance(active, Mapping) or not isinstance(processed, Mapping):
            raise PostTradeCashflowError("cashflows and processed must be objects")
        active_next, processed_next = dict(active), dict(processed)
        for key in tuple(active_next):
            _date(str(key))
        for key in tuple(processed_next):
            _date(str(key))
        consumed = []
        for date in dates:
            if date not in active_next:
                continue
            value = active_next[date]
            if date in processed_next and processed_next[date] != value:
                raise PostTradeCashflowConflictError("Processed cashflow conflicts for %s" % date)
            processed_next[date] = value
            del active_next[date]
            consumed.append(date)
        result["cashflows"] = active_next
        result["processed"] = processed_next
        return CashflowCommit(tuple(consumed), result, result != dict(raw_cashflow_config))
    if "cash_flow_today" in result and dates:
        value = result.get("cash_flow_today")
        if value not in (0, 0.0, "0", None):
            result["cash_flow_today"] = 0
            return CashflowCommit((dates[0],), result, True)
    return CashflowCommit((), result, False)
