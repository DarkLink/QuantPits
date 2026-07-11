"""Build deterministic legacy-compatible state output payloads."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import pandas as pd

from quantpits.post_trade.state import PostTradeStateChangeSet


@dataclass(frozen=True)
class StateOutputPayloads:
    trade_details: Tuple[Tuple[str, bytes], ...]
    trade_log: bytes
    holding_log: bytes
    daily_log: bytes
    prod_config: bytes


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"证券代码": str}) if path.exists() else pd.DataFrame()


def _replace_dates(existing, current, dates, column="成交日期"):
    if not existing.empty and column in existing.columns:
        existing = existing.loc[~existing[column].astype(str).isin(dates)]
    combined = pd.concat([existing, current], ignore_index=True) if not current.empty else existing
    sort = [name for name in (column, "证券代码", "交易类别") if name in combined.columns]
    return combined.sort_values(sort, kind="stable").reset_index(drop=True) if sort else combined.reset_index(drop=True)


def build_state_output_payloads(ctx, change_set: PostTradeStateChangeSet, settlement_frames: Mapping[str, pd.DataFrame], *, model="GATs") -> StateOutputPayloads:
    dates = set(change_set.processed_dates)
    detail_pairs, trade_frames, holding_rows, daily_rows = [], [], [], []
    transition_map = {item.trade_date: item for item in change_set.transitions}
    for date in change_set.processed_dates:
        detail = settlement_frames.get(date, pd.DataFrame()).copy()
        if not detail.empty:
            if "证券代码" in detail:
                from quantpits.post_trade.state import normalize_instrument
                detail["证券代码"] = detail["证券代码"].map(normalize_instrument)
            detail["成交日期"] = date; detail["model"] = model
            detail_pairs.append((date, _csv_bytes(detail))); trade_frames.append(detail)
        transition = transition_map[date]
        closes = transition.valuation.close_map()
        total_cost = transition.after.cash; total_value = transition.after.cash
        for position in transition.after.positions:
            close = closes[position.instrument]
            value = position.quantity * close
            pnl = value - position.cost
            holding_rows.append({"成交日期": date, "证券代码": position.instrument, "持仓数量": float(position.quantity), "持仓成本": float(position.cost), "持仓均价": float(position.average_cost), "收盘价格": float(close), "收盘价值": float(value), "当前浮盈": float(pnl), "浮盈收益率": float(pnl / position.cost) if position.cost else 0})
            total_cost += position.cost; total_value += value
        holding_rows.append({"成交日期": date, "证券代码": "CASH", "持仓数量": float(transition.after.cash), "持仓成本": float(transition.after.cash), "持仓均价": 1, "收盘价格": 1, "收盘价值": float(transition.after.cash), "当前浮盈": 0, "浮盈收益率": 0})
        pnl = total_value - total_cost
        daily_rows.append({"成交日期": date, "持仓成本": float(total_cost), "收盘价值": float(total_value), "当前浮盈": float(pnl), "当前浮盈率": float(pnl / total_cost) if total_cost else 0, "CSI300": str(transition.valuation.benchmark), "CASHFLOW": str(float(transition.external_cashflow))})
    trade_current = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    holding_current, daily_current = pd.DataFrame(holding_rows), pd.DataFrame(daily_rows)
    trade_log = _replace_dates(_read_csv(ctx.data_path("trade_log_full.csv")), trade_current, dates)
    holding_log = _replace_dates(_read_csv(ctx.data_path("holding_log_full.csv")), holding_current, dates)
    daily_log = _replace_dates(_read_csv(ctx.data_path("daily_amount_log_full.csv")), daily_current, dates)
    return StateOutputPayloads(tuple(detail_pairs), _csv_bytes(trade_log), _csv_bytes(holding_log), _csv_bytes(daily_log), json.dumps(dict(change_set.next_prod_config), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
