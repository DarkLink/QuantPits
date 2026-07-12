"""Recoverable cursor-last persistence for deterministic state outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from quantpits.post_trade.contracts import PostTradeStateConflictError, PostTradeStatePersistenceError
from quantpits.post_trade.ingestion import _atomic_bytes
from quantpits.post_trade.state_outputs import StateOutputPayloads
from quantpits.utils.workspace import fingerprint_file


@dataclass(frozen=True)
class StateArtifactLedger:
    outputs: Tuple[Path, ...]
    processed_dates: Tuple[str, ...]
    cursor_committed: bool
    transaction_id: str | None = None


def capture_state_fingerprints(ctx):
    names = (("config", "prod_config.json"), ("config", "cashflow.json"), ("data", "trade_log_full.csv"), ("data", "holding_log_full.csv"), ("data", "daily_amount_log_full.csv"))
    return tuple((kind + "/" + name, fingerprint_file(ctx.root / kind / name) if (ctx.root / kind / name).exists() else None) for kind, name in names)


def persist_state_outputs(ctx, payloads: StateOutputPayloads, processed_dates, *, expected_fingerprints=None):
    if not processed_dates:
        return StateArtifactLedger((), (), False)
    if expected_fingerprints is not None and tuple(expected_fingerprints) != capture_state_fingerprints(ctx):
        raise PostTradeStateConflictError("Post-trade state inputs changed before commit")
    committed = []
    try:
        for date, payload in payloads.trade_details:
            path = ctx.data_path("trade_detail_%s.csv" % date); _atomic_bytes(path, payload); committed.append(path)
        for path, payload in ((ctx.data_path("trade_log_full.csv"), payloads.trade_log), (ctx.data_path("holding_log_full.csv"), payloads.holding_log), (ctx.data_path("daily_amount_log_full.csv"), payloads.daily_log)):
            _atomic_bytes(path, payload); committed.append(path)
        cursor = ctx.config_path("prod_config.json"); _atomic_bytes(cursor, payloads.prod_config); committed.append(cursor)
        return StateArtifactLedger(tuple(committed), tuple(processed_dates), True)
    except Exception as exc:
        raise PostTradeStatePersistenceError("Failed to persist post-trade state: %s" % exc, committed_outputs=committed) from exc


def transaction_payloads(ctx, payloads: StateOutputPayloads):
    """Return deterministic transaction targets; prod cursor is always last."""
    values = []
    for index, (date, payload) in enumerate(payloads.trade_details, 10):
        values.append((index, "trade_detail", ctx.data_path("trade_detail_%s.csv" % date), payload))
    values.extend([
        (100, "trade_log", ctx.data_path("trade_log_full.csv"), payloads.trade_log),
        (200, "holding_log", ctx.data_path("holding_log_full.csv"), payloads.holding_log),
        (300, "daily_log", ctx.data_path("daily_amount_log_full.csv"), payloads.daily_log),
        (400, "cashflow_config", ctx.config_path("cashflow.json"), payloads.cashflow_config),
        (500, "prod_config_cursor", ctx.config_path("prod_config.json"), payloads.prod_config),
    ])
    return tuple(values)
