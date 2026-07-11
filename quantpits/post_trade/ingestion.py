"""Idempotent execution-evidence persistence keyed by source fingerprints."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Tuple

import pandas as pd

from quantpits.post_trade.contracts import IngestionPersistenceError, ParsedPostTradeInput
from quantpits.utils.workspace import WorkspaceContext


@dataclass(frozen=True)
class IngestionResult:
    outputs: Tuple[Path, ...]
    ingested_sources: Tuple[str, ...]
    max_trade_date: Optional[str] = None


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try: os.unlink(tmp_name)
        except OSError: pass
        raise


def _merged_csv(path: Path, frames: list[pd.DataFrame]) -> bytes:
    existing = pd.read_csv(path, dtype={"证券代码": str}) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing] + frames, ignore_index=True) if frames else existing
    combined = combined.drop_duplicates()
    return combined.to_csv(index=False).encode("utf-8-sig")


def ingest_execution_evidence(
    ctx: WorkspaceContext,
    parsed: Mapping[Tuple[str, str], ParsedPostTradeInput], *, run_id: str,
) -> IngestionResult:
    pending = [
        item for (stream, _), item in parsed.items()
        if stream in {"order", "trade"} and item.source.status == "present"
    ]
    order_items = [item for item in pending if item.source.stream == "order"]
    trade_items = [item for item in pending if item.source.stream == "trade"]
    order_path = ctx.data_path("raw_order_log_full.csv")
    trade_path = ctx.data_path("raw_trade_log_full.csv")
    ledger_path = ctx.data_path("post_trade_ingestion_state.json")
    legacy_path = ctx.data_path(".order_trade_state.json")
    if not pending:
        return IngestionResult((), ())
    try:
        # Keep the direct engine API safe as well as the command path: source
        # contents may change between strict parsing and the first write.
        from quantpits.post_trade.intake import verify_source
        for item in pending:
            verify_source(item.source)
        payloads = []
        if order_items: payloads.append((order_path, _merged_csv(order_path, [x.dataframe for x in order_items])))
        if trade_items: payloads.append((trade_path, _merged_csv(trade_path, [x.dataframe for x in trade_items])))
        for path, payload in payloads: _atomic_bytes(path, payload)
        ledger = {"schema_version": 1, "sources": {}}
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger.setdefault("schema_version", 1); ledger.setdefault("sources", {})
        now = datetime.now(timezone.utc).isoformat()
        committed_keys = []
        for item in pending:
            key = "%s:%s" % (item.source.stream, item.source.display_path)
            committed_keys.append(key)
            ledger["sources"][key] = {
                "stream": item.source.stream, "trade_date": item.source.trade_date,
                "path": item.source.display_path, "sha256": item.source.fingerprint,
                "row_count": item.row_count, "ingested_at": now, "run_id": run_id,
            }
        _atomic_bytes(ledger_path, json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
        max_date = max(item.source.trade_date for item in pending)
        try:
            _atomic_bytes(legacy_path, json.dumps({"last_processed_date": max_date}).encode("utf-8"))
        except Exception as exc:
            warnings.warn(
                "Execution evidence committed, but the legacy cursor mirror could not be updated: %s" % exc,
                RuntimeWarning,
                stacklevel=2,
            )
        outputs = tuple(path for path, _ in payloads) + (ledger_path,)
        return IngestionResult(outputs, tuple(sorted(committed_keys)), max_date)
    except Exception as exc:
        if isinstance(exc, IngestionPersistenceError): raise
        raise IngestionPersistenceError("Failed to persist execution evidence: %s" % exc) from exc
