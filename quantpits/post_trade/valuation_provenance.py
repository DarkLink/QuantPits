"""Deterministic, privacy-safe account valuation evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Optional, Tuple


@dataclass(frozen=True)
class ValuationQuoteEvidence:
    instrument: str
    price: Decimal
    purpose: str
    market_date: str
    observed_at: Optional[str]
    effective_date: Optional[str]
    source_type: str
    source_field: str
    basis_claim: str
    basis_verification: str
    raw_close: Optional[Decimal]
    factor: Optional[Decimal]
    corporate_action_status: str = "unknown"

    def to_public_dict(self):
        return {key: (str(value) if isinstance(value, Decimal) else value) for key, value in asdict(self).items()}


def valuation_evidence_jsonl(snapshots) -> bytes:
    """Serialize one deterministic sanitized record per market date."""
    lines = []
    for snapshot in sorted(snapshots, key=lambda item: item.trade_date):
        payload = {
            "schema_version": 1,
            "market_date": snapshot.trade_date,
            "purpose": "account_valuation",
            "quotes": [item.to_public_dict() for item in sorted(snapshot.quote_evidence, key=lambda x: x.instrument)],
            "benchmark": snapshot.benchmark_evidence.to_public_dict() if snapshot.benchmark_evidence else None,
        }
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def merge_valuation_evidence(existing: bytes, snapshots) -> bytes:
    replacement = {item.trade_date: item for item in snapshots}
    records = {}
    if existing:
        for line in existing.decode("utf-8").splitlines():
            value = json.loads(line)
            if value.get("schema_version") != 1 or not value.get("market_date"):
                raise ValueError("Unsupported valuation evidence record")
            date = value["market_date"]
            if date in records:
                raise ValueError("Duplicate valuation evidence date")
            records[date] = value
    for date, snapshot in replacement.items():
        records[date] = json.loads(valuation_evidence_jsonl((snapshot,)).decode("utf-8"))
    return ("".join(json.dumps(records[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for key in sorted(records))).encode("utf-8")
