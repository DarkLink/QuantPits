"""Privacy-safe broker account snapshot contracts and parsing helpers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional, Tuple

from quantpits.post_trade.contracts import BrokerAccountSnapshotError
from quantpits.post_trade.state import decimal_value, normalize_instrument


@dataclass(frozen=True)
class BrokerPositionObservation:
    instrument: str
    quantity: Decimal
    available_quantity: Optional[Decimal]
    display_price: Decimal
    market_value: Decimal
    corporate_action_status: str = "none"


@dataclass(frozen=True)
class BrokerAccountSnapshot:
    broker: str
    source_path: str
    source_fingerprint: str
    observed_at: str
    effective_date: Optional[str]
    asserted_market_date: Optional[str]
    cash: Decimal
    equity_market_value: Decimal
    total_assets: Decimal
    positions: Tuple[BrokerPositionObservation, ...]
    date_source: str = "broker_observation"

    def to_public_dict(self):
        value = asdict(self)
        value["positions"] = [
            {key: (str(item) if isinstance(item, Decimal) else item) for key, item in row.items()}
            for row in value["positions"]
        ]
        for key in ("cash", "equity_market_value", "total_assets"):
            value[key] = str(value[key])
        return value


def source_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_snapshot(snapshot: BrokerAccountSnapshot, *, tolerance=Decimal("0.01")):
    names = [item.instrument for item in snapshot.positions]
    if len(names) != len(set(names)):
        raise BrokerAccountSnapshotError("Broker snapshot contains duplicate instruments")
    if abs(snapshot.cash + snapshot.equity_market_value - snapshot.total_assets) > tolerance:
        raise BrokerAccountSnapshotError("Broker account totals do not reconcile")
    if abs(sum((item.market_value for item in snapshot.positions), Decimal("0")) - snapshot.equity_market_value) > tolerance:
        raise BrokerAccountSnapshotError("Broker position values do not reconcile to equity")
    for item in snapshot.positions:
        if item.quantity < 0 or item.market_value < 0 or item.display_price < 0:
            raise BrokerAccountSnapshotError("Broker snapshot contains a negative position value")
    return snapshot


def position_from_mapping(row):
    note = str(row.get("corporate_action_note", "")).strip()
    status = "none"
    if note:
        lowered = note.lower()
        if "提前" in note or "除权" in note or "adjust" in lowered:
            status = "adjusted_in_advance"
        else:
            status = "review_required"
    return BrokerPositionObservation(
        normalize_instrument(row["instrument"]),
        decimal_value(row["quantity"], field="broker quantity"),
        decimal_value(row["available_quantity"], field="broker available quantity") if row.get("available_quantity") not in (None, "") else None,
        decimal_value(row["display_price"], field="broker display price"),
        decimal_value(row["market_value"], field="broker market value"),
        status,
    )
