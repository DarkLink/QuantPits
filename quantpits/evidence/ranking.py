"""Canonical, full-universe signal ranking."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple

from quantpits.evidence.contracts import ContractError, finite_number


@dataclass(frozen=True)
class RankingResult:
    rows: Tuple[dict, ...]
    eligible_count: int
    scored_count: int
    missing_count: int
    complete: bool

    def __post_init__(self) -> None:
        if self.eligible_count != len(self.rows):
            raise ContractError("ranking count must derive from terminal rows")
        if self.scored_count + self.missing_count != self.eligible_count:
            raise ContractError("ranking terminal partition is not exact")

    def to_csv_bytes(self) -> bytes:
        stream = io.StringIO(newline="")
        fields = (
            "instrument", "eligible", "scored", "coverage_status",
            "raw_score", "rank", "normalized_score",
        )
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in self.rows:
            writer.writerow(row)
        return stream.getvalue().encode("utf-8")


def canonical_full_ranking(
    eligible: Iterable[str], scores: Mapping[str, Any],
) -> RankingResult:
    universe = tuple(str(item) for item in eligible)
    if not universe or len(set(universe)) != len(universe) or any(not item for item in universe):
        raise ContractError("eligible universe must be non-empty, unique, and exact")
    foreign = set(str(item) for item in scores) - set(universe)
    if foreign:
        raise ContractError("prediction contains foreign universe members")
    valid = {str(key): float(value) for key, value in scores.items() if finite_number(value)}
    invalid = set(str(key) for key in scores) - set(valid)
    ordered = sorted(valid, key=lambda item: (-valid[item], item))
    ranks = {item: position + 1 for position, item in enumerate(ordered)}
    values = list(valid.values())
    low, high = (min(values), max(values)) if values else (0.0, 0.0)
    span = high - low
    normalized = {
        item: 0.0 if span == 0 else ((valid[item] - low) / span * 200.0 - 100.0)
        for item in valid
    }
    rows = []
    for instrument in sorted(universe, key=lambda item: (ranks.get(item, len(universe) + 1), item)):
        scored = instrument in valid
        rows.append({
            "instrument": instrument,
            "eligible": True,
            "scored": scored,
            "coverage_status": (
                "scored" if scored else
                ("invalid_prediction" if instrument in invalid else "missing_prediction")
            ),
            "raw_score": format(valid[instrument], ".17g") if scored else "",
            "rank": ranks[instrument] if scored else "",
            "normalized_score": format(normalized[instrument], ".17g") if scored else "",
        })
    return RankingResult(
        tuple(rows), len(rows), len(valid), len(rows) - len(valid),
        not invalid and len(valid) == len(rows),
    )
