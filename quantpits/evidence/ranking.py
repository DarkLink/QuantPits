"""Canonical, full-universe signal ranking."""

from __future__ import annotations

import csv
import io
import math
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
        if not isinstance(self.rows, tuple):
            raise ContractError("ranking rows must be an exact tuple")
        expected_fields = {
            "instrument", "eligible", "scored", "coverage_status",
            "raw_score", "rank", "normalized_score",
        }
        instruments = []
        scored_rows = []
        if not self.rows:
            raise ContractError("ranking rows must be non-empty")
        for row in self.rows:
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise ContractError("ranking row fields are not exact")
            instrument = row["instrument"]
            if not isinstance(instrument, str) or not instrument:
                raise ContractError("ranking instrument must be non-empty text")
            if row["eligible"] is not True or not isinstance(row["scored"], bool):
                raise ContractError("ranking eligibility flags are invalid")
            instruments.append(instrument)
            if row["scored"]:
                if (
                    row["coverage_status"] != "scored"
                    or not isinstance(row["raw_score"], str)
                    or not isinstance(row["normalized_score"], str)
                    or isinstance(row["rank"], bool)
                    or not isinstance(row["rank"], int)
                    or row["rank"] <= 0
                ):
                    raise ContractError("scored ranking row is invalid")
                try:
                    if not math.isfinite(float(row["raw_score"])) or not math.isfinite(float(row["normalized_score"])):
                        raise ValueError
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ContractError("scored ranking values must be finite") from exc
                scored_rows.append(row)
            elif (
                row["coverage_status"] not in {"missing_prediction", "invalid_prediction"}
                or row["raw_score"] != "" or row["rank"] != ""
                or row["normalized_score"] != ""
            ):
                raise ContractError("unscored ranking row is invalid")
        if len(set(instruments)) != len(instruments):
            raise ContractError("ranking instruments must be unique")
        if self.eligible_count != len(self.rows):
            raise ContractError("ranking count must derive from terminal rows")
        if self.scored_count != len(scored_rows) or self.missing_count != len(self.rows) - len(scored_rows):
            raise ContractError("ranking terminal partition is not exact")
        if sorted(row["rank"] for row in scored_rows) != list(range(1, len(scored_rows) + 1)):
            raise ContractError("ranking positions must be contiguous")
        ordered_scored = sorted(
            scored_rows,
            key=lambda row: (-float(row["raw_score"]), row["instrument"]),
        )
        if [row["instrument"] for row in scored_rows] != [row["instrument"] for row in ordered_scored]:
            raise ContractError("ranking rows do not follow score and instrument order")
        expected_order = sorted(
            self.rows,
            key=lambda row: (
                row["rank"] if row["scored"] else len(self.rows) + 1,
                row["instrument"],
            ),
        )
        if list(self.rows) != expected_order:
            raise ContractError("ranking terminal row order is not canonical")
        if scored_rows:
            values = [float(row["raw_score"]) for row in scored_rows]
            low, high = min(values), max(values)
            span = high - low
            for row in scored_rows:
                expected = 0.0 if span == 0 else ((float(row["raw_score"]) - low) / span * 200.0 - 100.0)
                if row["normalized_score"] != format(expected, ".17g"):
                    raise ContractError("ranking normalized score is not derived from raw scores")
        expected_complete = self.missing_count == 0
        if not isinstance(self.complete, bool) or self.complete != expected_complete:
            raise ContractError("ranking completeness must derive from terminal rows")

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
    universe = tuple(eligible)
    if (
        not universe or len(set(universe)) != len(universe)
        or any(not isinstance(item, str) or not item for item in universe)
    ):
        raise ContractError("eligible universe must be non-empty, unique, and exact")
    if not isinstance(scores, Mapping) or any(not isinstance(item, str) or not item for item in scores):
        raise ContractError("prediction instrument identities must be non-empty text")
    foreign = set(scores) - set(universe)
    if foreign:
        raise ContractError("prediction contains foreign universe members")
    valid = {key: float(value) for key, value in scores.items() if finite_number(value)}
    invalid = set(scores) - set(valid)
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
