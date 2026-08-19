import math

import pytest

from quantpits.evidence.contracts import ContractError
from quantpits.evidence.ranking import canonical_full_ranking


def test_full_ranking_retains_eligible_unscored_members():
    result = canonical_full_ranking(("BBB", "AAA", "CCC"), {"AAA": 2.0, "CCC": 1.0})
    assert [row["instrument"] for row in result.rows] == ["AAA", "CCC", "BBB"]
    assert result.missing_count == 1
    assert result.rows[-1]["coverage_status"] == "missing_prediction"
    assert result.complete is False


def test_ties_and_constant_scores_are_deterministic_and_finite():
    first = canonical_full_ranking(("BBB", "AAA"), {"BBB": 4.0, "AAA": 4.0})
    second = canonical_full_ranking(("AAA", "BBB"), {"AAA": 4.0, "BBB": 4.0})
    assert first.to_csv_bytes() == second.to_csv_bytes()
    assert [row["instrument"] for row in first.rows] == ["AAA", "BBB"]
    assert all(math.isfinite(float(row["normalized_score"])) for row in first.rows)


def test_empty_or_duplicate_universe_cannot_forge_coverage():
    with pytest.raises(ContractError):
        canonical_full_ranking((), {})
    with pytest.raises(ContractError):
        canonical_full_ranking(("AAA", "AAA"), {"AAA": 1.0})


def test_foreign_and_non_finite_predictions_are_fail_closed():
    with pytest.raises(ContractError):
        canonical_full_ranking(("AAA",), {"FOREIGN": 1.0})
    result = canonical_full_ranking(("AAA",), {"AAA": float("nan")})
    assert result.complete is False
    assert result.rows[0]["coverage_status"] == "invalid_prediction"
