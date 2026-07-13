import json
from decimal import Decimal

import pytest

from quantpits.post_trade.state import ValuationSnapshot
from quantpits.post_trade.contracts import ValuationSchemaError
from quantpits.post_trade.valuation_provenance import (
    ValuationQuoteEvidence, merge_valuation_evidence,
)


def quote(name="SH000001", price="10", date="2026-01-02"):
    return ValuationQuoteEvidence(
        name, Decimal(price), "account_valuation", date, None, date,
        "qlib_local", "Div($close,$factor)", "unadjusted_close_from_close_div_factor",
        "formula_verified", Decimal("20"), Decimal("2"),
    )


def test_evidence_must_match_operational_closes():
    with pytest.raises(ValuationSchemaError):
        ValuationSnapshot("2026-01-02", (("SH000001", Decimal("11")),), Decimal("20"), (quote(),))


def test_sidecar_is_deterministic_and_replaces_exact_date():
    first = ValuationSnapshot("2026-01-02", (("SH000001", Decimal("10")),), Decimal("20"), (quote(),), quote("SH000300", "20"))
    payload = merge_valuation_evidence(b"", (first,))
    assert payload == merge_valuation_evidence(payload, (first,))
    assert json.loads(payload)["market_date"] == "2026-01-02"
