import pytest

from quantpits.post_trade.contracts import PostTradeReceiptLedgerSchemaError
from quantpits.post_trade.intake import load_ingestion_receipts


def test_strict_receipt_ledger_fails_closed(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{broken", encoding="utf-8")
    assert load_ingestion_receipts(path) == {}
    with pytest.raises(PostTradeReceiptLedgerSchemaError):
        load_ingestion_receipts(path, strict=True)
