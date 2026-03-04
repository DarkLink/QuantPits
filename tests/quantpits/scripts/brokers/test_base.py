import pytest
import pandas as pd
from quantpits.scripts.brokers.base import (
    BaseBrokerAdapter,
    SELL_TYPES,
    BUY_TYPES,
    INTEREST_TYPES,
    REQUIRED_COLUMNS,
)


class MockAdapter(BaseBrokerAdapter):
    """Concrete implementation for testing the abstract base class."""

    @property
    def name(self):
        return "mock_broker"

    def read_settlement(self, file_path: str) -> pd.DataFrame:
        return pd.DataFrame()


def test_constants():
    """Standard constants should be non-empty lists."""
    assert len(SELL_TYPES) > 0
    assert len(BUY_TYPES) > 0
    assert len(INTEREST_TYPES) > 0
    assert len(REQUIRED_COLUMNS) > 0
    assert "证券代码" in REQUIRED_COLUMNS
    assert "交易类别" in REQUIRED_COLUMNS


def test_validate_pass():
    adapter = MockAdapter()
    df = pd.DataFrame({
        "证券代码": ["000001"],
        "交易类别": ["买入"],
        "成交价格": [10.0],
        "成交数量": [100.0],
        "成交金额": [1000.0],
        "资金发生数": [-1000.0],
        "交收日期": ["2026-03-01"],
    })
    result = adapter.validate(df)
    assert not result.empty
    assert list(result.columns) == list(df.columns)


def test_validate_missing_columns():
    adapter = MockAdapter()
    df = pd.DataFrame({
        "证券代码": ["000001"],
        # Missing other required columns
    })
    with pytest.raises(ValueError, match="缺失必要列"):
        adapter.validate(df)


def test_validate_empty_df():
    adapter = MockAdapter()
    df = pd.DataFrame()
    result = adapter.validate(df)
    assert result.empty


def test_adapter_name():
    adapter = MockAdapter()
    assert adapter.name == "mock_broker"
