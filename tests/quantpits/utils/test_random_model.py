"""Unit tests for RandomModel (quantpits/utils/random_model.py)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from quantpits.utils.random_model import RandomModel


class TestRandomModel:
    """Test suite for RandomModel."""

    def test_init(self):
        """Test model initialization with and without seed."""
        model_none = RandomModel()
        assert model_none.seed is None
        assert model_none.fitted is True

        model_seed = RandomModel(seed=42)
        assert model_seed.seed == 42
        assert model_seed.fitted is True

    def test_fit_noop(self):
        """Test fit is a no-op and updates evals_result if provided."""
        model = RandomModel()
        mock_dataset = MagicMock()

        # Without evals_result
        model.fit(mock_dataset)

        # With evals_result dict
        evals_result = {}
        model.fit(mock_dataset, evals_result=evals_result)
        assert "train" in evals_result
        assert "valid" in evals_result
        assert evals_result["train"] == [0.0]
        assert evals_result["valid"] == [0.0]

    def test_predict_structure_and_bounds(self):
        """Test predict generates valid float scores in [0, 1) with matching MultiIndex."""
        model = RandomModel(seed=123)

        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        instruments = ["SH600000", "SZ000001", "SZ000002"]
        multi_index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

        mock_dl = MagicMock()
        mock_dl.get_index.return_value = multi_index

        mock_dataset = MagicMock()
        mock_dataset.prepare.return_value = mock_dl

        preds = model.predict(mock_dataset, segment="test")

        assert isinstance(preds, pd.Series)
        assert preds.name == "score"
        assert len(preds) == len(multi_index)
        assert preds.index.equals(multi_index)
        assert np.all(preds.values >= 0.0)
        assert np.all(preds.values <= 1.0)
        assert preds.dtype == np.float64

    def test_predict_seed_determinism(self):
        """Test that same seed yields identical predictions, and different seeds yield different predictions."""
        dates = pd.date_range("2023-01-01", periods=2, freq="D")
        instruments = ["SH600000", "SZ000001"]
        multi_index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

        mock_dl = MagicMock()
        mock_dl.get_index.return_value = multi_index
        mock_dataset = MagicMock()
        mock_dataset.prepare.return_value = mock_dl

        model_1a = RandomModel(seed=42)
        preds_1a = model_1a.predict(mock_dataset)

        model_1b = RandomModel(seed=42)
        preds_1b = model_1b.predict(mock_dataset)

        model_2 = RandomModel(seed=999)
        preds_2 = model_2.predict(mock_dataset)

        np.testing.assert_array_equal(preds_1a.values, preds_1b.values)
        assert not np.array_equal(preds_1a.values, preds_2.values)
