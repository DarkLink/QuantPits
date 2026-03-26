import pytest
import pandas as pd
from quantpits.utils.predict_utils import split_is_oos

def test_split_is_oos_basic():
    # Setup dummy data
    dates = pd.date_range('2026-01-01', periods=10, freq='D')
    instruments = ['SH600000', 'SZ000001']
    idx = pd.MultiIndex.from_product([dates, instruments], names=['datetime', 'instrument'])
    df = pd.DataFrame({'score': range(len(idx))}, index=idx)
    
    cutoff = pd.Timestamp('2026-01-05')
    
    is_df, oos_df = split_is_oos(df, cutoff)
    
    # Check IS
    assert is_df.index.get_level_values('datetime').max() == cutoff
    assert len(is_df.index.get_level_values('datetime').unique()) == 5
    
    # Check OOS
    assert oos_df.index.get_level_values('datetime').min() == pd.Timestamp('2026-01-06')
    assert len(oos_df.index.get_level_values('datetime').unique()) == 5

def test_split_is_oos_with_filters():
    dates = pd.date_range('2026-01-01', periods=10, freq='D')
    instruments = ['SH600000']
    idx = pd.MultiIndex.from_product([dates, instruments], names=['datetime', 'instrument'])
    df = pd.DataFrame({'score': range(len(idx))}, index=idx)
    
    cutoff = pd.Timestamp('2026-01-05')
    start = pd.Timestamp('2026-01-02')
    end = pd.Timestamp('2026-01-08')
    
    is_df, oos_df = split_is_oos(df, cutoff, start_date=start, end_date=end)
    
    # Check IS (should be 2nd, 3rd, 4th, 5th)
    assert is_df.index.get_level_values('datetime').min() == start
    assert is_df.index.get_level_values('datetime').max() == cutoff
    assert len(is_df) == 4
    
    # Check OOS (should be 6th, 7th, 8th)
    assert oos_df.index.get_level_values('datetime').min() == pd.Timestamp('2026-01-06')
    assert oos_df.index.get_level_values('datetime').max() == end
    assert len(oos_df) == 3

def test_split_is_oos_edge_cases():
    dates = pd.date_range('2026-01-01', periods=5, freq='D')
    idx = pd.MultiIndex.from_product([dates, ['A']], names=['datetime', 'instrument'])
    df = pd.DataFrame({'s': range(5)}, index=idx)
    
    # Cutoff before any data
    is_df, oos_df = split_is_oos(df, pd.Timestamp('2025-12-31'))
    assert is_df.empty
    assert len(oos_df) == 5
    
    # Cutoff after all data
    is_df, oos_df = split_is_oos(df, pd.Timestamp('2026-01-10'))
    assert len(is_df) == 5
    assert oos_df.empty

def test_save_predictions_to_recorder():
    from unittest.mock import patch, MagicMock
    from quantpits.utils.predict_utils import save_predictions_to_recorder
    
    df = pd.DataFrame({'score': [1.0]})
    with patch('qlib.workflow.R') as mock_R:
        mock_rec = MagicMock()
        mock_rec.info = {"id": "rec_123"}
        mock_R.get_recorder.return_value = mock_rec
        
        # Test with default tags
        # Signature: pred, experiment_name, model_name, tags
        rid = save_predictions_to_recorder(df, "exp_b", "model_a")
        
        assert rid == "rec_123"
        mock_R.start.assert_called_with(experiment_name="exp_b")
        mock_R.set_tags.assert_called()
        mock_R.save_objects.assert_called()
        
        # Test with custom tags
        rid2 = save_predictions_to_recorder(df, "exp_b", "model_a", tags={"custom": "val"})
        assert rid2 == "rec_123"
