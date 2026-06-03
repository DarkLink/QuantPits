import pytest
import gc
import sys
from unittest.mock import MagicMock, patch

from quantpits.scripts.rolling.memory import (
    log_memory,
    cleanup_after_window,
    deep_cleanup_after_model,
    check_memory_pressure
)

def test_log_memory_success(capsys):
    mock_proc = MagicMock()
    mock_proc.memory_info.return_value.rss = 1.5 * 1e9
    
    mock_vm = MagicMock()
    mock_vm.percent = 45
    mock_vm.used = 8.0 * 1e9
    mock_vm.total = 16.0 * 1e9
    
    with patch('psutil.Process', return_value=mock_proc), \
         patch('psutil.virtual_memory', return_value=mock_vm):
        log_memory("test_tag")
        
    captured = capsys.readouterr()
    assert "🧹 [test_tag] RSS=1.5GB" in captured.out
    assert "System=45% (8.0/16.0GB)" in captured.out

def test_log_memory_import_error(capsys):
    with patch.dict('sys.modules', {'psutil': None}):
        log_memory("test_tag")
    captured = capsys.readouterr()
    assert captured.out == ""

def test_cleanup_after_window(capsys):
    mock_proc = MagicMock()
    mock_proc.memory_info.return_value.rss = 1.0 * 1e9
    mock_vm = MagicMock()
    mock_vm.percent = 30
    mock_vm.used = 5.0 * 1e9
    mock_vm.total = 16.0 * 1e9
    
    with patch('gc.collect') as mock_gc, \
         patch('psutil.Process', return_value=mock_proc), \
         patch('psutil.virtual_memory', return_value=mock_vm):
        cleanup_after_window("model_a", 2)
        mock_gc.assert_called_once()
        
    captured = capsys.readouterr()
    assert "🧹 [model_a|W2] RSS=1.0GB" in captured.out

def test_deep_cleanup_after_model_success(capsys):
    mock_H = MagicMock()
    
    mock_proc = MagicMock()
    mock_proc.memory_info.return_value.rss = 1.0 * 1e9
    mock_vm = MagicMock()
    mock_vm.percent = 30
    mock_vm.used = 5.0 * 1e9
    mock_vm.total = 16.0 * 1e9
    
    # We patch the qlib import using patch.dict for the full module hierarchy
    mock_qlib_cache = MagicMock()
    mock_qlib_cache.H = mock_H
    with patch.dict('sys.modules', {
             'qlib': MagicMock(),
             'qlib.data': MagicMock(),
             'qlib.data.cache': mock_qlib_cache
         }), \
         patch('gc.collect') as mock_gc, \
         patch('psutil.Process', return_value=mock_proc), \
         patch('psutil.virtual_memory', return_value=mock_vm):
        deep_cleanup_after_model("model_a")
        mock_H.clear.assert_called_once()
        assert mock_gc.call_count == 2
        
    captured = capsys.readouterr()
    assert "qlib MemCache cleared" in captured.out
    assert "ALL_DONE" in captured.out

def test_deep_cleanup_after_model_exception(capsys):
    # Simulate import qlib.data.cache fails or clear() throws
    mock_proc = MagicMock()
    mock_proc.memory_info.return_value.rss = 1.0 * 1e9
    mock_vm = MagicMock()
    mock_vm.percent = 30
    mock_vm.used = 5.0 * 1e9
    mock_vm.total = 16.0 * 1e9
    
    with patch.dict('sys.modules', {'qlib.data.cache': None}), \
         patch('gc.collect') as mock_gc, \
         patch('psutil.Process', return_value=mock_proc), \
         patch('psutil.virtual_memory', return_value=mock_vm):
        deep_cleanup_after_model("model_a")
        assert mock_gc.call_count == 2

def test_check_memory_pressure_low():
    mock_vm = MagicMock()
    mock_vm.percent = 50
    
    with patch('psutil.virtual_memory', return_value=mock_vm), \
         patch('quantpits.scripts.rolling.memory.deep_cleanup_after_model') as mock_cleanup:
        check_memory_pressure("test_tag", threshold_pct=80)
        mock_cleanup.assert_not_called()

def test_check_memory_pressure_high_under_90(capsys):
    mock_vm = MagicMock()
    mock_vm.percent = 85
    mock_vm.used = 13.6 * 1e9
    mock_vm.total = 16.0 * 1e9
    
    mock_vm_after = MagicMock()
    mock_vm_after.percent = 89
    
    with patch('psutil.virtual_memory', side_effect=[mock_vm, mock_vm_after]), \
         patch('quantpits.scripts.rolling.memory.deep_cleanup_after_model') as mock_cleanup:
        check_memory_pressure("test_tag", threshold_pct=80)
        mock_cleanup.assert_called_once_with("test_tag")

    captured = capsys.readouterr()
    assert "Memory pressure: 85%" in captured.out

def test_check_memory_pressure_high_over_90(capsys):
    mock_vm = MagicMock()
    mock_vm.percent = 92
    mock_vm.used = 14.7 * 1e9
    mock_vm.total = 16.0 * 1e9
    
    mock_vm_after = MagicMock()
    mock_vm_after.percent = 91
    
    with patch('psutil.virtual_memory', side_effect=[mock_vm, mock_vm_after]), \
         patch('quantpits.scripts.rolling.memory.deep_cleanup_after_model') as mock_cleanup:
        with pytest.raises(MemoryError, match="超过安全阈值"):
            check_memory_pressure("test_tag", threshold_pct=80)
        mock_cleanup.assert_called_once_with("test_tag")

def test_check_memory_pressure_import_error():
    with patch.dict('sys.modules', {'psutil': None}):
        check_memory_pressure("test_tag", threshold_pct=80)
