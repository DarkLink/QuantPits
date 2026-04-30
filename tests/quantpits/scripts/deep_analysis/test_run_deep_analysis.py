import os
os.environ["QLIB_WORKSPACE_DIR"] = "/tmp"

import pytest
from unittest.mock import MagicMock, patch, mock_open
import sys
import json

# To handle the top-level os.chdir
with patch('os.chdir'):
    from quantpits.scripts import run_deep_analysis

def test_parse_args():
    # Test default args
    with patch('sys.argv', ['run_deep_analysis.py']):
        args = run_deep_analysis.parse_args()
        assert args.windows == 'full,weekly_era,1y,6m,3m,1m'
        assert args.llm is False
        assert args.agents == 'all'
        assert args.critic is False
        assert args.critic_dry_run is False

    # Test custom args
    with patch('sys.argv', ['run_deep_analysis.py', '--windows', '1m', '--llm', '--agents', 'model_health']):
        args = run_deep_analysis.parse_args()
        assert args.windows == '1m'
        assert args.llm is True
        assert args.agents == 'model_health'

    # Test critic args
    with patch('sys.argv', ['run_deep_analysis.py', '--critic']):
        args = run_deep_analysis.parse_args()
        assert args.critic is True
        assert args.critic_dry_run is False

    with patch('sys.argv', ['run_deep_analysis.py', '--critic-dry-run']):
        args = run_deep_analysis.parse_args()
        assert args.critic_dry_run is True

def test_load_deep_analysis_config(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config_dir = workspace / "config"
    config_dir.mkdir()
    
    # 1. Missing file
    assert run_deep_analysis.load_deep_analysis_config(str(workspace)) == {}
    
    # 2. Valid file
    config_file = config_dir / "deep_analysis_config.json"
    config_data = {"freq_change_date": "2024-01-01"}
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    assert run_deep_analysis.load_deep_analysis_config(str(workspace)) == config_data
    
    # 3. Invalid JSON
    with open(config_file, 'w') as f:
        f.write("invalid json")
    assert run_deep_analysis.load_deep_analysis_config(str(workspace)) == {}

@patch('quantpits.scripts.run_deep_analysis.parse_args')
@patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
@patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
@patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
@patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
@patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
@patch('quantpits.scripts.deep_analysis.config_ledger.snapshot_configs')
@patch('quantpits.scripts.deep_analysis.config_ledger.save_snapshot')
@patch('os.makedirs')
@patch('builtins.open', new_callable=mock_open)
@patch('os.path.exists')
@patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
def test_main_full_flow(mock_exists, mock_open_file, mock_makedirs, mock_save_snap, mock_snap_configs, 
                        mock_report_gen, mock_llm, mock_synth, mock_coord, 
                        mock_load_config, mock_parse_args):
    
    # Setup mocks
    args = MagicMock()
    args.windows = '1m,3m'
    args.freq_change_date = None
    args.output = 'output/report.md'
    args.llm = True
    args.llm_model = 'gpt-4'
    args.api_key = 'sk-test'
    args.base_url = None
    args.agents = 'all'
    args.notes = 'manual notes'
    args.notes_file = 'notes.txt'
    args.snapshot_config = True
    args.no_snapshot = False
    args.shareable = False
    args.critic = False
    args.critic_dry_run = False
    mock_parse_args.return_value = args

    mock_load_config.return_value = {'freq_change_date': '2024-01-01'}
    mock_exists.side_effect = lambda x: x == 'notes.txt' or x.endswith('config/deep_analysis_config.json')

    # Mock notes file content
    mock_open_file.return_value.read.return_value = "notes from file"

    mock_coord_inst = mock_coord.return_value
    mock_coord_inst.run.return_value = [MagicMock()]

    mock_synth_inst = mock_synth.return_value
    mock_synth_inst.synthesize.return_value = {'health_status': 'Healthy', 'cross_findings': [], 'recommendations': []}

    mock_llm_inst = mock_llm.return_value
    mock_llm_inst.is_available.return_value = True
    mock_llm_inst.generate_executive_summary.return_value = "Executive Summary"

    mock_report_gen_inst = mock_report_gen.return_value
    mock_report_gen_inst.generate.return_value = "# Deep Analysis Report"

    # Run main
    result = run_deep_analysis.main()

    assert result == 0
    mock_coord.assert_called_once()
    mock_synth.assert_called_once()
    mock_llm.assert_called_once_with(api_key='sk-test', model='gpt-4', base_url=None)
    mock_report_gen.assert_called_once()
    mock_snap_configs.assert_called_once()
    mock_save_snap.assert_called_once()
    
    # Check if report was written
    mock_open_file.assert_any_call('/tmp/root/output/report.md', 'w', encoding='utf-8')

@patch('quantpits.scripts.run_deep_analysis.parse_args')
@patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
@patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
@patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
@patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
@patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
@patch('os.path.exists', return_value=False)
@patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
def test_main_no_agents(mock_exists, mock_report_gen, mock_llm, mock_synth, mock_coord, 
                        mock_load_config, mock_parse_args):
    
    args = MagicMock()
    args.agents = 'non_existent_agent'
    args.windows = '1m'
    args.snapshot_config = False
    args.no_snapshot = True
    args.notes_file = None
    args.notes = ''
    mock_parse_args.return_value = args
    mock_load_config.return_value = {}
    
    result = run_deep_analysis.main()
    assert result == 1

@patch('quantpits.scripts.run_deep_analysis.parse_args')
@patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
@patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
@patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
@patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
@patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
@patch('quantpits.scripts.deep_analysis.config_ledger.snapshot_configs')
@patch('os.path.exists', return_value=False)
@patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
def test_main_snapshot_failure(mock_exists, mock_snap_configs, mock_report_gen, mock_llm, 
                               mock_synth, mock_coord, mock_load_config, mock_parse_args):
    
    args = MagicMock()
    args.agents = 'all'
    args.windows = '1m'
    args.snapshot_config = True
    args.no_snapshot = False
    args.notes_file = None
    args.notes = ''
    args.llm = False
    args.output = 'report.md'
    args.critic = False
    args.critic_dry_run = False
    mock_parse_args.return_value = args
    mock_load_config.return_value = {}
    
    mock_snap_configs.side_effect = Exception("Snapshot failed")
    
    mock_coord_inst = mock_coord.return_value
    mock_coord_inst.run.return_value = []
    mock_synth.return_value.synthesize.return_value = {}
    mock_llm.return_value.generate_executive_summary.return_value = ""
    mock_report_gen.return_value.generate.return_value = ""
    
    with patch('builtins.open', mock_open()):
        result = run_deep_analysis.main()
    
    assert result == 0 # Should continue even if snapshot fails

@patch('quantpits.scripts.run_deep_analysis.parse_args')
@patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
@patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
@patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
@patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
@patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
@patch('os.path.exists', return_value=False)
@patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
def test_main_llm_not_available(mock_exists, mock_report_gen, mock_llm, 
                                mock_synth, mock_coord, mock_load_config, mock_parse_args):
    
    args = MagicMock()
    args.agents = 'all'
    args.windows = '1m'
    args.snapshot_config = False
    args.no_snapshot = True
    args.notes_file = None
    args.notes = ''
    args.llm = True
    args.output = 'report.md'
    args.critic = False
    args.critic_dry_run = False
    mock_parse_args.return_value = args
    mock_load_config.return_value = {}
    
    mock_coord_inst = mock_coord.return_value
    mock_coord_inst.run.return_value = []
    mock_synth.return_value.synthesize.return_value = {}
    
    mock_llm_inst = mock_llm.return_value
    mock_llm_inst.is_available.return_value = False
    mock_llm_inst.generate_executive_summary.return_value = "Template Summary"
    
    mock_report_gen.return_value.generate.return_value = ""
    
    with patch('builtins.open', mock_open()):
        result = run_deep_analysis.main()
    
    assert result == 0
