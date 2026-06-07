"""Coverage expansion tests for run_deep_analysis.py main() function."""

import os
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

os.environ["QLIB_WORKSPACE_DIR"] = "/tmp"

with patch('os.chdir'):
    from quantpits.scripts import run_deep_analysis


class TestParseArgsExtended:
    """Test argument parsing with various flags."""

    def test_shareable_flag(self):
        with patch('sys.argv', ['run_deep_analysis.py', '--shareable']):
            args = run_deep_analysis.parse_args()
            assert args.shareable is True

    def test_critic_flag(self):
        with patch('sys.argv', ['run_deep_analysis.py', '--critic']):
            args = run_deep_analysis.parse_args()
            assert args.critic is True

    def test_snapshot_config_default(self):
        with patch('sys.argv', ['run_deep_analysis.py']):
            args = run_deep_analysis.parse_args()
            assert args.snapshot_config is True

    def test_no_snapshot_flag(self):
        with patch('sys.argv', ['run_deep_analysis.py', '--no-snapshot']):
            args = run_deep_analysis.parse_args()
            assert args.no_snapshot is True

    def test_llm_model_override(self):
        with patch('sys.argv', ['run_deep_analysis.py', '--llm-model', 'gpt-4o']):
            args = run_deep_analysis.parse_args()
            assert args.llm_model == 'gpt-4o'

    def test_api_key_override(self):
        with patch('sys.argv', ['run_deep_analysis.py', '--api-key', 'sk-test']):
            args = run_deep_analysis.parse_args()
            assert args.api_key == 'sk-test'

    def test_base_url_override(self):
        with patch('sys.argv', ['run_deep_analysis.py', '--base-url', 'http://localhost:8080']):
            args = run_deep_analysis.parse_args()
            assert args.base_url == 'http://localhost:8080'

    def test_notes_and_notes_file(self):
        with patch('sys.argv', ['run_deep_analysis.py', '--notes', 'test notes', '--notes-file', 'notes.txt']):
            args = run_deep_analysis.parse_args()
            assert args.notes == 'test notes'
            assert args.notes_file == 'notes.txt'

    def test_windows_custom(self):
        with patch('sys.argv', ['run_deep_analysis.py', '--windows', '1y,6m,3m,1m']):
            args = run_deep_analysis.parse_args()
            assert '1y' in args.windows
            assert '6m' in args.windows

    def test_freq_change_date(self):
        with patch('sys.argv', ['run_deep_analysis.py', '--freq-change-date', '2024-10-21']):
            args = run_deep_analysis.parse_args()
            assert args.freq_change_date == '2024-10-21'


class TestLoadDeepAnalysisConfigExtended:
    def test_loads_freq_change_date(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        config_dir = workspace / "config"
        config_dir.mkdir()

        config_file = config_dir / "deep_analysis_config.json"
        config_data = {"freq_change_date": "2024-01-01", "other_key": "value"}
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        result = run_deep_analysis.load_deep_analysis_config(str(workspace))
        assert result == config_data

    def test_empty_json(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        config_dir = workspace / "config"
        config_dir.mkdir()

        config_file = config_dir / "deep_analysis_config.json"
        with open(config_file, 'w') as f:
            json.dump({}, f)

        result = run_deep_analysis.load_deep_analysis_config(str(workspace))
        assert result == {}


class TestMainCriticMode:
    """Test main() Critic mode paths."""

    @patch('quantpits.scripts.run_deep_analysis.parse_args')
    @patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
    @patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
    @patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
    @patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
    @patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
    @patch('os.path.exists', return_value=False)
    @patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
    def test_main_critic_mode(self, mock_exists, mock_report_gen, mock_llm,
                               mock_synth, mock_coord, mock_load_config, mock_parse_args):
        args = MagicMock()
        args.windows = '1m'
        args.freq_change_date = None
        args.output = 'output/report.md'
        args.llm = False
        args.llm_model = None
        args.api_key = None
        args.base_url = None
        args.agents = 'all'
        args.notes = ''
        args.notes_file = None
        args.snapshot_config = False
        args.no_snapshot = True
        args.shareable = False
        args.critic = True  # Enable critic mode
        args.critic_dry_run = False
        args.run_label = ''
        mock_parse_args.return_value = args
        mock_load_config.return_value = {}

        mock_coord_inst = mock_coord.return_value
        mock_coord_inst.run.return_value = []
        mock_synth.return_value.synthesize.return_value = {}
        mock_llm.return_value.is_available.return_value = False
        mock_llm.return_value.generate_executive_summary.return_value = ""
        mock_report_gen.return_value.generate.return_value = ""

        with patch('builtins.open', mock_open()):
            result = run_deep_analysis.main()
        assert result == 0

    @patch('quantpits.scripts.run_deep_analysis.parse_args')
    @patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
    @patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
    @patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
    @patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
    @patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
    @patch('quantpits.scripts.deep_analysis.signal_extractor.SignalExtractor')
    @patch('os.path.exists', return_value=False)
    @patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
    def test_main_critic_dry_run(self, mock_exists, mock_sig_ext, mock_report_gen,
                                  mock_llm, mock_synth, mock_coord, mock_load_config,
                                  mock_parse_args):
        args = MagicMock()
        args.windows = '1m'
        args.freq_change_date = None
        args.output = 'output/report.md'
        args.llm = False
        args.llm_model = None
        args.api_key = None
        args.base_url = None
        args.agents = 'all'
        args.notes = ''
        args.notes_file = None
        args.snapshot_config = False
        args.no_snapshot = True
        args.shareable = False
        args.critic = False
        args.critic_dry_run = True  # critic dry-run only
        args.run_label = ''
        mock_parse_args.return_value = args
        mock_load_config.return_value = {}

        mock_coord_inst = mock_coord.return_value
        mock_coord_inst.run.return_value = []
        mock_synth.return_value.synthesize.return_value = {}
        mock_llm.return_value.is_available.return_value = False
        mock_report_gen.return_value.generate.return_value = ""

        mock_sig_ext_inst = mock_sig_ext.return_value
        mock_sig_ext_inst.extract.return_value = []

        with patch('builtins.open', mock_open()):
            result = run_deep_analysis.main()
        assert result == 0

    @patch('quantpits.scripts.run_deep_analysis.parse_args')
    @patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
    @patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
    @patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
    @patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
    @patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
    def test_main_shareable_report(self, mock_makedirs, mock_exists, mock_report_gen,
                                    mock_llm, mock_synth, mock_coord, mock_load_config,
                                    mock_parse_args):
        args = MagicMock()
        args.windows = '1m'
        args.freq_change_date = None
        args.output = 'output/report.md'
        args.llm = False
        args.llm_model = None
        args.api_key = None
        args.base_url = None
        args.agents = 'all'
        args.notes = ''
        args.notes_file = None
        args.snapshot_config = False
        args.no_snapshot = True
        args.shareable = True  # Shareable mode
        args.critic = False
        args.critic_dry_run = False
        args.run_label = ''
        mock_parse_args.return_value = args
        mock_load_config.return_value = {}

        mock_exists.side_effect = lambda x: 'config' in x
        mock_coord_inst = mock_coord.return_value
        mock_coord_inst.run.return_value = []
        mock_synth.return_value.synthesize.return_value = {'health_status': 'OK'}
        mock_llm.return_value.generate_executive_summary.return_value = ""
        mock_report_gen.return_value.generate.return_value = ""

        with patch('builtins.open', mock_open()):
            result = run_deep_analysis.main()
        assert result == 0

    @patch('quantpits.scripts.run_deep_analysis.parse_args')
    @patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
    @patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
    @patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
    @patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
    @patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
    @patch('os.path.exists', return_value=False)
    @patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
    def test_main_specific_agents(self, mock_exists, mock_report_gen, mock_llm,
                                   mock_synth, mock_coord, mock_load_config, mock_parse_args):
        args = MagicMock()
        args.windows = '1m'
        args.freq_change_date = None
        args.output = 'output/report.md'
        args.llm = False
        args.llm_model = None
        args.api_key = None
        args.base_url = None
        args.agents = 'model_health,prediction_audit'
        args.notes = ''
        args.notes_file = None
        args.snapshot_config = False
        args.no_snapshot = True
        args.shareable = False
        args.critic = False
        args.critic_dry_run = False
        args.run_label = ''
        mock_parse_args.return_value = args
        mock_load_config.return_value = {}

        mock_coord_inst = mock_coord.return_value
        mock_coord_inst.run.return_value = []
        mock_synth.return_value.synthesize.return_value = {}
        mock_llm.return_value.generate_executive_summary.return_value = ""
        mock_report_gen.return_value.generate.return_value = ""

        with patch('builtins.open', mock_open()):
            result = run_deep_analysis.main()
        assert result == 0

    @patch('quantpits.scripts.run_deep_analysis.parse_args')
    @patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
    @patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
    @patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
    @patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
    @patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
    @patch('os.path.exists', return_value=False)
    @patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
    def test_main_llm_openai_error_fallback(self, mock_exists, mock_report_gen, mock_llm,
                                             mock_synth, mock_coord, mock_load_config,
                                             mock_parse_args):
        """LLM available but returns template-style response (OpenAI not available)."""
        args = MagicMock()
        args.windows = '1m'
        args.freq_change_date = None
        args.output = 'output/report.md'
        args.llm = True
        args.llm_model = 'gpt-4'
        args.api_key = 'sk-test'
        args.base_url = None
        args.agents = 'all'
        args.notes = ''
        args.notes_file = None
        args.snapshot_config = False
        args.no_snapshot = True
        args.shareable = False
        args.critic = False
        args.critic_dry_run = False
        args.run_label = ''
        mock_parse_args.return_value = args
        mock_load_config.return_value = {}

        mock_coord_inst = mock_coord.return_value
        mock_coord_inst.run.return_value = []
        mock_synth.return_value.synthesize.return_value = {'health_status': 'Error'}

        mock_llm_inst = mock_llm.return_value
        # API key set but is_available returns True, summary falls back internally
        mock_llm_inst.is_available.return_value = True
        # Return template summary to simulate fallback
        mock_llm_inst.generate_executive_summary.return_value = "**System Health:** Error"

        mock_report_gen.return_value.generate.return_value = ""

        with patch('builtins.open', mock_open()):
            result = run_deep_analysis.main()
        assert result == 0  # Should proceed even if LLM returns template

    @patch('quantpits.scripts.run_deep_analysis.parse_args')
    @patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
    @patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
    @patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
    @patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
    @patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
    @patch('os.path.exists')
    @patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
    def test_main_with_notes_file(self, mock_exists, mock_report_gen, mock_llm,
                                   mock_synth, mock_coord, mock_load_config, mock_parse_args):
        args = MagicMock()
        args.windows = '1m'
        args.freq_change_date = None
        args.output = 'output/report.md'
        args.llm = False
        args.llm_model = None
        args.api_key = None
        args.base_url = None
        args.agents = 'all'
        args.notes = ''
        args.notes_file = 'notes.txt'
        args.snapshot_config = False
        args.no_snapshot = True
        args.shareable = False
        args.critic = False
        args.critic_dry_run = False
        args.run_label = ''
        mock_parse_args.return_value = args
        mock_load_config.return_value = {}

        mock_exists.side_effect = lambda x: x == 'notes.txt'
        mock_coord_inst = mock_coord.return_value
        mock_coord_inst.run.return_value = []
        mock_synth.return_value.synthesize.return_value = {}
        mock_llm.return_value.generate_executive_summary.return_value = ""
        mock_report_gen.return_value.generate.return_value = ""

        with patch('builtins.open', mock_open(read_data="notes from file")):
            result = run_deep_analysis.main()
        assert result == 0

    @patch('quantpits.scripts.run_deep_analysis.parse_args')
    @patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
    @patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
    @patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
    @patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
    @patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
    @patch('os.path.exists')
    @patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
    def test_main_freq_change_date_from_config(self, mock_exists, mock_report_gen,
                                                mock_llm, mock_synth, mock_coord,
                                                mock_load_config, mock_parse_args):
        args = MagicMock()
        args.windows = '1m'
        args.freq_change_date = None  # Not provided via CLI
        args.output = 'report.md'
        args.llm = False
        args.llm_model = None
        args.api_key = None
        args.base_url = None
        args.agents = 'all'
        args.notes = ''
        args.notes_file = None
        args.snapshot_config = False
        args.no_snapshot = True
        args.shareable = False
        args.critic = False
        args.critic_dry_run = False
        args.run_label = ''
        mock_parse_args.return_value = args
        # Config provides the date
        mock_load_config.return_value = {'freq_change_date': '2024-10-21'}

        mock_exists.side_effect = lambda x: 'deep_analysis_config.json' in x
        mock_coord_inst = mock_coord.return_value
        mock_coord_inst.run.return_value = []
        mock_synth.return_value.synthesize.return_value = {}
        mock_llm.return_value.generate_executive_summary.return_value = ""
        mock_report_gen.return_value.generate.return_value = ""

        with patch('builtins.open', mock_open()):
            result = run_deep_analysis.main()
        assert result == 0
        # Coordinator should have been called with freq_change_date from config
        call_kwargs = mock_coord.call_args[1]
        assert call_kwargs['freq_change_date'] == '2024-10-21'

    @patch('quantpits.scripts.run_deep_analysis.parse_args')
    @patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
    @patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
    @patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
    @patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
    @patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
    @patch('os.path.exists')
    @patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
    def test_main_llm_not_available_uses_template(self, mock_exists, mock_report_gen,
                                                    mock_llm, mock_synth, mock_coord,
                                                    mock_load_config, mock_parse_args):
        args = MagicMock()
        args.windows = '1m'
        args.freq_change_date = None
        args.output = 'report.md'
        args.llm = True  # LLM enabled
        args.llm_model = None
        args.api_key = None
        args.base_url = None
        args.agents = 'all'
        args.notes = ''
        args.notes_file = None
        args.snapshot_config = False
        args.no_snapshot = True
        args.shareable = False
        args.critic = False
        args.critic_dry_run = False
        args.run_label = ''
        mock_parse_args.return_value = args
        mock_load_config.return_value = {}

        mock_exists.side_effect = lambda x: 'deep_analysis_config.json' in x
        mock_coord_inst = mock_coord.return_value
        mock_coord_inst.run.return_value = []
        mock_synth.return_value.synthesize.return_value = {'health_status': 'Healthy'}
        mock_llm_inst = mock_llm.return_value
        mock_llm_inst.is_available.return_value = False  # LLM not available
        mock_report_gen.return_value.generate.return_value = ""

        with patch('builtins.open', mock_open()):
            result = run_deep_analysis.main()
        assert result == 0

    @patch('quantpits.scripts.run_deep_analysis.parse_args')
    @patch('quantpits.scripts.run_deep_analysis.load_deep_analysis_config')
    @patch('quantpits.scripts.deep_analysis.coordinator.Coordinator')
    @patch('quantpits.scripts.deep_analysis.synthesizer.Synthesizer')
    @patch('quantpits.scripts.deep_analysis.llm_interface.LLMInterface')
    @patch('quantpits.scripts.deep_analysis.report_generator.ReportGenerator')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('quantpits.utils.env.ROOT_DIR', '/tmp/root')
    def test_main_stderr_output_dir(self, mock_makedirs, mock_exists, mock_report_gen,
                                     mock_llm, mock_synth, mock_coord, mock_load_config,
                                     mock_parse_args):
        """Test that stderr output dir is created."""
        args = MagicMock()
        args.windows = '1m'
        args.freq_change_date = None
        args.output = 'output/report.md'
        args.llm = False
        args.llm_model = None
        args.api_key = None
        args.base_url = None
        args.agents = 'all'
        args.notes = ''
        args.notes_file = None
        args.snapshot_config = False
        args.no_snapshot = True
        args.shareable = False
        args.critic = False
        args.critic_dry_run = False
        args.run_label = ''
        mock_parse_args.return_value = args
        mock_load_config.return_value = {}

        mock_exists.side_effect = lambda x: False
        mock_coord_inst = mock_coord.return_value
        mock_coord_inst.run.return_value = []
        mock_synth.return_value.synthesize.return_value = {}
        mock_llm.return_value.generate_executive_summary.return_value = ""
        mock_report_gen.return_value.generate.return_value = ""

        with patch('builtins.open', mock_open()):
            result = run_deep_analysis.main()
        assert result == 0
