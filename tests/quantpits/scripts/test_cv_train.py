"""
Tests for cv_train.py -- CPCV (Purged Cross-Validation) training script.

Coverage targets:
  - parse_args(): all 16 CLI flags, defaults, mode combinations
  - _resolve_targets(): model resolution, data_slice_mode validation, --skip
  - run_full_train_cpcv(): full flow, dry-run, errors, mixed success/failure
  - run_incremental_train_cpcv(): incremental, resume, dry-run, mixed results
  - run_predict_only_cpcv(): predict-only, dry-run, filtering, errors
  - main(): dispatcher routing for --list, --show-state, --clear-state, modes
  - Edge cases: file I/O errors, empty registries, corrupted state

Pattern mirrors tests/quantpits/scripts/test_static_train.py.
"""

import pytest
import os
import sys
import json
import yaml
import importlib
from unittest.mock import MagicMock, patch, mock_open


# ===================================================================
# Shared fixture -- sets up a minimal CPCV workspace
# ===================================================================

@pytest.fixture(autouse=True)
def mock_env(monkeypatch, tmp_path):
    """Create a minimal workspace for cv_train tests.

    Every test reloads env/train_utils/cv_train so module-level constants
    are resolved from the temporary workspace.
    """
    workspace = tmp_path / "MockWorkspace"
    (workspace / "config").mkdir(parents=True)
    (workspace / "data").mkdir(parents=True)
    (workspace / "output").mkdir(parents=True)

    # model_config with purged_cv mode enabled
    (workspace / "config" / "model_config.json").write_text(json.dumps({
        "market": "csi300",
        "benchmark": "SH000300",
        "freq": "week",
        "anchor_date": "2026-03-01",
        "data_slice_mode": "purged_cv",
        "test_start_time": "2025-07-07",
        "test_end_time": "2025-12-29",
        "purged_cv": {
            "n_test_groups": 4,
            "n_val_groups": 2,
            "n_groups": 60,
            "purge_steps": 10,
            "embargo_steps": 0,
        },
    }))

    # model registry with 4 models
    (workspace / "config" / "model_registry.yaml").write_text(yaml.dump({
        "models": {
            "gru_Alpha158": {
                "algorithm": "gru", "dataset": "Alpha158", "enabled": True,
                "yaml_file": "gru_Alpha158.yaml", "tags": ["ts"],
            },
            "lightgbm_Alpha158": {
                "algorithm": "lightgbm", "dataset": "Alpha158", "enabled": True,
                "yaml_file": "lightgbm_Alpha158.yaml", "tags": ["tree"],
            },
            "lstm_Alpha360": {
                "algorithm": "lstm", "dataset": "Alpha360", "enabled": False,
                "yaml_file": "lstm_Alpha360.yaml", "tags": ["ts"],
            },
            "xgb_Alpha158": {
                "algorithm": "xgb", "dataset": "Alpha158", "enabled": True,
                "yaml_file": "xgb_Alpha158.yaml", "tags": ["tree"],
            },
        }
    }))

    # Create a dummy yaml file that model training functions might reference
    (workspace / "config" / "gru_Alpha158.yaml").write_text("model: {class: GRU}\n")
    (workspace / "config" / "lightgbm_Alpha158.yaml").write_text("model: {class: LightGBM}\n")
    (workspace / "config" / "xgb_Alpha158.yaml").write_text("model: {class: XGB}\n")
    (workspace / "config" / "lstm_Alpha360.yaml").write_text("model: {class: LSTM}\n")

    # Add scripts to path so bare imports in script modules work
    scripts_dir = os.path.join(os.getcwd(), "quantpits", "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    monkeypatch.setattr(sys, 'argv', ['script.py'])
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    # Reload env + train_utils to pick up the temporary workspace
    for mod_name in ['env', 'quantpits.utils.env',
                     'train_utils', 'quantpits.utils.train_utils',
                     'cv_train', 'quantpits.scripts.cv_train']:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    # Patch module-level path constants to use the temporary workspace
    from quantpits.utils import train_utils
    monkeypatch.setattr(train_utils, 'ROOT_DIR', str(workspace))
    monkeypatch.setattr(train_utils, 'RECORD_OUTPUT_FILE',
                        str(workspace / "latest_train_records.json"))
    monkeypatch.setattr(train_utils, 'HISTORY_DIR',
                        str(workspace / "data" / "history"))
    monkeypatch.setattr(train_utils, 'RUN_STATE_FILE',
                        str(workspace / "data" / "run_state.json"))
    monkeypatch.setattr(train_utils, 'PRETRAINED_DIR',
                        str(workspace / "data" / "pretrain"))
    monkeypatch.setattr(train_utils, 'PREDICTION_OUTPUT_DIR',
                        str(workspace / "output" / "predictions"))
    os.makedirs(train_utils.PRETRAINED_DIR, exist_ok=True)

    from quantpits.scripts import cv_train as ct
    # Also patch ROOT_DIR on cv_train module (used by _resolve_targets, main)
    monkeypatch.setattr(ct, 'ROOT_DIR', str(workspace))
    yield ct, workspace


# Helper to build a standard CPCV params dict
def _make_cpcv_params(**overrides):
    p = {
        "freq": "week",
        "anchor_date": "2026-03-01",
        "data_slice_mode": "purged_cv",
        "cpcv_folds": [
            {"train_segments": [["2020-01-06", "2022-01-03"]],
             "valid_start_time": "2022-01-10", "valid_end_time": "2022-06-27",
             "test_start_time": "2025-01-06", "test_end_time": "2025-06-30"},
            {"train_segments": [["2022-07-04", "2025-06-30"]],
             "valid_start_time": "2020-01-06", "valid_end_time": "2020-06-29",
             "test_start_time": "2025-01-06", "test_end_time": "2025-06-30"},
        ],
        "test_start_time": "2025-07-07",
        "test_end_time": "2025-12-29",
    }
    p.update(overrides)
    return p


# ===================================================================
# 1. CLI parse_args tests
# ===================================================================

class TestParseArgs:
    """Cover all CLI flags, defaults, and mode combinations."""

    def test_defaults(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py']):
            args = ct.parse_args()
        assert args.full is False
        assert args.predict_only is False
        assert args.models is None
        assert args.algorithm is None
        assert args.dataset is None
        assert args.market is None
        assert args.tag is None
        assert args.all_enabled is False
        assert args.skip is None
        assert args.resume is False
        assert args.dry_run is False
        assert args.experiment_name is None
        assert args.no_pretrain is False
        assert args.source_records == 'latest_train_records.json'
        assert args.list is False
        assert args.show_state is False
        assert args.clear_state is False

    def test_full_mode(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--full']):
            args = ct.parse_args()
        assert args.full is True
        assert args.predict_only is False

    def test_predict_only_mode(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--predict-only', '--all-enabled']):
            args = ct.parse_args()
        assert args.predict_only is True
        assert args.all_enabled is True

    def test_incremental_default_models(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--models', 'gru_Alpha158,lightgbm_Alpha158', '--dry-run']):
            args = ct.parse_args()
        assert args.models == 'gru_Alpha158,lightgbm_Alpha158'
        assert args.dry_run is True
        assert args.full is False
        assert args.predict_only is False

    def test_resume_flag(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--models', 'gru_Alpha158', '--resume']):
            args = ct.parse_args()
        assert args.resume is True

    def test_list_flag(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--list']):
            args = ct.parse_args()
        assert args.list is True

    def test_show_state_flag(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--show-state']):
            args = ct.parse_args()
        assert args.show_state is True

    def test_clear_state_flag(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--clear-state']):
            args = ct.parse_args()
        assert args.clear_state is True

    def test_filter_by_algorithm(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--algorithm', 'gru', '--skip', 'xgb_Alpha158']):
            args = ct.parse_args()
        assert args.algorithm == 'gru'
        assert args.skip == 'xgb_Alpha158'

    def test_filter_by_dataset(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--dataset', 'Alpha360']):
            args = ct.parse_args()
        assert args.dataset == 'Alpha360'

    def test_filter_by_market(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--market', 'csi300']):
            args = ct.parse_args()
        assert args.market == 'csi300'

    def test_filter_by_tag(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--tag', 'ts']):
            args = ct.parse_args()
        assert args.tag == 'ts'

    def test_experiment_name(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--experiment-name', 'MyExp']):
            args = ct.parse_args()
        assert args.experiment_name == 'MyExp'

    def test_no_pretrain(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--no-pretrain']):
            args = ct.parse_args()
        assert args.no_pretrain is True

    def test_source_records_custom(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--source-records', 'my_records.json']):
            args = ct.parse_args()
        assert args.source_records == 'my_records.json'

    def test_combined_model_filters(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--algorithm', 'gru', '--dataset', 'Alpha158',
                                        '--tag', 'ts', '--market', 'csi300']):
            args = ct.parse_args()
        assert args.algorithm == 'gru'
        assert args.dataset == 'Alpha158'
        assert args.tag == 'ts'
        assert args.market == 'csi300'


# ===================================================================
# 2. _resolve_targets tests
# ===================================================================

class TestResolveTargets:
    """Test model resolution with data_slice_mode validation."""

    def test_resolve_by_names(self, mock_env):
        ct, _ = mock_env
        args = MagicMock()
        args.models = "gru_Alpha158,lightgbm_Alpha158"
        args.all_enabled = False
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None
        args.skip = None

        with patch('quantpits.utils.train_utils.resolve_target_models',
                   return_value={"gru_Alpha158": {}, "lightgbm_Alpha158": {}}):
            targets = ct._resolve_targets(args, {})
        assert "gru_Alpha158" in targets
        assert "lightgbm_Alpha158" in targets

    def test_resolve_all_enabled(self, mock_env):
        ct, _ = mock_env
        args = MagicMock()
        args.models = None
        args.all_enabled = True
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None
        args.skip = None

        with patch('quantpits.utils.train_utils.resolve_target_models',
                   return_value={"gru_Alpha158": {}, "lightgbm_Alpha158": {}, "xgb_Alpha158": {}}):
            targets = ct._resolve_targets(args, {})
        assert "gru_Alpha158" in targets

    def test_resolve_by_tag(self, mock_env):
        ct, _ = mock_env
        args = MagicMock()
        args.models = None
        args.all_enabled = False
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = "tree"
        args.skip = None

        with patch('quantpits.utils.train_utils.resolve_target_models',
                   return_value={"lightgbm_Alpha158": {}, "xgb_Alpha158": {}}):
            targets = ct._resolve_targets(args, {})
        assert "lightgbm_Alpha158" in targets
        assert "xgb_Alpha158" in targets

    def test_resolve_by_algorithm(self, mock_env):
        ct, _ = mock_env
        args = MagicMock()
        args.models = None
        args.all_enabled = False
        args.algorithm = "gru"
        args.dataset = None
        args.market = None
        args.tag = None
        args.skip = None

        with patch('quantpits.utils.train_utils.resolve_target_models',
                   return_value={"gru_Alpha158": {}}):
            targets = ct._resolve_targets(args, {})
        assert "gru_Alpha158" in targets

    def test_skip_models(self, mock_env):
        ct, _ = mock_env
        args = MagicMock()
        args.models = None
        args.all_enabled = True
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None
        args.skip = "gru_Alpha158,xgb_Alpha158"

        with patch('quantpits.utils.train_utils.resolve_target_models',
                   return_value={"gru_Alpha158": {}, "lightgbm_Alpha158": {}, "xgb_Alpha158": {}}):
            targets = ct._resolve_targets(args, {})
        assert "gru_Alpha158" not in targets
        assert "xgb_Alpha158" not in targets
        assert "lightgbm_Alpha158" in targets

    def test_no_targets_exits(self, mock_env):
        ct, _ = mock_env
        args = MagicMock()
        args.models = "nonexistent"
        args.all_enabled = False
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None
        args.skip = None

        with patch('quantpits.utils.train_utils.resolve_target_models',
                   return_value=None):
            with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
                with pytest.raises(SystemExit):
                    ct._resolve_targets(args, {})
                mock_exit.assert_called_once_with(1)

    def test_empty_targets_exits(self, mock_env):
        ct, _ = mock_env
        args = MagicMock()
        args.models = "nonexistent"
        args.all_enabled = False
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None
        args.skip = None

        with patch('quantpits.utils.train_utils.resolve_target_models',
                   return_value={}):
            with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
                with pytest.raises(SystemExit):
                    ct._resolve_targets(args, {})
                mock_exit.assert_called_once_with(1)

    def test_skip_with_empty_string(self, mock_env):
        """Line 130: --skip with only whitespace/commas."""
        ct, _ = mock_env
        args = MagicMock()
        args.models = None
        args.all_enabled = True
        args.algorithm = None
        args.dataset = None
        args.market = None
        args.tag = None
        args.skip = ",  ,"

        with patch('quantpits.utils.train_utils.resolve_target_models',
                   return_value={"gru_Alpha158": {}, "lightgbm_Alpha158": {}}):
            targets = ct._resolve_targets(args, {})
        assert "gru_Alpha158" in targets
        assert "lightgbm_Alpha158" in targets


# ===================================================================
# 3. run_full_train_cpcv tests
# ===================================================================

class TestRunFullTrainCpcv:
    """Full CPCV training orchestration."""

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_train_success(self, mock_json, mock_file, mock_backup,
                                 mock_overwrite, mock_train, mock_enabled,
                                 mock_registry, mock_dates, mock_init, mock_env):
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {
            "gru_Alpha158": {"yaml_file": "gru_Alpha158.yaml"},
            "lightgbm_Alpha158": {"yaml_file": "lightgbm_Alpha158.yaml"},
            "xgb_Alpha158": {"yaml_file": "xgb_Alpha158.yaml"},
        }
        mock_enabled.return_value = {
            "gru_Alpha158": {"yaml_file": "gru_Alpha158.yaml", "enabled": True},
            "lightgbm_Alpha158": {"yaml_file": "lightgbm_Alpha158.yaml", "enabled": True},
            "xgb_Alpha158": {"yaml_file": "xgb_Alpha158.yaml", "enabled": True},
        }
        mock_train.return_value = {
            "success": True, "record_id": "rid1",
            "performance": {"IC_Mean": 0.08, "ICIR": 0.5},
        }

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)

        mock_init.assert_called_once()
        assert mock_train.call_count == 3
        mock_overwrite.assert_called_once()

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    def test_full_train_dry_run(self, mock_train, mock_enabled, mock_registry,
                                 mock_dates, mock_init, mock_env):
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {}
        mock_enabled.return_value = {
            "gru_Alpha158": {"yaml_file": "gru_Alpha158.yaml"},
            "xgb_Alpha158": {"yaml_file": "xgb_Alpha158.yaml"},
        }

        args = MagicMock()
        args.dry_run = True
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)
        mock_train.assert_not_called()

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    def test_not_purged_cv_mode_exits(self, mock_dates, mock_init, mock_env):
        """Line 155-157: data_slice_mode is not purged_cv -> sys.exit(1)."""
        ct, _ = mock_env
        mock_dates.return_value = {"freq": "week", "data_slice_mode": "slide"}

        args = MagicMock()
        with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
            with pytest.raises(SystemExit):
                ct.run_full_train_cpcv(args)
            mock_exit.assert_called_once_with(1)

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    def test_no_cpcv_folds_exits(self, mock_dates, mock_init, mock_env):
        """Line 160-162: empty cpcv_folds -> sys.exit(1)."""
        ct, _ = mock_env
        mock_dates.return_value = {"freq": "week", "data_slice_mode": "purged_cv",
                                   "cpcv_folds": []}

        args = MagicMock()
        with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
            with pytest.raises(SystemExit):
                ct.run_full_train_cpcv(args)
            mock_exit.assert_called_once_with(1)

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    def test_no_enabled_models(self, mock_enabled, mock_registry, mock_dates,
                                mock_init, mock_env):
        """Line 169-171: no enabled models -- prints warning, returns early."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {}
        mock_enabled.return_value = {}

        args = MagicMock()
        ct.run_full_train_cpcv(args)  # Should print warning and return, not raise

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_train_mixed_results(self, mock_json, mock_file, mock_backup,
                                       mock_overwrite, mock_train, mock_enabled,
                                       mock_registry, mock_dates, mock_init, mock_env):
        """Lines 208-214: some models fail, some succeed."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {}
        mock_enabled.return_value = {
            "gru_Alpha158": {"yaml_file": "gru.yaml"},
            "lightgbm_Alpha158": {"yaml_file": "lgbm.yaml"},
            "xgb_Alpha158": {"yaml_file": "xgb.yaml"},
        }
        mock_train.side_effect = [
            {"success": True, "record_id": "rid1", "performance": {"IC_Mean": 0.08}},
            {"success": False, "error": "MLflow init failed"},
            {"success": True, "record_id": "rid3", "performance": {"IC_Mean": 0.05}},
        ]

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)
        assert mock_train.call_count == 3
        mock_overwrite.assert_called_once()

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_train_custom_experiment_name(self, mock_json, mock_file,
                                                mock_backup, mock_overwrite,
                                                mock_train, mock_enabled,
                                                mock_registry, mock_dates,
                                                mock_init, mock_env):
        """Line 186: custom experiment name forwarded to train_cpcv_model."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {}
        mock_enabled.return_value = {"gru_Alpha158": {"yaml_file": "gru.yaml"}}
        mock_train.return_value = {"success": True, "record_id": "rid1", "performance": {}}

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = "CustomCPCV"
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)

        call_args = mock_train.call_args_list[0]
        assert call_args[0][3] == "CustomCPCV"

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_train_no_pretrain_passed(self, mock_json, mock_file, mock_backup,
                                            mock_overwrite, mock_train, mock_enabled,
                                            mock_registry, mock_dates, mock_init,
                                            mock_env):
        """Line 206: --no-pretrain flag forwarded to train_cpcv_model."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {}
        mock_enabled.return_value = {"gru_Alpha158": {"yaml_file": "gru.yaml"}}
        mock_train.return_value = {"success": True, "record_id": "rid1", "performance": {}}

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = True

        ct.run_full_train_cpcv(args)

        call_kwargs = mock_train.call_args_list[0][1]
        assert call_kwargs.get('no_pretrain') is True

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_train_performance_file_saved(self, mock_json, mock_file,
                                                mock_backup, mock_overwrite,
                                                mock_train, mock_enabled,
                                                mock_registry, mock_dates,
                                                mock_init, mock_env):
        """Lines 220-225: performance JSON saved with anchor_date in filename."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {}
        mock_enabled.return_value = {"gru_Alpha158": {"yaml_file": "gru.yaml"}}
        mock_train.return_value = {"success": True, "record_id": "rid1",
                                    "performance": {"IC_Mean": 0.1, "ICIR": 0.6}}

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)

        mock_json.assert_called()
        assert mock_backup.call_count >= 1

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_train_all_fail(self, mock_json, mock_file, mock_backup,
                                  mock_overwrite, mock_train, mock_enabled,
                                  mock_registry, mock_dates, mock_init, mock_env):
        """All models fail -- records are overwritten with empty models dict."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {}
        mock_enabled.return_value = {
            "gru_Alpha158": {"yaml_file": "gru.yaml"},
            "lightgbm_Alpha158": {"yaml_file": "lgbm.yaml"},
            "xgb_Alpha158": {"yaml_file": "xgb.yaml"},
        }
        mock_train.side_effect = [
            {"success": False, "error": "OOM"},
            {"success": False, "error": "Config error"},
            {"success": False, "error": "Data error"},
        ]

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)
        mock_overwrite.assert_called_once()
        records_arg = mock_overwrite.call_args[0][0]
        assert records_arg['models'] == {}

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_train_mlflow_experiment_default(self, mock_json, mock_file,
                                                   mock_backup, mock_overwrite,
                                                   mock_train, mock_enabled,
                                                   mock_registry, mock_dates,
                                                   mock_init, mock_env):
        """Default experiment name when freq=month in params."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params(freq="month")
        mock_registry.return_value = {}
        mock_enabled.return_value = {"gru_Alpha158": {"yaml_file": "gru.yaml"}}
        mock_train.return_value = {"success": True, "record_id": "rid1", "performance": {}}

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)
        call_args = mock_train.call_args_list[0]
        assert call_args[0][3] == "Prod_Train_CPCV_MONTH"

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_train_performance_with_none(self, mock_json, mock_file,
                                               mock_backup, mock_overwrite,
                                               mock_train, mock_enabled,
                                               mock_registry, mock_dates,
                                               mock_init, mock_env):
        """Line 211-212: model succeeds but performance is None/empty."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {}
        mock_enabled.return_value = {"gru_Alpha158": {"yaml_file": "gru.yaml"}}
        mock_train.return_value = {"success": True, "record_id": "rid1",
                                    "performance": None}

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)  # Should not crash

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_train_performance_no_perf_key(self, mock_json, mock_file,
                                                 mock_backup, mock_overwrite,
                                                 mock_train, mock_enabled,
                                                 mock_registry, mock_dates,
                                                 mock_init, mock_env):
        """Line 211: result['success']=True but no 'performance' key."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {}
        mock_enabled.return_value = {"gru_Alpha158": {"yaml_file": "gru.yaml"}}
        mock_train.return_value = {"success": True, "record_id": "rid1",
                                    "performance": None}  # falsy, won't be recorded

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)  # Should not crash with None performance

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    def test_full_dry_run_multi_segment_folds(self, mock_enabled, mock_registry,
                                               mock_dates, mock_init, mock_env):
        """Lines 177-180: folds with multiple train segments displayed."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params(cpcv_folds=[{
            "train_segments": [
                ["2020-01-06", "2021-06-28"],
                ["2022-01-03", "2023-06-26"],
            ],
            "valid_start_time": "2021-07-05", "valid_end_time": "2021-12-27",
            "test_start_time": "2025-01-06", "test_end_time": "2025-06-30",
        }])
        mock_registry.return_value = {}
        mock_enabled.return_value = {"gru_Alpha158": {"yaml_file": "gru.yaml"}}

        args = MagicMock()
        args.dry_run = True
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)  # Should print multi-segment info

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.load_model_registry')
    @patch('quantpits.utils.train_utils.get_enabled_models')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.overwrite_train_records')
    @patch('quantpits.utils.train_utils.backup_file_with_date')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_train_single_model_only(self, mock_json, mock_file, mock_backup,
                                           mock_overwrite, mock_train, mock_enabled,
                                           mock_registry, mock_dates, mock_init,
                                           mock_env):
        """Only one enabled model in registry."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_registry.return_value = {}
        mock_enabled.return_value = {"gru_Alpha158": {"yaml_file": "gru.yaml"}}
        mock_train.return_value = {"success": True, "record_id": "rid1", "performance": {}}

        args = MagicMock()
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_full_train_cpcv(args)
        assert mock_train.call_count == 1


# ===================================================================
# 4. run_incremental_train_cpcv tests
# ===================================================================

class TestRunIncrementalTrainCpcv:
    """Incremental CPCV training with resume support."""

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.merge_performance_file')
    @patch('quantpits.utils.train_utils.save_run_state')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_success(self, mock_table, mock_save, mock_perf,
                                  mock_merge, mock_train, mock_dates,
                                  mock_init, mock_env):
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()

        mock_train.return_value = {"success": True, "record_id": "rid1",
                                    "performance": {"ICIR": 0.3}}

        args = MagicMock()
        args.resume = False
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        targets = {"gru_Alpha158": {"yaml_file": "gru_Alpha158.yaml"}}
        ct.run_incremental_train_cpcv(args, targets)

        mock_train.assert_called_once()
        mock_merge.assert_called_once()
        mock_perf.assert_called_once()

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_dry_run(self, mock_table, mock_dates, mock_init, mock_env):
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()

        args = MagicMock()
        args.resume = False
        args.dry_run = True

        targets = {"gru_Alpha158": {"yaml_file": "gru_Alpha158.yaml"}}
        ct.run_incremental_train_cpcv(args, targets)
        # Should init qlib + calculate_dates but NOT train

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.save_run_state')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_failure(self, mock_table, mock_save, mock_train,
                                  mock_dates, mock_init, mock_env):
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_train.return_value = {"success": False, "error": "GPU OOM"}

        args = MagicMock()
        args.resume = False
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        targets = {"gru_Alpha158": {"yaml_file": "gru_Alpha158.yaml"}}
        ct.run_incremental_train_cpcv(args, targets)

        mock_train.assert_called_once()
        # run_state should be saved (initial + after failed train)
        assert mock_save.call_count >= 2

    @patch('quantpits.utils.train_utils.load_run_state')
    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_resume_skip_completed(self, mock_table, mock_dates,
                                                mock_init, mock_load, mock_env):
        """Lines 252-267: resume skips completed models."""
        ct, _ = mock_env
        mock_load.return_value = {"completed": ["gru_Alpha158"]}
        mock_dates.return_value = _make_cpcv_params()

        args = MagicMock()
        args.resume = True
        args.dry_run = True

        targets = {"gru_Alpha158": {"yaml_file": "gru.yaml"},
                   "lightgbm_Alpha158": {"yaml_file": "lgbm.yaml"}}
        ct.run_incremental_train_cpcv(args, targets)

    @patch('quantpits.utils.train_utils.load_run_state')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_resume_all_done(self, mock_table, mock_load, mock_env):
        """Lines 264-266: resume with all targets already completed."""
        ct, _ = mock_env
        mock_load.return_value = {"completed": ["gru_Alpha158", "lightgbm_Alpha158"]}

        args = MagicMock()
        args.resume = True
        args.dry_run = True

        targets = {"gru_Alpha158": {}, "lightgbm_Alpha158": {}}
        ct.run_incremental_train_cpcv(args, targets)
        # Should print "All target models already completed." and return

    @patch('quantpits.utils.train_utils.load_run_state')
    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_resume_no_prior_state(self, mock_table, mock_dates,
                                                mock_init, mock_load, mock_env):
        """Line 268-269: resume flag but no prior run state."""
        ct, _ = mock_env
        mock_load.return_value = None
        mock_dates.return_value = _make_cpcv_params()

        args = MagicMock()
        args.resume = True
        args.dry_run = True

        targets = {"gru_Alpha158": {}}
        ct.run_incremental_train_cpcv(args, targets)

    @patch('quantpits.utils.train_utils.load_run_state')
    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_resume_empty_state(self, mock_table, mock_dates,
                                             mock_init, mock_load, mock_env):
        """Resume with empty state dict (no 'completed' key)."""
        ct, _ = mock_env
        mock_load.return_value = {}
        mock_dates.return_value = _make_cpcv_params()

        args = MagicMock()
        args.resume = True
        args.dry_run = True

        targets = {"gru_Alpha158": {}}
        ct.run_incremental_train_cpcv(args, targets)

    @patch('quantpits.utils.train_utils.load_run_state')
    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_resume_state_without_completed_key(self, mock_table,
                                                             mock_dates,
                                                             mock_init,
                                                             mock_load, mock_env):
        """Line 255: state exists but has no 'completed' key."""
        ct, _ = mock_env
        mock_load.return_value = {"started_at": "2026-01-01"}

        args = MagicMock()
        args.resume = True
        args.dry_run = True

        targets = {"gru_Alpha158": {}}
        ct.run_incremental_train_cpcv(args, targets)

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_not_purged_cv_exits(self, mock_table, mock_dates,
                                              mock_init, mock_env):
        """Lines 294-296: data_slice_mode is not purged_cv -> sys.exit(1)."""
        ct, _ = mock_env
        mock_dates.return_value = {"anchor_date": "2026-01-01", "freq": "week",
                                   "data_slice_mode": "slide"}

        args = MagicMock()
        args.resume = False
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        targets = {"gru_Alpha158": {"yaml_file": "gru.yaml"}}
        with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
            with pytest.raises(SystemExit):
                ct.run_incremental_train_cpcv(args, targets)
            mock_exit.assert_called_once_with(1)

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.merge_performance_file')
    @patch('quantpits.utils.train_utils.save_run_state')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_mixed_results(self, mock_table, mock_save, mock_perf,
                                        mock_merge, mock_train, mock_dates,
                                        mock_init, mock_env):
        """Mixed success/failure across models."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_train.side_effect = [
            {"success": True, "record_id": "rid1", "performance": {"ICIR": 0.3}},
            {"success": False, "error": "GPU OOM"},
        ]

        args = MagicMock()
        args.resume = False
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        targets = {"gru_Alpha158": {"yaml_file": "gru.yaml"},
                   "lightgbm_Alpha158": {"yaml_file": "lgbm.yaml"}}
        ct.run_incremental_train_cpcv(args, targets)

        mock_merge.assert_called_once()

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.merge_performance_file')
    @patch('quantpits.utils.train_utils.save_run_state')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_no_merge_when_all_fail(self, mock_table, mock_save,
                                                 mock_perf, mock_merge,
                                                 mock_train, mock_dates,
                                                 mock_init, mock_env):
        """Line 348: new_records['models'] is empty -> no merge call."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_train.return_value = {"success": False, "error": "All failed"}

        args = MagicMock()
        args.resume = False
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        targets = {"gru_Alpha158": {"yaml_file": "gru.yaml"}}
        ct.run_incremental_train_cpcv(args, targets)

        mock_merge.assert_not_called()
        mock_perf.assert_not_called()

    @patch('quantpits.utils.train_utils.load_run_state')
    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.merge_performance_file')
    @patch('quantpits.utils.train_utils.save_run_state')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_resume_train_remaining(self, mock_table, mock_save,
                                                 mock_perf, mock_merge,
                                                 mock_train, mock_dates,
                                                 mock_init, mock_load, mock_env):
        """Resume mode: some completed, train only the rest."""
        ct, _ = mock_env
        mock_load.return_value = {"completed": ["gru_Alpha158"]}
        mock_dates.return_value = _make_cpcv_params()
        mock_train.return_value = {"success": True, "record_id": "rid1",
                                    "performance": {"ICIR": 0.3}}

        args = MagicMock()
        args.resume = True
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        targets = {"gru_Alpha158": {"yaml_file": "gru.yaml"},
                   "lightgbm_Alpha158": {"yaml_file": "lgbm.yaml"}}
        ct.run_incremental_train_cpcv(args, targets)

        assert mock_train.call_count == 1
        call_name = mock_train.call_args[0][0]
        assert call_name == "lightgbm_Alpha158"

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.train_cpcv_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.merge_performance_file')
    @patch('quantpits.utils.train_utils.save_run_state')
    @patch('quantpits.utils.train_utils.print_model_table')
    def test_incremental_summary_with_failures(self, mock_table, mock_save,
                                                mock_perf, mock_merge,
                                                mock_train, mock_dates,
                                                mock_init, mock_env):
        """Lines 360-369: summary includes failed models section."""
        ct, _ = mock_env
        mock_dates.return_value = _make_cpcv_params()
        mock_train.side_effect = [
            {"success": True, "record_id": "rid1", "performance": {"ICIR": 0.3}},
            {"success": False, "error": "Error: " + "x" * 100},
        ]

        args = MagicMock()
        args.resume = False
        args.dry_run = False
        args.experiment_name = None
        args.no_pretrain = False

        targets = {"gru_Alpha158": {"yaml_file": "gru.yaml"},
                   "lightgbm_Alpha158": {"yaml_file": "lgbm.yaml"}}
        ct.run_incremental_train_cpcv(args, targets)


# ===================================================================
# 5. run_predict_only_cpcv tests
# ===================================================================

class TestRunPredictOnlyCpcv:
    """CPCV predict-only mode tests."""

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.predict_cpcv_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.filter_models_by_mode')
    def test_predict_only_success(self, mock_filter, mock_merge, mock_predict,
                                   mock_dates, mock_init, mock_env, tmp_path):
        ct, _ = mock_env
        source_file = tmp_path / "records.json"
        source_file.write_text(json.dumps({
            "models": {"gru_Alpha158@cpcv": "rid1", "lightgbm_Alpha158@cpcv": "rid2"},
            "experiment_name": "CPCV_Train",
        }))

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_filter.return_value = {"gru_Alpha158@cpcv": "rid1",
                                     "lightgbm_Alpha158@cpcv": "rid2"}
        mock_predict.return_value = {"success": True, "record_id": "new_rid"}

        args = MagicMock()
        args.dry_run = False
        args.source_records = str(source_file)
        args.experiment_name = None
        args.no_pretrain = False
        args.models = None
        args.tag = None

        ct.run_predict_only_cpcv(args)

        assert mock_predict.call_count == 2
        mock_merge.assert_called_once()

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.filter_models_by_mode')
    def test_predict_only_dry_run(self, mock_filter, mock_dates, mock_init,
                                   mock_env, tmp_path):
        ct, _ = mock_env
        source_file = tmp_path / "records.json"
        source_file.write_text(json.dumps({
            "models": {"gru_Alpha158@cpcv": "rid1"},
            "experiment_name": "CPCV_Train",
        }))

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_filter.return_value = {"gru_Alpha158@cpcv": "rid1"}

        args = MagicMock()
        args.dry_run = True
        args.source_records = str(source_file)
        args.models = None
        args.tag = None
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_predict_only_cpcv(args)

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    def test_predict_only_source_not_found(self, mock_dates, mock_init, mock_env):
        """Line 392-394: source records file does not exist -> sys.exit(1)."""
        ct, _ = mock_env
        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        args = MagicMock()
        args.source_records = "/nonexistent/records.json"

        with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
            with pytest.raises(SystemExit):
                ct.run_predict_only_cpcv(args)
            mock_exit.assert_called_once_with(1)

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.filter_models_by_mode')
    def test_predict_only_no_cpcv_models(self, mock_filter, mock_dates, mock_init,
                                          mock_env, tmp_path):
        """Line 402-404: no @cpcv models in source records."""
        ct, _ = mock_env
        source_file = tmp_path / "records.json"
        source_file.write_text(json.dumps({
            "models": {"gru_Alpha158": "rid1"},  # No @cpcv suffix
            "experiment_name": "Train",
        }))

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_filter.return_value = {}

        args = MagicMock()
        args.dry_run = False
        args.source_records = str(source_file)
        args.models = None
        args.tag = None
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_predict_only_cpcv(args)  # Should print warning, return

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.predict_cpcv_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.filter_models_by_mode')
    def test_predict_only_filter_by_models(self, mock_filter, mock_merge,
                                            mock_predict, mock_dates, mock_init,
                                            mock_env, tmp_path):
        """Line 408-411: filter by --models."""
        ct, _ = mock_env
        source_file = tmp_path / "records.json"
        source_file.write_text(json.dumps({
            "models": {"gru_Alpha158@cpcv": "rid1", "lightgbm_Alpha158@cpcv": "rid2"},
            "experiment_name": "CPCV_Train",
        }))

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_filter.return_value = {"gru_Alpha158@cpcv": "rid1",
                                     "lightgbm_Alpha158@cpcv": "rid2"}
        mock_predict.return_value = {"success": True, "record_id": "new_rid"}

        args = MagicMock()
        args.dry_run = False
        args.source_records = str(source_file)
        args.experiment_name = None
        args.no_pretrain = False
        args.models = "gru_Alpha158"
        args.tag = None

        ct.run_predict_only_cpcv(args)
        assert mock_predict.call_count == 1

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.predict_cpcv_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.filter_models_by_mode')
    @patch('quantpits.utils.train_utils.get_models_by_filter')
    def test_predict_only_filter_by_tag(self, mock_get_tag, mock_filter,
                                         mock_merge, mock_predict, mock_dates,
                                         mock_init, mock_env, tmp_path):
        """Line 412-416: filter by --tag."""
        ct, _ = mock_env
        source_file = tmp_path / "records.json"
        source_file.write_text(json.dumps({
            "models": {"gru_Alpha158@cpcv": "rid1", "xgb_Alpha158@cpcv": "rid3"},
            "experiment_name": "CPCV_Train",
        }))

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_filter.return_value = {"gru_Alpha158@cpcv": "rid1",
                                     "xgb_Alpha158@cpcv": "rid3"}
        mock_get_tag.return_value = {"gru_Alpha158": {}}
        mock_predict.return_value = {"success": True, "record_id": "new_rid"}

        args = MagicMock()
        args.dry_run = False
        args.source_records = str(source_file)
        args.experiment_name = None
        args.no_pretrain = False
        args.models = None
        args.tag = "ts"

        ct.run_predict_only_cpcv(args)
        assert mock_predict.call_count == 1

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.filter_models_by_mode')
    def test_predict_only_no_models_match_filter(self, mock_filter, mock_dates,
                                                  mock_init, mock_env, tmp_path):
        """Line 418-420: no models match selection after filtering."""
        ct, _ = mock_env
        source_file = tmp_path / "records.json"
        source_file.write_text(json.dumps({
            "models": {"gru_Alpha158@cpcv": "rid1"},
            "experiment_name": "CPCV_Train",
        }))

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_filter.return_value = {"gru_Alpha158@cpcv": "rid1"}

        args = MagicMock()
        args.dry_run = False
        args.source_records = str(source_file)
        args.experiment_name = None
        args.no_pretrain = False
        args.models = "nonexistent_model"
        args.tag = None

        ct.run_predict_only_cpcv(args)  # Should print message and return

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.predict_cpcv_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.filter_models_by_mode')
    def test_predict_only_mixed_results(self, mock_filter, mock_merge, mock_predict,
                                         mock_dates, mock_init, mock_env, tmp_path):
        """Lines 457-463: some predictions succeed, some fail."""
        ct, _ = mock_env
        source_file = tmp_path / "records.json"
        source_file.write_text(json.dumps({
            "models": {"gru_Alpha158@cpcv": "rid1", "lightgbm_Alpha158@cpcv": "rid2"},
            "experiment_name": "CPCV_Train",
        }))

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_filter.return_value = {"gru_Alpha158@cpcv": "rid1",
                                     "lightgbm_Alpha158@cpcv": "rid2"}
        mock_predict.side_effect = [
            {"success": True, "record_id": "new_rid1"},
            {"success": False, "error": "Model load error"},
        ]

        args = MagicMock()
        args.dry_run = False
        args.source_records = str(source_file)
        args.experiment_name = None
        args.no_pretrain = False
        args.models = None
        args.tag = None

        ct.run_predict_only_cpcv(args)
        mock_merge.assert_called_once()

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.predict_cpcv_model')
    @patch('quantpits.utils.train_utils.merge_train_records')
    @patch('quantpits.utils.train_utils.filter_models_by_mode')
    def test_predict_only_custom_source(self, mock_filter, mock_merge, mock_predict,
                                         mock_dates, mock_init, mock_env, tmp_path):
        """Line 391: custom --source-records path."""
        ct, _ = mock_env
        custom_file = tmp_path / "custom_records.json"
        custom_file.write_text(json.dumps({
            "models": {"gru_Alpha158@cpcv": "rid1"},
            "experiment_name": "OldCPCV",
        }))

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_filter.return_value = {"gru_Alpha158@cpcv": "rid1"}
        mock_predict.return_value = {"success": True, "record_id": "new_rid"}

        args = MagicMock()
        args.dry_run = False
        args.source_records = str(custom_file)
        args.experiment_name = None
        args.no_pretrain = False
        args.models = None
        args.tag = None

        ct.run_predict_only_cpcv(args)
        mock_predict.assert_called_once()

    @patch('quantpits.utils.env.init_qlib')
    @patch('quantpits.utils.train_utils.calculate_dates')
    @patch('quantpits.utils.train_utils.filter_models_by_mode')
    def test_predict_only_empty_models_dict(self, mock_filter, mock_dates,
                                             mock_init, mock_env, tmp_path):
        """Line 399: source records has empty models dict -> no @cpcv."""
        ct, _ = mock_env
        source_file = tmp_path / "records.json"
        source_file.write_text(json.dumps({
            "models": {},
            "experiment_name": "Empty",
        }))

        mock_dates.return_value = {"anchor_date": "2026-01-08", "freq": "week"}
        mock_filter.return_value = {}

        args = MagicMock()
        args.dry_run = False
        args.source_records = str(source_file)
        args.models = None
        args.tag = None
        args.experiment_name = None
        args.no_pretrain = False

        ct.run_predict_only_cpcv(args)


# ===================================================================
# 6. main() dispatcher tests
# ===================================================================

class TestMain:
    """Test the main() dispatch function for all modes."""

    def test_main_list_no_filters(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--list']):
            with patch('quantpits.utils.train_utils.print_model_table') as mock_print:
                with patch('quantpits.utils.train_utils.load_model_registry',
                           return_value={}):
                    ct.main()
                    mock_print.assert_called_once()

    def test_main_list_with_filters(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--list', '--tag', 'ts']):
            with patch('quantpits.utils.train_utils.print_model_table') as mock_print:
                ct.main()
                mock_print.assert_called_once()

    def test_main_show_state_exists(self, mock_env):
        ct, _ = mock_env
        state = {"started_at": "2026-01-01", "completed": ["m1"]}
        with patch.object(sys, 'argv', ['script.py', '--show-state']):
            with patch('quantpits.utils.train_utils.load_run_state',
                       return_value=state):
                ct.main()

    def test_main_show_state_empty(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--show-state']):
            with patch('quantpits.utils.train_utils.load_run_state',
                       return_value=None):
                ct.main()

    def test_main_clear_state(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--clear-state']):
            with patch('quantpits.utils.train_utils.clear_run_state') as mock_clear:
                ct.main()
                mock_clear.assert_called_once()

    def test_main_full_mode(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--full']):
            with patch('quantpits.scripts.cv_train.run_full_train_cpcv') as mock_run:
                ct.main()
                mock_run.assert_called_once()

    def test_main_predict_only_mode(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--predict-only', '--all-enabled']):
            with patch('quantpits.scripts.cv_train.run_predict_only_cpcv') as mock_run:
                ct.main()
                mock_run.assert_called_once()

    def test_main_incremental_default(self, mock_env):
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--models', 'gru_Alpha158']):
            with patch('quantpits.utils.train_utils.resolve_target_models',
                       return_value={"gru_Alpha158": {}}):
                with patch('quantpits.scripts.cv_train.run_incremental_train_cpcv') as mock_run:
                    ct.main()
                    mock_run.assert_called_once()

    def test_main_incremental_no_targets(self, mock_env):
        """Line 517-518: incremental mode with no target selection -> sys.exit(1)."""
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py']):
            with patch('quantpits.utils.train_utils.resolve_target_models',
                       return_value=None):
                with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
                    with pytest.raises(SystemExit):
                        ct.main()
                    mock_exit.assert_called_once_with(1)

    def test_main_all_skipped(self, mock_env):
        """Lines 517-518: --skip removes all selected models -> sys.exit(1)."""
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--models', 'gru_Alpha158',
                                        '--skip', 'gru_Alpha158']):
            with patch('quantpits.utils.train_utils.resolve_target_models',
                       return_value={"gru_Alpha158": {}}):
                with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
                    with pytest.raises(SystemExit):
                        ct.main()
                    mock_exit.assert_called_once_with(1)

    def test_main_predict_only_precedence_over_full(self, mock_env):
        """--predict-only checked before --full."""
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--predict-only', '--full']):
            with patch('quantpits.scripts.cv_train.run_predict_only_cpcv') as mock_predict:
                with patch('quantpits.scripts.cv_train.run_full_train_cpcv') as mock_full:
                    ct.main()
                    mock_predict.assert_called_once()
                    mock_full.assert_not_called()

    def test_main_list_precedence(self, mock_env):
        """--list is checked before any training mode."""
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--list', '--full']):
            with patch('quantpits.utils.train_utils.print_model_table'):
                with patch('quantpits.utils.train_utils.load_model_registry',
                           return_value={}):
                    with patch('quantpits.scripts.cv_train.run_full_train_cpcv') as mock_full:
                        ct.main()
                        mock_full.assert_not_called()

    def test_main_show_state_precedence(self, mock_env):
        """--show-state checked before training modes."""
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--show-state', '--models', 'm1']):
            with patch('quantpits.utils.train_utils.load_run_state', return_value=None):
                with patch('quantpits.scripts.cv_train.run_incremental_train_cpcv') as mock_run:
                    ct.main()
                    mock_run.assert_not_called()

    def test_main_clear_state_precedence(self, mock_env):
        """--clear-state checked before training modes."""
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--clear-state', '--full']):
            with patch('quantpits.utils.train_utils.clear_run_state'):
                with patch('quantpits.scripts.cv_train.run_full_train_cpcv') as mock_full:
                    ct.main()
                    mock_full.assert_not_called()


# ===================================================================
# 7. Edge cases
# ===================================================================

class TestEdgeCases:
    """Tricky boundary conditions."""

    @patch('quantpits.utils.train_utils.print_model_table')
    def test_main_empty_sys_argv_exits(self, mock_table, mock_env):
        """main() with no model selection flags -> sys.exit(1)."""
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py']):
            with patch('quantpits.utils.train_utils.resolve_target_models',
                       return_value=None):
                with patch('sys.exit', side_effect=SystemExit(1)) as mock_exit:
                    with pytest.raises(SystemExit):
                        ct.main()
                    mock_exit.assert_called_once_with(1)

    @patch('quantpits.utils.train_utils.print_model_table')
    def test_main_list_with_filters_present(self, mock_table, mock_env):
        """--list with --algorithm filter delegates to _resolve_targets."""
        ct, _ = mock_env
        with patch.object(sys, 'argv', ['script.py', '--list', '--algorithm', 'gru']):
            with patch('quantpits.utils.train_utils.resolve_target_models',
                       return_value={"gru_Alpha158": {}}):
                ct.main()
                mock_table.assert_called_once()
