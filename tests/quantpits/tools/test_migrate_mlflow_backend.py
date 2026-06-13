"""
Tests for migrate_mlflow_backend.py — MLflow file-store → SQLite migration tool.

Covers: _mlflow_version(), _has_experiment_data(), migrate(), main().
"""

import os
import sys
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from quantpits.tools.migrate_mlflow_backend import (
    _has_experiment_data,
    _mlflow_version,
    main,
    migrate,
)


# ---------------------------------------------------------------------------
# _mlflow_version
# ---------------------------------------------------------------------------

class TestMlflowVersion:
    def test_normal_version(self):
        with patch("mlflow.__version__", "3.0.1"):
            assert _mlflow_version() == (3, 0, 1)

    def test_nonstandard_version(self):
        with patch("mlflow.__version__", "2.14.0.dev0"):
            assert _mlflow_version() == (2, 14, 0)


# ---------------------------------------------------------------------------
# _has_experiment_data
# ---------------------------------------------------------------------------

class TestHasExperimentData:
    def test_true_with_subdirs(self, tmp_path):
        mlruns = tmp_path / "mlruns"
        mlruns.mkdir()
        (mlruns / "0").mkdir()
        assert _has_experiment_data(str(mlruns)) is True

    def test_false_only_gitkeep(self, tmp_path):
        mlruns = tmp_path / "mlruns"
        mlruns.mkdir()
        (mlruns / ".gitkeep").touch()
        assert _has_experiment_data(str(mlruns)) is False

    def test_false_empty_dir(self, tmp_path):
        mlruns = tmp_path / "mlruns"
        mlruns.mkdir()
        assert _has_experiment_data(str(mlruns)) is False

    def test_false_no_dir(self, tmp_path):
        assert _has_experiment_data(str(tmp_path / "nonexistent")) is False

    def test_false_path_is_file(self, tmp_path):
        f = tmp_path / "mlruns"
        f.touch()
        assert _has_experiment_data(str(f)) is False


# ---------------------------------------------------------------------------
# migrate()
# ---------------------------------------------------------------------------

class TestMigrate:
    @pytest.fixture
    def ws_with_mlruns(self, tmp_path):
        """Create a workspace with mlruns/ containing experiment data."""
        ws = tmp_path / "ws"
        mlruns = ws / "mlruns"
        mlruns.mkdir(parents=True)
        (mlruns / "1").mkdir()
        (mlruns / "0").mkdir()
        config = ws / "config"
        config.mkdir()
        return str(ws)

    def test_missing_workspace(self, tmp_path):
        result = migrate(str(tmp_path / "no_such_dir"))
        assert result != 0

    def test_no_experiment_data(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        mlruns = ws / "mlruns"
        mlruns.mkdir()
        (mlruns / ".gitkeep").touch()
        result = migrate(str(ws))
        assert result == 0

    def test_old_mlflow_version(self, ws_with_mlruns):
        with patch("quantpits.tools.migrate_mlflow_backend._mlflow_version",
                   return_value=(2, 14, 0)):
            result = migrate(ws_with_mlruns)
            assert result == 1

    def test_existing_db_warning(self, ws_with_mlruns, capsys):
        db_path = ws_with_mlruns + "/mlflow.db"
        with open(db_path, "w") as f:
            f.write("fake db")
        with patch("quantpits.tools.migrate_mlflow_backend._mlflow_version",
                   return_value=(3, 0, 1)), \
             patch("subprocess.run") as mock_run, \
             patch("builtins.input", return_value="y"):
            mock_run.return_value = MagicMock(returncode=0)
            migrate(ws_with_mlruns)
            captured = capsys.readouterr()
            assert "already exists" in captured.out

    def test_confirm_yes(self, ws_with_mlruns):
        with patch("quantpits.tools.migrate_mlflow_backend._mlflow_version",
                   return_value=(3, 0, 1)), \
             patch("subprocess.run") as mock_run, \
             patch("builtins.input", return_value="y"), \
             patch("shutil.move"):
            mock_run.return_value = MagicMock(returncode=0)
            result = migrate(ws_with_mlruns)
            assert result == 0

    def test_confirm_no(self, ws_with_mlruns, capsys):
        with patch("quantpits.tools.migrate_mlflow_backend._mlflow_version",
                   return_value=(3, 0, 1)), \
             patch("subprocess.run"), \
             patch("builtins.input", return_value="n"):
            result = migrate(ws_with_mlruns)
            assert result == 0
            captured = capsys.readouterr()
            assert "Aborted" in captured.out

    def test_yes_flag_skips_prompt(self, ws_with_mlruns):
        with patch("quantpits.tools.migrate_mlflow_backend._mlflow_version",
                   return_value=(3, 0, 1)), \
             patch("subprocess.run") as mock_run, \
             patch("shutil.move"):
            mock_run.return_value = MagicMock(returncode=0)
            result = migrate(ws_with_mlruns, yes=True)
            assert result == 0

    def test_subprocess_failure(self, ws_with_mlruns):
        with patch("quantpits.tools.migrate_mlflow_backend._mlflow_version",
                   return_value=(3, 0, 1)), \
             patch("subprocess.run") as mock_run, \
             patch("shutil.move"):
            mock_run.return_value = MagicMock(returncode=1)
            result = migrate(ws_with_mlruns, yes=True)
            assert result == 1

    def test_subprocess_returns_nonzero_code(self, ws_with_mlruns):
        with patch("quantpits.tools.migrate_mlflow_backend._mlflow_version",
                   return_value=(3, 0, 1)), \
             patch("subprocess.run") as mock_run, \
             patch("shutil.move"):
            mock_run.return_value = MagicMock(returncode=42)
            result = migrate(ws_with_mlruns, yes=True)
            assert result == 42

    def test_full_success_renames_mlruns(self, ws_with_mlruns):
        with patch("quantpits.tools.migrate_mlflow_backend._mlflow_version",
                   return_value=(3, 0, 1)), \
             patch("subprocess.run") as mock_run, \
             patch("shutil.move") as mock_move:
            mock_run.return_value = MagicMock(returncode=0)
            result = migrate(ws_with_mlruns, yes=True)
            assert result == 0
            mock_move.assert_called_once()
            # First arg: mlruns dir
            assert mock_move.call_args[0][0] == os.path.join(ws_with_mlruns, "mlruns")

    def test_full_success_output(self, ws_with_mlruns, capsys):
        with patch("quantpits.tools.migrate_mlflow_backend._mlflow_version",
                   return_value=(3, 0, 1)), \
             patch("subprocess.run") as mock_run, \
             patch("shutil.move"):
            mock_run.return_value = MagicMock(returncode=0)
            migrate(ws_with_mlruns, yes=True)
            captured = capsys.readouterr()
            assert "Migration complete" in captured.out
            assert "sqlite:///" in captured.out


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_required_workspace_arg(self, tmp_path):
        ws = str(tmp_path)
        with patch("sys.argv", ["migrate_mlflow_backend.py", "--workspace", ws]), \
             patch("quantpits.tools.migrate_mlflow_backend.migrate") as mock_migrate:
            mock_migrate.return_value = 0
            # main() calls sys.exit(migrate(...)), so catch SystemExit
            with pytest.raises(SystemExit):
                main()
            mock_migrate.assert_called_once_with(ws, yes=False)

    def test_yes_flag(self, tmp_path):
        ws = str(tmp_path)
        with patch("sys.argv", ["migrate_mlflow_backend.py", "--workspace", ws, "--yes"]), \
             patch("quantpits.tools.migrate_mlflow_backend.migrate") as mock_migrate:
            mock_migrate.return_value = 0
            with pytest.raises(SystemExit):
                main()
            mock_migrate.assert_called_once_with(ws, yes=True)

    def test_short_yes_flag(self, tmp_path):
        ws = str(tmp_path)
        with patch("sys.argv", ["migrate_mlflow_backend.py", "--workspace", ws, "-y"]), \
             patch("quantpits.tools.migrate_mlflow_backend.migrate") as mock_migrate:
            mock_migrate.return_value = 0
            with pytest.raises(SystemExit):
                main()
            mock_migrate.assert_called_once_with(ws, yes=True)
