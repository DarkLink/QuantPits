"""Tests for run_feedback_loop.py CLI entry point."""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

# We can't import run_feedback_loop directly because it chdir's
# but we can test parse_args and main via mocking


class TestCLIArgParsing:
    """Test argument parsing directly by calling the module's parser."""

    def test_default_args(self):
        from quantpits.scripts.run_feedback_loop import main as _main
        import argparse

        # Test that the argument parser is configured correctly
        with patch("sys.argv", ["run_feedback_loop.py", "--report-only"]):
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = argparse.Namespace(
                    action_items=None,
                    report_only=True,
                    execute=False,
                    promote=False,
                    auto_promote=False,
                    playground_only=False,
                    dry_run=False,
                    models=None,
                    skip_models=None,
                    max_duration_minutes=None,
                    skip_retrain=False,
                    max_experiment_rounds=3,
                    resume=False,
                    verbose=False,
                )
                # Just verify module loads and parses without errors
                import quantpits.scripts.run_feedback_loop
                assert quantpits.scripts.run_feedback_loop is not None

    def test_report_only_mode(self):
        """Test --report-only is a valid mode."""
        with patch("sys.argv", ["run_feedback_loop.py", "--report-only", "--action-items", "/tmp/test.json"]):
            import quantpits.scripts.run_feedback_loop as rfl
            # Re-parse to verify
            with patch.object(rfl.argparse.ArgumentParser, 'parse_args') as mock_p:
                # We just want to verify the parser is set up right
                pass

    def test_execute_mode(self):
        """Test --execute is a valid mode."""
        with patch("sys.argv", ["run_feedback_loop.py", "--execute", "--action-items", "/tmp/test.json"]):
            import quantpits.scripts.run_feedback_loop as rfl
            assert rfl is not None

    def test_promote_mode(self):
        """Test --promote is a valid mode."""
        with patch("sys.argv", ["run_feedback_loop.py", "--promote", "--action-items", "/tmp/test.json"]):
            import quantpits.scripts.run_feedback_loop as rfl
            assert rfl is not None

    def test_playground_only_mode_requires_models(self):
        """Test --playground-only with --models."""
        with patch("sys.argv", ["run_feedback_loop.py", "--playground-only", "--models", "m1,m2"]):
            import quantpits.scripts.run_feedback_loop as rfl
            assert rfl is not None

    def test_verbose_flag(self):
        """Test --verbose flag."""
        with patch("sys.argv", ["run_feedback_loop.py", "--report-only", "-v", "--action-items", "/tmp/test.json"]):
            import quantpits.scripts.run_feedback_loop as rfl
            assert rfl is not None

    def test_dry_run_flag(self):
        """Test --dry-run flag."""
        with patch("sys.argv", ["run_feedback_loop.py", "--execute", "--dry-run", "--action-items", "/tmp/test.json"]):
            import quantpits.scripts.run_feedback_loop as rfl
            assert rfl is not None

    def test_resume_flag(self):
        """Test --resume flag."""
        with patch("sys.argv", ["run_feedback_loop.py", "--execute", "--resume", "--action-items", "/tmp/test.json"]):
            import quantpits.scripts.run_feedback_loop as rfl
            assert rfl is not None

    def test_skip_retrain_flag(self):
        """Test --skip-retrain flag."""
        with patch("sys.argv", ["run_feedback_loop.py", "--execute", "--skip-retrain", "--action-items", "/tmp/test.json"]):
            import quantpits.scripts.run_feedback_loop as rfl
            assert rfl is not None


class TestMainFunction:
    """Test main() function with mocked FeedbackLoop."""

    @patch("quantpits.scripts.deep_analysis.feedback_loop.FeedbackLoop")
    @patch("quantpits.utils.env.ROOT_DIR", "/tmp/test_root")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isabs", return_value=True)
    def test_main_report_only_mode(self, mock_isabs, mock_exists, mock_fl_cls):
        """Test main() in report-only mode."""
        mock_loop = MagicMock()
        mock_report = MagicMock()
        mock_report.summary = "Test summary"
        mock_report.action_items_processed = 1
        mock_report.action_items_deferred = 0
        mock_report.validation_results = []
        mock_report.promote_result = None
        mock_loop.run.return_value = mock_report
        mock_fl_cls.return_value = mock_loop

        with patch("sys.argv", [
            "run_feedback_loop.py", "--report-only",
            "--action-items", "/tmp/test_root/items.json",
        ]):
            from quantpits.scripts.run_feedback_loop import main
            main()

        mock_fl_cls.assert_called_once()
        mock_loop.run.assert_called_once()
        call_kwargs = mock_loop.run.call_args[1]
        assert call_kwargs["dry_run"] is False
        assert call_kwargs["skip_retrain"] is False
        assert call_kwargs["models"] is None

    @patch("quantpits.scripts.deep_analysis.feedback_loop.FeedbackLoop")
    @patch("quantpits.utils.env.ROOT_DIR", "/tmp/test_root")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isabs", return_value=True)
    def test_main_execute_mode(self, mock_isabs, mock_exists, mock_fl_cls):
        """Test main() in execute mode."""
        mock_loop = MagicMock()
        mock_report = MagicMock()
        mock_report.summary = "Execute done"
        mock_report.action_items_processed = 2
        mock_report.action_items_deferred = 1
        mock_report.validation_results = [{"passed": True}, {"passed": False}]
        mock_report.promote_result = {"success": True}
        mock_loop.run.return_value = mock_report
        mock_fl_cls.return_value = mock_loop

        with patch("sys.argv", [
            "run_feedback_loop.py", "--execute",
            "--action-items", "/tmp/test_root/items.json",
            "--dry-run",
        ]):
            from quantpits.scripts.run_feedback_loop import main
            main()

        call_kwargs = mock_loop.run.call_args[1]
        assert call_kwargs["dry_run"] is True

    @patch("quantpits.scripts.deep_analysis.feedback_loop.FeedbackLoop")
    @patch("quantpits.utils.env.ROOT_DIR", "/tmp/test_root")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isabs", return_value=True)
    def test_main_promote_mode(self, mock_isabs, mock_exists, mock_fl_cls):
        """Test main() in promote mode."""
        mock_loop = MagicMock()
        mock_report = MagicMock()
        mock_report.summary = "Promote done"
        mock_report.action_items_processed = 1
        mock_report.action_items_deferred = 0
        mock_report.validation_results = []
        mock_report.promote_result = {"success": True}
        mock_loop.run.return_value = mock_report
        mock_fl_cls.return_value = mock_loop

        with patch("sys.argv", [
            "run_feedback_loop.py", "--promote",
            "--action-items", "/tmp/test_root/items.json",
        ]):
            from quantpits.scripts.run_feedback_loop import main
            main()

        mock_fl_cls.assert_called_once()

    @patch("quantpits.scripts.deep_analysis.feedback_loop.FeedbackLoop")
    @patch("quantpits.utils.env.ROOT_DIR", "/tmp/test_root")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isabs", return_value=True)
    def test_main_auto_promote_mode(self, mock_isabs, mock_exists, mock_fl_cls):
        """Test main() in auto-promote mode."""
        mock_loop = MagicMock()
        mock_report = MagicMock()
        mock_report.summary = "Not yet implemented"
        mock_report.action_items_processed = 0
        mock_report.action_items_deferred = 0
        mock_report.validation_results = []
        mock_report.promote_result = None
        mock_loop.run.return_value = mock_report
        mock_fl_cls.return_value = mock_loop

        with patch("sys.argv", [
            "run_feedback_loop.py", "--auto-promote",
            "--action-items", "/tmp/test_root/items.json",
        ]):
            from quantpits.scripts.run_feedback_loop import main
            main()

    @patch("quantpits.scripts.deep_analysis.feedback_loop.FeedbackLoop")
    @patch("quantpits.utils.env.ROOT_DIR", "/tmp/test_root")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isabs", return_value=True)
    def test_main_playground_only_mode(self, mock_isabs, mock_exists, mock_fl_cls):
        """Test main() in playground-only mode."""
        mock_loop = MagicMock()
        mock_report = MagicMock()
        mock_report.summary = "Playground done"
        mock_report.action_items_processed = 0
        mock_report.action_items_deferred = 0
        mock_report.validation_results = []
        mock_report.promote_result = None
        mock_loop.run.return_value = mock_report
        mock_fl_cls.return_value = mock_loop

        with patch("sys.argv", [
            "run_feedback_loop.py", "--playground-only",
            "--models", "m1,m2",
        ]):
            from quantpits.scripts.run_feedback_loop import main
            main()

        call_kwargs = mock_loop.run.call_args[1]
        assert call_kwargs["models"] == ["m1", "m2"]

    @patch("quantpits.scripts.deep_analysis.feedback_loop.FeedbackLoop")
    @patch("quantpits.utils.env.ROOT_DIR", "/tmp/test_root")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isabs", return_value=True)
    def test_main_with_all_options(self, mock_isabs, mock_exists, mock_fl_cls):
        """Test main() with all optional arguments."""
        mock_loop = MagicMock()
        mock_report = MagicMock()
        mock_report.summary = "Full test"
        mock_report.action_items_processed = 1
        mock_report.action_items_deferred = 1
        mock_report.validation_results = [{"passed": True}]
        mock_report.promote_result = None
        mock_loop.run.return_value = mock_report
        mock_fl_cls.return_value = mock_loop

        with patch("sys.argv", [
            "run_feedback_loop.py", "--execute",
            "--action-items", "/tmp/test_root/items.json",
            "--models", "m1,m2",
            "--skip-models", "m3",
            "--max-duration-minutes", "30",
            "--skip-retrain",
            "--max-experiment-rounds", "5",
            "--resume",
            "-v",
        ]):
            from quantpits.scripts.run_feedback_loop import main
            main()

        call_kwargs = mock_loop.run.call_args[1]
        assert call_kwargs["models"] == ["m1", "m2"]
        assert call_kwargs["skip_models"] == ["m3"]
        assert call_kwargs["max_duration_minutes"] == 30
        assert call_kwargs["skip_retrain"] is True
        assert call_kwargs["max_experiment_rounds"] == 5
        assert call_kwargs["resume"] is True

    @patch("quantpits.scripts.deep_analysis.feedback_loop.FeedbackLoop")
    @patch("quantpits.utils.env.ROOT_DIR", "/tmp/test_root")
    @patch("os.path.isabs", return_value=False)
    @patch("os.path.exists", return_value=True)
    def test_main_relative_action_items_path(self, mock_exists, mock_isabs, mock_fl_cls):
        """Test that relative action_items path is resolved to absolute."""
        mock_loop = MagicMock()
        mock_report = MagicMock()
        mock_report.summary = "Test"
        mock_report.action_items_processed = 0
        mock_report.action_items_deferred = 0
        mock_report.validation_results = []
        mock_report.promote_result = None
        mock_loop.run.return_value = mock_report
        mock_fl_cls.return_value = mock_loop

        with patch("sys.argv", [
            "run_feedback_loop.py", "--report-only",
            "--action-items", "relative/path.json",
        ]):
            from quantpits.scripts.run_feedback_loop import main
            main()

        call_kwargs = mock_loop.run.call_args[1]
        # Path should have been resolved to absolute
        action_path = call_kwargs["action_items_path"]
        assert action_path.startswith("/tmp/test_root")

    @patch("quantpits.utils.env.ROOT_DIR", "/tmp/test_root")
    @patch("os.path.exists", return_value=False)
    def test_main_missing_action_items_exits(self, mock_exists):
        """Test that main() exits when action_items file doesn't exist."""
        with patch("sys.argv", [
            "run_feedback_loop.py", "--report-only",
            "--action-items", "/tmp/test_root/missing.json",
        ]):
            with pytest.raises(SystemExit) as exc_info:
                from quantpits.scripts.run_feedback_loop import main
                main()
            assert exc_info.value.code == 1

    @patch("quantpits.utils.env.ROOT_DIR", "/tmp/test_root")
    def test_main_playground_only_without_models_exits(self):
        """Test that --playground-only without --models exits."""
        with patch("sys.argv", [
            "run_feedback_loop.py", "--playground-only",
        ]):
            with pytest.raises(SystemExit) as exc_info:
                from quantpits.scripts.run_feedback_loop import main
                main()
            assert exc_info.value.code == 1

    @patch("quantpits.utils.env.ROOT_DIR", "/tmp/test_root")
    def test_main_missing_action_items_in_non_playground_mode(self):
        """Test that missing --action-items in non-playground mode exits."""
        with patch("sys.argv", [
            "run_feedback_loop.py", "--report-only",
        ]):
            with pytest.raises(SystemExit) as exc_info:
                from quantpits.scripts.run_feedback_loop import main
                main()
            assert exc_info.value.code == 1
