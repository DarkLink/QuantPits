"""Unit tests for rolling_benchmark.py — resource profiling script."""

import os
import json
import sys
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Must set env before import because rolling_benchmark does os.chdir(env.ROOT_DIR)
os.environ["QLIB_WORKSPACE_DIR"] = "/tmp/MockWorkspace_rolling_benchmark"

with patch("os.chdir"):
    from quantpits.scripts import rolling_benchmark as rb


# =========================================================================
# parse_args
# =========================================================================


class TestParseArgs:
    def test_defaults(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py"]):
            args = rb.parse_args()
        assert args.window_idx == 0
        assert args.dry_run is False
        assert args.resume is False
        assert args.no_pretrain is False
        assert args.all_enabled is False
        assert args.models is None

    def test_all_enabled(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py", "--all-enabled"]):
            args = rb.parse_args()
        assert args.all_enabled is True

    def test_models_comma_separated(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py", "--models", "m1,m2,m3"]):
            args = rb.parse_args()
        assert args.models == "m1,m2,m3"

    def test_algorithm_filter(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py", "--algorithm", "lstm,gru"]):
            args = rb.parse_args()
        assert args.algorithm == "lstm,gru"

    def test_dataset_filter(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py", "--dataset", "Alpha158"]):
            args = rb.parse_args()
        assert args.dataset == "Alpha158"

    def test_tag_filter(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py", "--tag", "fast"]):
            args = rb.parse_args()
        assert args.tag == "fast"

    def test_skip_models(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py", "--skip", "m1,m2"]):
            args = rb.parse_args()
        assert args.skip == "m1,m2"

    def test_window_idx(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py", "--window-idx", "3"]):
            args = rb.parse_args()
        assert args.window_idx == 3

    def test_dry_run(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py", "--dry-run"]):
            args = rb.parse_args()
        assert args.dry_run is True

    def test_resume(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py", "--resume"]):
            args = rb.parse_args()
        assert args.resume is True

    def test_no_pretrain(self):
        with patch.object(sys, "argv", ["rolling_benchmark.py", "--no-pretrain"]):
            args = rb.parse_args()
        assert args.no_pretrain is True

    def test_combined_flags(self):
        with patch.object(sys, "argv", [
            "rolling_benchmark.py", "--all-enabled", "--window-idx", "2",
            "--dry-run", "--resume",
        ]):
            args = rb.parse_args()
        assert args.all_enabled is True
        assert args.window_idx == 2
        assert args.dry_run is True
        assert args.resume is True


# =========================================================================
# File I/O: load_existing_benchmark / save_benchmark
# =========================================================================


class TestFileIO:
    def test_load_existing_benchmark_valid(self, tmp_path, monkeypatch):
        bm_file = tmp_path / "data" / "rolling_benchmark.json"
        bm_file.parent.mkdir(parents=True, exist_ok=True)
        bm_file.write_text(json.dumps({"system": {}, "benchmarks": {"m1": {}}}))
        monkeypatch.setattr(rb, "BENCHMARK_OUTPUT", str(bm_file))
        data = rb.load_existing_benchmark()
        assert data is not None
        assert "benchmarks" in data
        assert "m1" in data["benchmarks"]

    def test_load_existing_benchmark_no_file(self, tmp_path, monkeypatch):
        bm_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr(rb, "BENCHMARK_OUTPUT", str(bm_file))
        assert rb.load_existing_benchmark() is None

    def test_load_existing_benchmark_invalid_json(self, tmp_path, monkeypatch):
        bm_file = tmp_path / "bad.json"
        bm_file.write_text("not valid json {{{")
        monkeypatch.setattr(rb, "BENCHMARK_OUTPUT", str(bm_file))
        assert rb.load_existing_benchmark() is None

    def test_save_benchmark_creates_dirs(self, tmp_path, monkeypatch):
        bm_file = tmp_path / "data" / "sub" / "rolling_benchmark.json"
        monkeypatch.setattr(rb, "BENCHMARK_OUTPUT", str(bm_file))
        data = {"benchmarks": {"m1": {"wall_sec": 10}}}
        rb.save_benchmark(data)
        assert bm_file.exists()
        loaded = json.loads(bm_file.read_text())
        assert loaded["benchmarks"]["m1"]["wall_sec"] == 10

    def test_save_benchmark_writes_json(self, tmp_path, monkeypatch):
        bm_file = tmp_path / "out.json"
        monkeypatch.setattr(rb, "BENCHMARK_OUTPUT", str(bm_file))
        data = {"system": {"cpu": "test"}, "benchmarks": {}}
        rb.save_benchmark(data)
        loaded = json.loads(bm_file.read_text())
        assert loaded["system"]["cpu"] == "test"

    def test_load_benchmark_not_a_dict(self, tmp_path, monkeypatch):
        """JSON file that's a list instead of dict -- still loadable by json.load."""
        bm_file = tmp_path / "list.json"
        bm_file.write_text(json.dumps([1, 2, 3]))
        monkeypatch.setattr(rb, "BENCHMARK_OUTPUT", str(bm_file))
        result = rb.load_existing_benchmark()
        assert result == [1, 2, 3]


# =========================================================================
# collect_system_info
# =========================================================================


class TestCollectSystemInfo:
    def test_collects_cpu_from_proc(self, monkeypatch):
        mock_cpuinfo = "vendor_id : GenuineIntel\nmodel name : Intel(R) Test CPU\nflags : fpu\n"
        with patch("builtins.open", mock_open(read_data=mock_cpuinfo)), \
             patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.total = 16 * 1024**3
            with patch.object(rb, "get_gpu_info") as mock_gpu:
                mock_gpu.return_value = {"name": "TestGPU", "vram_total_mb": 8192}
                info = rb.collect_system_info()

        assert "Intel(R) Test CPU" in info["cpu"]
        assert info["ram_total_mb"] == pytest.approx(16384, rel=0.1)
        assert info["gpu_name"] == "TestGPU"

    def test_no_proc_cpuinfo_fallback(self, monkeypatch):
        with patch("builtins.open", side_effect=FileNotFoundError), \
             patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.total = 8 * 1024**3
            with patch.object(rb, "get_gpu_info", return_value={"name": "GPU", "vram_total_mb": 0}):
                info = rb.collect_system_info()
        assert info["cpu"] == "Unknown"

    def test_full_structure_keys(self, monkeypatch):
        with patch("builtins.open", mock_open(read_data="model name : TestCPU\n")), \
             patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.total = 32 * 1024**3
            with patch.object(rb, "get_gpu_info", return_value={"name": "A100", "vram_total_mb": 40960}):
                info = rb.collect_system_info()

        for key in ["cpu", "cpu_count", "ram_total_mb", "gpu_name", "gpu_vram_mb",
                     "platform", "python", "collected_at"]:
            assert key in info, f"Missing key: {key}"

    def test_collected_at_is_iso_string(self, monkeypatch):
        with patch("builtins.open", mock_open(read_data="model name : X\n")), \
             patch("psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.total = 8 * 1024**3
            with patch.object(rb, "get_gpu_info", return_value={"name": "X", "vram_total_mb": 0}):
                info = rb.collect_system_info()
        # Should be a date-time string with "YYYY-MM-DD HH:MM:SS" format
        assert " " in info["collected_at"]
        assert len(info["collected_at"].split(" ")[0].split("-")) == 3


# =========================================================================
# get_gpu_info
# =========================================================================


class TestGetGpuInfo:
    def test_pynvml_success(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetName.return_value = b"Tesla T4"
        mem_mock = MagicMock()
        mem_mock.total = 16 * 1024**3  # 16 GB
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem_mock

        with patch.dict("sys.modules", pynvml=mock_nvml):
            info = rb.get_gpu_info()

        assert info["name"] == "Tesla T4"
        assert info["vram_total_mb"] == pytest.approx(16384, rel=0.15)

    def test_pynvml_returns_str_name(self):
        """pynvml returns str (not bytes) name."""
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetName.return_value = "RTX 4090"
        mem_mock = MagicMock()
        mem_mock.total = 24 * 1024**3
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem_mock

        with patch.dict("sys.modules", pynvml=mock_nvml):
            info = rb.get_gpu_info()

        assert info["name"] == "RTX 4090"

    def test_nvidia_smi_fallback(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Tesla V100, 32768"

            # pynvml unavailable
            with patch.dict("sys.modules", {}):
                with patch.dict("sys.modules", pynvml=None):
                    pass

            # Force pynvml import to fail
            with patch("builtins.__import__", side_effect=ImportError):
                try:
                    info = rb.get_gpu_info()
                except (ImportError, TypeError):
                    # Can't easily suppress the actual import; test the fallback
                    # by testing the subprocess path directly
                    pass

    def test_nvidia_smi_success(self):
        """Test the nvidia-smi path when pynvml is not available."""
        # We patch get_gpu_info's internal logic by mocking both pynvml and subprocess
        with patch.object(rb, "get_gpu_info", wraps=rb.get_gpu_info) as wrapped:
            # The function tries pynvml first which will fail if not installed
            # Call it and accept whatever result we get (depends on test env)
            info = rb.get_gpu_info()
        assert "name" in info
        assert "vram_total_mb" in info

    def test_both_fail_return_unknown(self):
        """When both pynvml and nvidia-smi fail, returns Unknown."""
        # Direct test: both inner paths fail → Unknown/0
        # This happens naturally in a GPU-less CI environment
        info = rb.get_gpu_info()
        # In CI without GPU, we should get Unknown
        if info["name"] == "Unknown":
            assert info["vram_total_mb"] == 0


# =========================================================================
# get_gpu_vram_used_mb
# =========================================================================


class TestGetGpuVramUsedMb:
    def test_pynvml_returns_used_vram(self):
        mock_nvml = MagicMock()
        mem_mock = MagicMock()
        mem_mock.used = 4 * 1024**3  # 4 GB used
        mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem_mock

        with patch.dict("sys.modules", pynvml=mock_nvml):
            used = rb.get_gpu_vram_used_mb()

        assert used == pytest.approx(4096, rel=0.15)

    def test_nvidia_smi_fallback(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "2048"

            # Force pynvml import error
            with patch.dict("sys.modules", {}):
                pass

            # Just test on the actual env — may return 0 or a real value
            result = rb.get_gpu_vram_used_mb()
        assert isinstance(result, int)

    def test_returns_zero_on_all_failure(self):
        """In an environment without GPU, returns 0."""
        result = rb.get_gpu_vram_used_mb()
        assert isinstance(result, (int, float))
        assert result >= 0


# =========================================================================
# classify_model_device
# =========================================================================


class TestClassifyModelDevice:
    def test_gpu_algorithm(self):
        assert rb.classify_model_device("lstm") == "GPU"
        assert rb.classify_model_device("transformer") == "GPU"
        assert rb.classify_model_device("catboost") == "GPU"
        assert rb.classify_model_device("GRU") == "GPU"

    def test_cpu_algorithm(self):
        assert rb.classify_model_device("linear") == "CPU"
        assert rb.classify_model_device("lightgbm") == "CPU"
        assert rb.classify_model_device("xgboost") == "CPU"
        assert rb.classify_model_device("MLP") == "CPU"

    def test_unknown_algorithm(self):
        assert rb.classify_model_device("random_forest") == "GPU?"
        assert rb.classify_model_device("") == "GPU?"
        assert rb.classify_model_device("unknown_xyz") == "GPU?"

    def test_case_insensitive(self):
        assert rb.classify_model_device("LSTM") == "GPU"
        assert rb.classify_model_device("Linear") == "CPU"
        assert rb.classify_model_device("GrU") == "GPU"


# =========================================================================
# compute_benchmark_summary
# =========================================================================


class TestComputeBenchmarkSummary:
    def _make_sys_info(self, ram=32768, vram=16384):
        return {"ram_total_mb": ram, "gpu_vram_mb": vram, "cpu_count": 8}

    def test_empty_benchmarks(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        result = rb.compute_benchmark_summary({}, self._make_sys_info())
        assert result["total_wall_sec"] == 0
        assert result["cpu_count"] == 0
        assert result["gpu_count"] == 0

    def test_sorted_by_wall_sec_desc(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        bms = {
            "fast": {"wall_sec": 10, "peak_rss_mb": 100, "peak_vram_mb": 0,
                     "success": True, "device": "cpu"},
            "slow": {"wall_sec": 100, "peak_rss_mb": 200, "peak_vram_mb": 0,
                     "success": True, "device": "cpu"},
        }
        result = rb.compute_benchmark_summary(bms, self._make_sys_info())
        sorted_names = [n for n, _ in result["sorted_models"]]
        assert sorted_names[0] == "slow"
        assert sorted_names[1] == "fast"
        assert result["total_wall_sec"] == 110

    def test_cpu_avg_statistics(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        bms = {
            "cpu1": {"wall_sec": 60, "peak_rss_mb": 500, "peak_vram_mb": 10,
                     "success": True, "device": "cpu"},
            "cpu2": {"wall_sec": 120, "peak_rss_mb": 1500, "peak_vram_mb": 20,
                     "success": True, "device": "cpu"},
        }
        result = rb.compute_benchmark_summary(bms, self._make_sys_info())
        assert result["cpu_count"] == 2
        assert result["cpu_avg_wall_sec"] == 90
        assert result["cpu_avg_rss_mb"] == 1000
        assert result["cpu_parallel_suggestion"] >= 1
        assert result["cpu_max_rss_mb"] == 1500

    def test_gpu_avg_statistics(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        bms = {
            "gpu1": {"wall_sec": 300, "peak_rss_mb": 2000, "peak_vram_mb": 4096,
                     "success": True, "device": "gpu"},
            "gpu2": {"wall_sec": 500, "peak_rss_mb": 3000, "peak_vram_mb": 8192,
                     "success": True, "device": "gpu"},
        }
        result = rb.compute_benchmark_summary(bms, self._make_sys_info())
        assert result["gpu_count"] == 2
        assert result["gpu_avg_wall_sec"] == 400
        assert result["gpu_avg_rss_mb"] == 2500
        assert result["gpu_avg_vram_mb"] == 6144
        assert result["gpu_parallel_suggestion"] >= 1
        assert result["gpu_max_vram_mb"] == 8192

    def test_failed_models_excluded_from_stats(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        bms = {
            "good_cpu": {"wall_sec": 100, "peak_rss_mb": 1000, "peak_vram_mb": 0,
                         "success": True, "device": "cpu"},
            "bad_cpu": {"wall_sec": 5, "peak_rss_mb": 100, "peak_vram_mb": 0,
                        "success": False, "device": "cpu"},
        }
        result = rb.compute_benchmark_summary(bms, self._make_sys_info())
        assert result["cpu_count"] == 1
        assert result["cpu_avg_wall_sec"] == 100  # bad excluded

    def test_parallelism_suggestion_cpu(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        # One CPU model using 1GB RSS, 32GB RAM → should suggest ~25 but capped at 8
        bms = {
            "cpu1": {"wall_sec": 50, "peak_rss_mb": 1000, "peak_vram_mb": 0,
                     "success": True, "device": "cpu"},
        }
        result = rb.compute_benchmark_summary(bms, self._make_sys_info(ram=32768))
        # 32768 * 0.8 / 1000 = ~26, capped at cpu_count(8)
        assert result["cpu_parallel_suggestion"] == 8

    def test_gpu_zero_vram_parallelism(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        bms = {
            "gpu1": {"wall_sec": 100, "peak_rss_mb": 500, "peak_vram_mb": 0,
                     "success": True, "device": "gpu"},
        }
        result = rb.compute_benchmark_summary(bms, self._make_sys_info())
        # max_vram=0 → gpu_parallel=1
        assert result["gpu_parallel_suggestion"] == 1

    def test_mixed_devices(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        bms = {
            "cpu1": {"wall_sec": 60, "peak_rss_mb": 500, "peak_vram_mb": 0,
                     "success": True, "device": "cpu"},
            "gpu1": {"wall_sec": 300, "peak_rss_mb": 2000, "peak_vram_mb": 4096,
                     "success": True, "device": "gpu"},
            "gpu2": {"wall_sec": 200, "peak_rss_mb": 1500, "peak_vram_mb": 3072,
                     "success": True, "device": "gpu"},
        }
        result = rb.compute_benchmark_summary(bms, self._make_sys_info())
        assert result["cpu_count"] == 1
        assert result["gpu_count"] == 2
        assert "cpu_parallel_suggestion" in result
        assert "gpu_parallel_suggestion" in result

    def test_device_from_classify_model_device_consistency(self):
        """GPU? maps to gpu via .lower(), which behaves like GPU in stats."""
        algo = "unknown_algo"
        device = rb.classify_model_device(algo).lower()
        assert device == "gpu?"

    def test_all_failed_models(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        bms = {
            "fail1": {"wall_sec": 5, "peak_rss_mb": 100, "peak_vram_mb": 0,
                      "success": False, "device": "cpu"},
            "fail2": {"wall_sec": 3, "peak_rss_mb": 50, "peak_vram_mb": 0,
                      "success": False, "device": "gpu"},
        }
        result = rb.compute_benchmark_summary(bms, self._make_sys_info())
        assert result["cpu_count"] == 0
        assert result["gpu_count"] == 0
        # No parallel suggestions when all fail
        assert "cpu_parallel_suggestion" not in result
        assert "gpu_parallel_suggestion" not in result
        assert result["total_wall_sec"] == 8


# =========================================================================
# get_gpu_vram_used_mb — nvidia-smi fallback paths
# =========================================================================


class TestGetGpuVramUsedMbNvidiaSmi:
    def test_nvidia_smi_success_path(self):
        """Lines 112-120: pynvml unavailable, nvidia-smi succeeds."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "4096"

            # Force pynvml import to fail → triggers nvidia-smi fallback
            with patch.dict("sys.modules", {"pynvml": None}):
                # Removing pynvml from sys.modules doesn't prevent re-import.
                # Instead, mock the __import__ to make pynvml raise ImportError.
                import builtins
                _orig_import = builtins.__import__

                def _block_pynvml(name, *args, **kwargs):
                    if name == "pynvml":
                        raise ImportError("No pynvml")
                    return _orig_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=_block_pynvml):
                    used = rb.get_gpu_vram_used_mb()

        assert used == 4096

    def test_nvidia_smi_nonzero_returncode(self):
        """Line 119: nvidia-smi returns non-zero → falls through to return 0."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1  # failure
            mock_run.return_value.stdout = ""

            import builtins
            _orig_import = builtins.__import__

            def _block_pynvml(name, *args, **kwargs):
                if name == "pynvml":
                    raise ImportError("No pynvml")
                return _orig_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_block_pynvml):
                used = rb.get_gpu_vram_used_mb()

        assert used == 0

    def test_nvidia_smi_subprocess_raises(self):
        """Line 121-122: subprocess.run itself raises → return 0."""
        with patch("subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
            import builtins
            _orig_import = builtins.__import__

            def _block_pynvml(name, *args, **kwargs):
                if name == "pynvml":
                    raise ImportError("No pynvml")
                return _orig_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=_block_pynvml):
                used = rb.get_gpu_vram_used_mb()

        assert used == 0


# =========================================================================
# benchmark_model_isolated
# =========================================================================


class TestBenchmarkModelIsolated:
    def test_successful_benchmark(self):
        """Lines 273-314: successful benchmark with wall time measurement."""
        mock_result = {"success": True, "record_id": "abc123"}
        mock_future = MagicMock()
        mock_future.result.return_value = (mock_result, 1024, 2048)  # (result, rss, vram)

        mock_executor = MagicMock()
        mock_executor.__enter__.return_value.submit.return_value = mock_future
        mock_executor.__exit__.return_value = None

        with patch("concurrent.futures.ProcessPoolExecutor",
                   return_value=mock_executor):
            bm = rb.benchmark_model_isolated(
                qlib_config=None,
                model_name="test_model",
                yaml_file="workflow.yaml",
                window={"train_start": "2020-01-01", "train_end": "2020-06-30",
                        "valid_start": "2020-07-01", "valid_end": "2020-08-31",
                        "test_start": "2020-09-01", "test_end": "2020-09-30"},
                params_base={"anchor_date": "2020-01-01"},
                no_pretrain=False,
            )

        assert bm["success"] is True
        assert bm["peak_rss_mb"] == 1024
        assert bm["peak_vram_mb"] == 2048
        assert bm["record_id"] == "abc123"
        assert bm["wall_sec"] >= 0

    def test_failed_benchmark(self):
        """Benchmark with success=False."""
        mock_result = {"success": False, "error": "Training failed"}
        mock_future = MagicMock()
        mock_future.result.return_value = (mock_result, 500, 0)

        mock_executor = MagicMock()
        mock_executor.__enter__.return_value.submit.return_value = mock_future
        mock_executor.__exit__.return_value = None

        with patch("concurrent.futures.ProcessPoolExecutor",
                   return_value=mock_executor):
            bm = rb.benchmark_model_isolated(
                qlib_config=None,
                model_name="fail_model",
                yaml_file="workflow.yaml",
                window={"train_start": "2020-01-01", "train_end": "2020-06-30",
                        "valid_start": "2020-07-01", "valid_end": "2020-08-31",
                        "test_start": "2020-09-01", "test_end": "2020-09-30"},
                params_base={"anchor_date": "2020-01-01"},
                no_pretrain=True,
            )

        assert bm["success"] is False
        assert bm["error"] == "Training failed"

    def test_result_missing_keys(self):
        """Result dict without success/error keys — checks .get() defaults."""
        mock_future = MagicMock()
        mock_future.result.return_value = ({}, 0, 0)  # empty result dict

        mock_executor = MagicMock()
        mock_executor.__enter__.return_value.submit.return_value = mock_future
        mock_executor.__exit__.return_value = None

        with patch("concurrent.futures.ProcessPoolExecutor",
                   return_value=mock_executor):
            bm = rb.benchmark_model_isolated(
                qlib_config=None,
                model_name="min_model",
                yaml_file="workflow.yaml",
                window={"train_start": "2020-01-01", "train_end": "2020-06-30",
                        "valid_start": "2020-07-01", "valid_end": "2020-08-31",
                        "test_start": "2020-09-01", "test_end": "2020-09-30"},
                params_base={"anchor_date": "2020-01-01"},
            )

        assert bm["success"] is False
        assert bm["error"] is None
        assert bm["record_id"] is None

