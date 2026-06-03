#!/usr/bin/env python
"""
Rolling Benchmark - 滚动训练资源采集

对每个模型在单个 rolling window 上运行完整的 train+predict 流程，
采集 wall-time、peak RSS、peak GPU VRAM 等资源指标，输出到 data/rolling_benchmark.json。

**安全保证**:
  - 不修改 rolling_state.json（不影响正式训练进度）
  - 使用独立的 MLflow experiment（不污染正式训练记录）
  - 不修改 latest_train_records.json

运行方式:
  cd QuantPits && python quantpits/scripts/rolling_benchmark.py [options]

示例:
  # 对所有 enabled 模型跑 Window 0
  python quantpits/scripts/rolling_benchmark.py --all-enabled

  # 只测 GPU 模型
  python quantpits/scripts/rolling_benchmark.py --algorithm lstm,gru,transformer,catboost

  # 指定 window index
  python quantpits/scripts/rolling_benchmark.py --all-enabled --window-idx 2

  # 指定模型
  python quantpits/scripts/rolling_benchmark.py --models linear_Alpha158,catboost_Alpha158

  # Dry-run: 仅显示将要 benchmark 的模型和 window
  python quantpits/scripts/rolling_benchmark.py --all-enabled --dry-run

  # 从上次中断处继续（跳过已采集的模型）
  python quantpits/scripts/rolling_benchmark.py --all-enabled --resume
"""

import os
import sys
import json
import time
import argparse
import resource
import platform
from datetime import datetime

from quantpits.utils import env
os.chdir(env.ROOT_DIR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = env.ROOT_DIR

# Benchmark 专用 MLflow experiment 名（不污染正式训练）
BENCHMARK_EXPERIMENT = "Rolling_Benchmark"

# 输出文件
BENCHMARK_OUTPUT = os.path.join(ROOT_DIR, "data", "rolling_benchmark.json")


# ============================================================================
# GPU 监控
# ============================================================================

def get_gpu_info():
    """获取 GPU 信息（名称、总 VRAM）"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return {
            "name": name,
            "vram_total_mb": round(mem.total / 1e6),
        }
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "name": parts[0].strip(),
                "vram_total_mb": int(parts[1].strip()),
            }
    except Exception:
        pass

    return {"name": "Unknown", "vram_total_mb": 0}


def get_gpu_vram_used_mb():
    """获取当前 GPU 0 的已使用 VRAM (MB)"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return round(mem.used / 1e6)
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass

    return 0


# ============================================================================
# 子进程 Benchmark 入口
# ============================================================================

def _benchmark_in_subprocess(qlib_config_dict, model_name, yaml_file, window,
                              params_base, experiment_name, no_pretrain):
    """
    子进程入口: 初始化 qlib → 训练 → 返回 (result, peak_rss_mb, peak_vram_mb)。

    在子进程中执行训练，通过 resource.getrusage 获取子进程 peak RSS。
    GPU VRAM 通过训练前后的差值估算。
    """
    from qlib.config import C
    C.register_from_C(qlib_config_dict)

    # 训练前 baseline
    vram_before = get_gpu_vram_used_mb()

    # 用 resource module 追踪 peak RSS（子进程自身 + 子子进程）
    # 注意: ru_maxrss 在 Linux 下单位是 KB
    usage_before = resource.getrusage(resource.RUSAGE_SELF)

    # 执行训练
    from quantpits.scripts.rolling.training import train_window_model
    result = train_window_model(
        model_name=model_name,
        yaml_file=yaml_file,
        window=window,
        params_base=params_base,
        experiment_name=experiment_name,
        no_pretrain=no_pretrain,
    )

    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    vram_after = get_gpu_vram_used_mb()

    # peak RSS (KB → MB)
    peak_rss_mb = round(usage_after.ru_maxrss / 1024)

    # VRAM: 使用训练后的值作为估算（不是精确 peak，但足够参考）
    peak_vram_mb = max(0, vram_after - vram_before)
    # 如果训练后 VRAM 回落了（模型释放），用 vram_after 作为上界
    # 更好的方案是后台线程采样，但对 benchmark 足够
    if peak_vram_mb == 0 and vram_after > vram_before:
        peak_vram_mb = vram_after - vram_before

    return result, peak_rss_mb, peak_vram_mb


def _benchmark_with_vram_sampling(qlib_config_dict, model_name, yaml_file,
                                   window, params_base, experiment_name,
                                   no_pretrain):
    """
    子进程入口（带 VRAM 后台采样）: 更准确的 peak VRAM 测量。

    启动一个后台线程每 0.5 秒采样 GPU VRAM，取最大值。
    """
    import threading
    from qlib.config import C
    C.register_from_C(qlib_config_dict)

    # VRAM 采样器
    vram_baseline = get_gpu_vram_used_mb()
    vram_samples = [0]
    stop_event = threading.Event()

    def _sample_vram():
        while not stop_event.is_set():
            current = get_gpu_vram_used_mb()
            delta = max(0, current - vram_baseline)
            vram_samples.append(delta)
            stop_event.wait(0.5)

    sampler = threading.Thread(target=_sample_vram, daemon=True)
    sampler.start()

    # 执行训练
    from quantpits.scripts.rolling.training import train_window_model
    result = train_window_model(
        model_name=model_name,
        yaml_file=yaml_file,
        window=window,
        params_base=params_base,
        experiment_name=experiment_name,
        no_pretrain=no_pretrain,
    )

    # 停止采样
    stop_event.set()
    sampler.join(timeout=2)

    # 收集指标
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mb = round(usage.ru_maxrss / 1024)
    peak_vram_mb = max(vram_samples)

    return result, peak_rss_mb, peak_vram_mb


def benchmark_model_isolated(qlib_config, model_name, yaml_file, window,
                              params_base, no_pretrain=False):
    """
    在独立子进程中运行 benchmark（确保内存完全回收）。

    Returns:
        dict: benchmark 结果
    """
    import multiprocessing as mp
    import concurrent.futures

    experiment_name = BENCHMARK_EXPERIMENT

    _mp_ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=_mp_ctx
    ) as executor:
        t_start = time.perf_counter()

        future = executor.submit(
            _benchmark_with_vram_sampling,
            qlib_config,
            model_name,
            yaml_file,
            window,
            params_base,
            experiment_name,
            no_pretrain,
        )

        result, peak_rss_mb, peak_vram_mb = future.result()

        wall_sec = round(time.perf_counter() - t_start, 1)

    return {
        "wall_sec": wall_sec,
        "peak_rss_mb": peak_rss_mb,
        "peak_vram_mb": peak_vram_mb,
        "success": result.get("success", False),
        "error": result.get("error"),
        "record_id": result.get("record_id"),
    }


# ============================================================================
# 系统信息采集
# ============================================================================

def collect_system_info():
    """采集系统硬件信息"""
    import psutil

    cpu_name = "Unknown"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    gpu = get_gpu_info()

    return {
        "cpu": cpu_name,
        "cpu_count": os.cpu_count(),
        "ram_total_mb": round(psutil.virtual_memory().total / 1e6),
        "gpu_name": gpu["name"],
        "gpu_vram_mb": gpu["vram_total_mb"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================================
# 结果 I/O
# ============================================================================

def load_existing_benchmark():
    """加载已有的 benchmark 结果（支持 --resume）"""
    if os.path.exists(BENCHMARK_OUTPUT):
        try:
            with open(BENCHMARK_OUTPUT, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_benchmark(data):
    """保存 benchmark 结果"""
    os.makedirs(os.path.dirname(BENCHMARK_OUTPUT), exist_ok=True)
    with open(BENCHMARK_OUTPUT, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Benchmark 结果已保存: {BENCHMARK_OUTPUT}")


# ============================================================================
# CLI & Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rolling Benchmark: 采集每个模型的训练资源消耗",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --all-enabled                    # 所有 enabled 模型
  %(prog)s --models linear_Alpha158         # 指定模型
  %(prog)s --algorithm lstm,gru             # 按算法筛选
  %(prog)s --all-enabled --window-idx 2     # 指定 Window
  %(prog)s --all-enabled --dry-run          # 仅查看计划
  %(prog)s --all-enabled --resume           # 跳过已采集的模型
        """,
    )

    select = parser.add_argument_group("模型选择")
    select.add_argument("--models", type=str,
                        help="指定模型名，逗号分隔")
    select.add_argument("--algorithm", type=str,
                        help="按算法筛选（逗号分隔，如 lstm,gru）")
    select.add_argument("--dataset", type=str,
                        help="按数据集筛选")
    select.add_argument("--tag", type=str,
                        help="按标签筛选")
    select.add_argument("--all-enabled", action="store_true",
                        help="所有 enabled 模型")
    select.add_argument("--skip", type=str,
                        help="跳过指定模型，逗号分隔")

    ctrl = parser.add_argument_group("运行控制")
    ctrl.add_argument("--window-idx", type=int, default=0,
                      help="使用哪个 Window 进行 benchmark (默认: 0)")
    ctrl.add_argument("--dry-run", action="store_true",
                      help="仅显示 benchmark 计划，不执行")
    ctrl.add_argument("--resume", action="store_true",
                      help="跳过已有 benchmark 数据的模型")
    ctrl.add_argument("--no-pretrain", action="store_true",
                      help="不加载预训练模型")

    return parser.parse_args()


def main():
    env.safeguard("Rolling Benchmark")
    args = parse_args()

    # 加载 rolling 配置
    from quantpits.utils.config_loader import load_rolling_config
    rolling_cfg = load_rolling_config(ROOT_DIR)
    if rolling_cfg is None:
        print("❌ 找不到 config/rolling_config.yaml")
        return

    # 初始化 qlib
    env.init_qlib()

    # 获取 anchor_date
    from quantpits.scripts.rolling_train import get_base_params
    params_base = get_base_params()
    anchor_date = params_base["anchor_date"]

    # 生成 windows
    from quantpits.scripts.rolling.windows import generate_rolling_windows
    windows = generate_rolling_windows(
        rolling_start=rolling_cfg["rolling_start"],
        train_years=rolling_cfg["train_years"],
        valid_years=rolling_cfg["valid_years"],
        test_step=rolling_cfg["test_step"],
        anchor_date=anchor_date,
    )

    if not windows:
        print("❌ 无法生成 rolling windows")
        return

    # 选择 benchmark window
    widx = args.window_idx
    if widx >= len(windows):
        print(f"❌ Window index {widx} 超出范围 (共 {len(windows)} 个 windows)")
        print(f"   有效范围: 0 ~ {len(windows) - 1}")
        return

    bm_window = windows[widx]
    print(f"\n📅 Benchmark Window {widx}:")
    print(f"   Train: [{bm_window['train_start']}, {bm_window['train_end']}]")
    print(f"   Valid: [{bm_window['valid_start']}, {bm_window['valid_end']}]")
    print(f"   Test:  [{bm_window['test_start']}, {bm_window['test_end']}]")

    # 解析目标模型
    has_selection = any([
        args.models, args.algorithm, args.dataset,
        args.tag, args.all_enabled,
    ])
    if not has_selection:
        print("❌ 请指定模型：--all-enabled, --models, --algorithm, --dataset, --tag")
        return

    from quantpits.utils.train_utils import resolve_target_models
    targets = resolve_target_models(args)
    if not targets:
        print("⚠️  没有匹配的模型")
        return

    # Classify models by algorithm
    from quantpits.utils.train_utils import load_model_registry
    registry = load_model_registry()

    # 按算法分类，预判 GPU/CPU
    gpu_algorithms = {
        "lstm", "gru", "alstm", "transformer", "localformer",
        "catboost", "tcn", "tabnet", "sfm", "sandwich",
        "tra", "adarnn", "add", "tft", "igmtf", "krnn",
        "gats", "tcts",
    }
    cpu_algorithms = {"linear", "lightgbm", "mlp", "xgboost", "ridge"}

    print(f"\n📋 Benchmark 计划:")
    print(f"{'='*70}")
    print(f"  {'模型名':<30} {'算法':<15} {'预判设备':<10}")
    print(f"  {'-'*30} {'-'*15} {'-'*10}")

    # Resume: 加载已有结果
    existing = None
    done_models = set()
    if args.resume:
        existing = load_existing_benchmark()
        if existing:
            done_models = set(existing.get("benchmarks", {}).keys())
            print(f"\n  ⏩ Resume: 已有 {len(done_models)} 个模型的数据")

    pending = {}
    for name, info in targets.items():
        algo = info.get("algorithm", "unknown")
        device = "GPU" if algo.lower() in gpu_algorithms else "CPU"
        if algo.lower() not in gpu_algorithms and algo.lower() not in cpu_algorithms:
            device = "GPU?"  # 未知算法默认标记为 GPU

        status = ""
        if name in done_models:
            status = " (已采集, 跳过)"
        else:
            pending[name] = info

        print(f"  {name:<30} {algo:<15} {device:<10}{status}")

    print(f"{'='*70}")
    print(f"  总计: {len(targets)} 个模型, 待 benchmark: {len(pending)} 个")

    if args.dry_run:
        print("\n🔍 Dry-run 模式: 不实际执行")
        return

    if not pending:
        print("\n✅ 所有模型已 benchmark 完毕!")
        return

    # 确认
    print(f"\n⏳ 即将开始 benchmark，预计耗时 ~{len(pending) * 5} 分钟 (粗估)")
    print("   按 Ctrl+C 可安全中断，已完成的结果会保存")
    print()

    # 采集系统信息
    sys_info = collect_system_info()
    print(f"📍 系统: {sys_info['cpu']}")
    print(f"   RAM: {sys_info['ram_total_mb']} MB")
    print(f"   GPU: {sys_info['gpu_name']} ({sys_info['gpu_vram_mb']} MB)")
    print()

    # 准备 qlib config（传给子进程）
    from qlib.config import C
    qlib_config = C

    # 准备输出数据结构
    if existing and args.resume:
        output = existing
        output["system"] = sys_info  # 更新系统信息
    else:
        output = {
            "system": sys_info,
            "benchmark_config": {
                "window_idx": widx,
                "window": bm_window,
                "rolling_config": rolling_cfg,
                "anchor_date": anchor_date,
            },
            "benchmarks": {},
        }

    # 执行 benchmark
    from quantpits.utils.operator_log import OperatorLog
    with OperatorLog("rolling_benchmark", args=sys.argv[1:]) as oplog:
        completed = 0
        failed = 0

        for model_name, model_info in pending.items():
            algo = model_info.get("algorithm", "unknown")
            yaml_file = model_info["yaml_file"]

            print(f"\n{'─'*60}")
            print(f"🔬 [{completed + failed + 1}/{len(pending)}] {model_name} ({algo})")
            print(f"{'─'*60}")

            try:
                bm = benchmark_model_isolated(
                    qlib_config=qlib_config,
                    model_name=model_name,
                    yaml_file=yaml_file,
                    window=bm_window,
                    params_base=params_base,
                    no_pretrain=args.no_pretrain,
                )

                # 补充算法信息
                bm["algorithm"] = algo
                bm["dataset"] = model_info.get("dataset", "unknown")
                device = "gpu" if algo.lower() in gpu_algorithms else "cpu"
                bm["device"] = device

                output["benchmarks"][model_name] = bm

                if bm["success"]:
                    completed += 1
                    print(f"  ✅ {model_name}: {bm['wall_sec']}s, "
                          f"RSS={bm['peak_rss_mb']}MB, "
                          f"VRAM={bm['peak_vram_mb']}MB")
                else:
                    failed += 1
                    print(f"  ❌ {model_name}: {bm.get('error', 'Unknown error')}")

            except KeyboardInterrupt:
                print(f"\n\n⚠️  用户中断！已完成 {completed} 个模型")
                break
            except Exception as e:
                failed += 1
                output["benchmarks"][model_name] = {
                    "algorithm": algo,
                    "dataset": model_info.get("dataset", "unknown"),
                    "success": False,
                    "error": str(e),
                    "wall_sec": 0,
                    "peak_rss_mb": 0,
                    "peak_vram_mb": 0,
                }
                print(f"  ❌ {model_name}: 异常 - {e}")

            # 每个模型完成后立即保存（防止崩溃丢失）
            save_benchmark(output)

        # 最终保存
        output["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_benchmark(output)

        # 打印汇总
        print(f"\n{'='*70}")
        print("📊 Benchmark 汇总")
        print(f"{'='*70}")

        benchmarks = output["benchmarks"]
        if benchmarks:
            # 按 wall_sec 排序
            sorted_bm = sorted(
                benchmarks.items(),
                key=lambda x: x[1].get("wall_sec", 0),
                reverse=True,
            )

            print(f"\n  {'模型名':<30} {'耗时(s)':<10} {'RSS(MB)':<10} "
                  f"{'VRAM(MB)':<10} {'设备':<6} {'状态'}")
            print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*6}")

            total_wall = 0
            for name, bm in sorted_bm:
                status = "✅" if bm.get("success") else "❌"
                wall = bm.get("wall_sec", 0)
                rss = bm.get("peak_rss_mb", 0)
                vram = bm.get("peak_vram_mb", 0)
                device = bm.get("device", "?")
                total_wall += wall
                print(f"  {name:<30} {wall:<10.1f} {rss:<10} "
                      f"{vram:<10} {device:<6} {status}")

            print(f"\n  总耗时: {total_wall:.0f}s ({total_wall/60:.1f}min)")
            print(f"  成功: {completed}, 失败: {failed}")

            # 按设备分组统计
            cpu_models = [
                (n, b) for n, b in sorted_bm
                if b.get("device") == "cpu" and b.get("success")
            ]
            gpu_models = [
                (n, b) for n, b in sorted_bm
                if b.get("device") == "gpu" and b.get("success")
            ]

            if cpu_models:
                avg_cpu = sum(b["wall_sec"] for _, b in cpu_models) / len(cpu_models)
                avg_rss = sum(b["peak_rss_mb"] for _, b in cpu_models) / len(cpu_models)
                print(f"\n  CPU 模型 ({len(cpu_models)}): "
                      f"平均 {avg_cpu:.0f}s, 平均 RSS {avg_rss:.0f}MB")

            if gpu_models:
                avg_gpu = sum(b["wall_sec"] for _, b in gpu_models) / len(gpu_models)
                avg_rss = sum(b["peak_rss_mb"] for _, b in gpu_models) / len(gpu_models)
                avg_vram = sum(b["peak_vram_mb"] for _, b in gpu_models) / len(gpu_models)
                print(f"  GPU 模型 ({len(gpu_models)}): "
                      f"平均 {avg_gpu:.0f}s, 平均 RSS {avg_rss:.0f}MB, "
                      f"平均 VRAM {avg_vram:.0f}MB")

            # 并行度建议
            ram_avail = sys_info["ram_total_mb"] * 0.8  # 留 20% 余量
            vram_avail = sys_info["gpu_vram_mb"] * 0.85  # 留 15% 余量

            if cpu_models:
                max_cpu_rss = max(b["peak_rss_mb"] for _, b in cpu_models)
                cpu_parallel = max(1, int(ram_avail / max_cpu_rss)) if max_cpu_rss > 0 else 8
                cpu_parallel = min(cpu_parallel, os.cpu_count() or 8)
                print(f"\n  💡 建议 CPU 并发度: {cpu_parallel} "
                      f"(基于 max RSS {max_cpu_rss}MB, "
                      f"可用 RAM ~{ram_avail:.0f}MB)")

            if gpu_models:
                max_gpu_vram = max(b["peak_vram_mb"] for _, b in gpu_models)
                if max_gpu_vram > 0:
                    gpu_parallel = max(1, int(vram_avail / max_gpu_vram))
                else:
                    gpu_parallel = 1
                print(f"  💡 建议 GPU 并发度: {gpu_parallel} "
                      f"(基于 max VRAM {max_gpu_vram}MB, "
                      f"可用 VRAM ~{vram_avail:.0f}MB)")

        print(f"\n  📁 结果文件: {BENCHMARK_OUTPUT}")
        print(f"{'='*70}")

        oplog.set_result({
            "n_models": len(benchmarks),
            "completed": completed,
            "failed": failed,
        })


if __name__ == "__main__":
    main()
