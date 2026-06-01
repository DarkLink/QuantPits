"""
内存管理工具模块

提供三级内存清理策略，防止 rolling 训练过程中 OOM:
  Level 1: cleanup_after_window — 每个 window 训练后轻量清理
  Level 2: deep_cleanup_after_model — 每个模型完成后深度清理
  Level 3: check_memory_pressure — 内存安全阀
"""

import gc


def log_memory(tag):
    """打印当前进程 RSS 和系统内存使用率"""
    try:
        import psutil
        proc = psutil.Process()
        rss_gb = proc.memory_info().rss / 1e9
        vm = psutil.virtual_memory()
        print(f"  🧹 [{tag}] RSS={rss_gb:.1f}GB, "
              f"System={vm.percent:.0f}% ({vm.used/1e9:.1f}/{vm.total/1e9:.1f}GB)")
    except ImportError:
        pass


def cleanup_after_window(model_name, widx):
    """
    Level 1: 每个 window×model 训练后的轻量清理。

    清理 PyTorch GPU 缓存并执行一轮 GC。
    此时 model/dataset/pred 应该已经被 del。
    """
    # 1. PyTorch GPU 缓存
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # 2. 强制 GC
    gc.collect()

    # 3. 内存监控
    log_memory(f"{model_name}|W{widx}")


def deep_cleanup_after_model(model_name):
    """
    Level 2: 一个模型完成所有 window 后的深度清理。

    清理 qlib 全局 MemCache (H)，释放累积的 feature/calendar/instrument 缓存。
    """
    # 1. 清理 qlib 全局 MemCache
    try:
        from qlib.data.cache import H
        H.clear()
        print(f"  🧹 [{model_name}] qlib MemCache cleared")
    except Exception:
        pass

    # 2. PyTorch GPU 全面清理
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # 3. 两轮 GC（第一轮释放循环引用，第二轮释放 weak ref 指向的对象）
    gc.collect()
    gc.collect()

    log_memory(f"{model_name}|ALL_DONE")


def check_memory_pressure(tag, threshold_pct=85):
    """
    Level 3: 内存安全阀。

    如果系统内存使用率超过阈值，先尝试深度清理。
    如果仍然超过 90%，抛出 MemoryError 主动终止进程，
    避免 WSL 拖垮 Windows 宿主。

    可使用 --resume 恢复。
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        if vm.percent > threshold_pct:
            print(f"  ⚠️ [{tag}] Memory pressure: {vm.percent:.0f}% "
                  f"({vm.used/1e9:.1f}/{vm.total/1e9:.1f}GB)")

            # 尝试深度清理
            deep_cleanup_after_model(tag)

            # 重新检查
            vm = psutil.virtual_memory()
            if vm.percent > 90:
                raise MemoryError(
                    f"内存使用率 {vm.percent:.0f}% 超过安全阈值，"
                    f"在 [{tag}] 主动终止以防宿主崩溃。"
                    f"请使用 --resume 恢复。"
                )
    except ImportError:
        pass
