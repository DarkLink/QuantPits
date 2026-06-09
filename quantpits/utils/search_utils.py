"""
组合搜索共享工具 — 信号处理、回测执行、IS/OOS 切分、分组穷举

从 brute_force_ensemble.py / brute_force_fast.py / minentropy_ensemble.py / analyze_ensembles.py
中抽取的公共逻辑，消除 hacky cross-script import 和代码重复。
"""

import os
import sys
import json
import signal
import itertools
import yaml
import pandas as pd
import numpy as np


# ============================================================================
# 全局中断标志 & 信号处理
# ============================================================================
_shutdown = False
_original_sigint = None
_original_sigterm = None


def _signal_handler(signum, frame):
    """收到 SIGINT/SIGTERM 后标记安全中断"""
    global _shutdown
    if _shutdown:
        # 第二次中断 → 强制退出
        print("\n\n⛔ 再次收到中断信号，强制退出...")
        sys.exit(1)
    _shutdown = True
    sig_name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else "SIGTERM"
    print(f"\n\n⚠️  收到 {sig_name}，将在当前批次完成后安全退出...")
    print("   (再次按 Ctrl+C 强制退出)")


def _install_signal_handlers():
    """安装信号处理器"""
    global _original_sigint, _original_sigterm
    _original_sigint = signal.getsignal(signal.SIGINT)
    _original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _restore_signal_handlers():
    """恢复原始信号处理器"""
    if _original_sigint is not None:
        signal.signal(signal.SIGINT, _original_sigint)
    if _original_sigterm is not None:
        signal.signal(signal.SIGTERM, _original_sigterm)


# ============================================================================
# 单组合回测核心
# ============================================================================

def _compute_combo_score(norm_df, combo_models, weight_df=None):
    """合成组合信号。

    Args:
        norm_df: 归一化预测宽表
        combo_models: 组合中的模型名列表
        weight_df: 可选，动态权重 DataFrame (index=datetime, columns=models)

    Returns:
        combo_score: Series (MultiIndex: datetime, instrument)
    """
    combo_data = norm_df[list(combo_models)].dropna(how='any')

    if weight_df is not None:
        weight_dates = weight_df.index
        date_mask = combo_data.index.get_level_values('datetime').isin(weight_dates)
        combo_data = combo_data[date_mask]
        if len(combo_data) == 0:
            return norm_df[list(combo_models)].dropna(how='any').mean(axis=1)

        dates = combo_data.index.get_level_values('datetime')
        w = weight_df.loc[dates, list(combo_models)]
        w_sum = w.sum(axis=1).replace(0, 1.0)
        w = w.div(w_sum, axis=0)
        combo_score = (combo_data.values * w.values).sum(axis=1)
        return pd.Series(combo_score, index=combo_data.index)

    return combo_data.mean(axis=1)


def run_single_backtest(
    combo_models, norm_df, top_k, drop_n, benchmark, freq,
    trade_exchange, bt_start, bt_end, st_config=None, bt_config=None,
    weight_df=None,
):
    """对指定的模型组合进行回测，返回指标字典或 None"""
    from quantpits.utils import strategy
    from quantpits.utils.backtest_utils import run_backtest_with_strategy, standard_evaluate_portfolio

    if st_config is None:
        st_config = strategy.load_strategy_config()
    if bt_config is None:
        bt_config = strategy.get_backtest_config(st_config)

    # 1. 合成信号
    combo_score = _compute_combo_score(norm_df, combo_models, weight_df)

    import copy
    st_config = copy.deepcopy(st_config)
    st_config["strategy"]["params"]["topk"] = top_k
    st_config["strategy"]["params"]["n_drop"] = drop_n

    strategy_inst = strategy.create_backtest_strategy(combo_score, st_config)

    # 2. 回测 (重定向 stderr 以屏蔽 qlib backtest loop 内部的 tqdm 进度条)
    try:
        _stderr_fd = os.open(os.devnull, os.O_WRONLY)
        _old_stderr = os.dup(2)
        os.dup2(_stderr_fd, 2)
        os.close(_stderr_fd)
        try:
            report, _ = run_backtest_with_strategy(
                strategy_inst=strategy_inst,
                trade_exchange=trade_exchange,
                freq=freq,
                account_cash=bt_config["account"],
                bt_start=bt_start,
                bt_end=bt_end
            )
        finally:
            os.dup2(_old_stderr, 2)
            os.close(_old_stderr)

        # 3. 标准化计算结果
        st_config_inner = strategy.load_strategy_config()
        benchmark_col = st_config_inner.get('benchmark', 'SH000300')
        
        metrics = standard_evaluate_portfolio(report, benchmark_col, freq)

        return {
            "models": ",".join(combo_models),
            "n_models": len(combo_models),
            "Ann_Ret": metrics.get("CAGR_252", 0),
            "Max_DD": metrics.get("Max_Drawdown", 0),
            "Excess_Ret": metrics.get("Absolute_Return", 0) - metrics.get("Benchmark_Absolute_Return", 0),
            "Ann_Excess": metrics.get("Excess_Return_CAGR_252", 0),
            "Total_Ret": metrics.get("Absolute_Return", 0),
            "Final_NAV": report.iloc[-1]["account"],
            "Calmar": metrics.get("Calmar", 0) if pd.notna(metrics.get("Calmar")) else 0,
        }
    except Exception as e:
        print(f"  [ERROR] Combo {combo_models} failed: {e}")
        return None


# ============================================================================
# 动态权重计算 (Rolling TopK Sharpe)
# ============================================================================

def compute_rolling_sharpe_weights(norm_df, top_k, window=60, min_periods=20,
                                    label_field=None):
    """为每个模型计算基于滚动 TopK Sharpe 的动态权重。

    对每个模型，每天取其预测分值最高的 top_k 只股票，计算这些股票
    未来 6 日收益率的均值，然后对该序列计算滚动 Sharpe，shift(1)
    避免未来信息泄露，最后跨模型归一化。

    Args:
        norm_df: 归一化预测宽表 (MultiIndex: datetime, instrument; columns: models)
        top_k: 每日选取的 TopK 股票数
        window: 滚动 Sharpe 窗口 (交易日)
        min_periods: 滚动窗口最小样本数
        label_field: 前向收益标签字段，默认 ['Ref($close, -6)/$close - 1']

    Returns:
        weight_df: DataFrame (index=datetime, columns=models) 每模型每日权重
    """
    from qlib.data import D

    if label_field is None:
        label_field = ['Ref($close, -6)/$close - 1']

    model_names = list(norm_df.columns)

    instruments = norm_df.index.get_level_values('instrument').unique().tolist()
    start_date = norm_df.index.get_level_values('datetime').min()
    end_date = norm_df.index.get_level_values('datetime').max()

    print(f"\n>>> 计算滚动 TopK Sharpe 动态权重 (Window={window}, TopK={top_k})")
    print(f"    标签字段: {label_field[0]}")
    print(f"    日期范围: {start_date.date()} ~ {end_date.date()}")

    label_df = D.features(instruments, label_field,
                          start_time=start_date, end_time=end_date)
    label_df.columns = ['label']

    eval_df = norm_df.join(label_df, how='inner')

    dates = eval_df.index.get_level_values('datetime').unique().sort_values()
    perf_dict = {m: [] for m in model_names}

    for date in dates:
        if date not in eval_df.index:
            for m in model_names:
                perf_dict[m].append(0)
            continue
        day_data = eval_df.loc[date]
        for model in model_names:
            top_stocks = day_data.nlargest(top_k, model)
            perf_dict[model].append(top_stocks['label'].mean())

    daily_topk_ret = pd.DataFrame(perf_dict, index=dates)

    rolling_mean = daily_topk_ret.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = daily_topk_ret.rolling(window=window, min_periods=min_periods).std()
    rolling_sharpe = (rolling_mean / (rolling_std + 1e-9)).fillna(0)

    raw_weights = rolling_sharpe.copy()
    raw_weights[raw_weights < 0] = 0
    weight_sum = raw_weights.sum(axis=1)
    equal_w = pd.DataFrame(1.0 / len(model_names),
                           index=raw_weights.index, columns=raw_weights.columns)
    final_weights = raw_weights.div(weight_sum, axis=0).fillna(equal_w)
    final_weights = final_weights.shift(1).fillna(1.0 / len(model_names))

    print('    平均权重分布:')
    for m, w in final_weights.mean().sort_values(ascending=False).items():
        print(f"      {m:<30}: {w:.4f} ({w*100:.1f}%)")

    return final_weights


# ============================================================================
# CSV 增量写入
# ============================================================================

def _append_results_to_csv(csv_path, results, write_header=False):
    """将一批结果追加写入 CSV 文件"""
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


# ============================================================================
# IS/OOS 数据集切分
# ============================================================================

def split_is_oos_by_args(norm_df, args):
    """根据参数将 norm_df 划分为 IS (In-Sample) 和 OOS (Out-of-Sample)"""
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    
    dates = norm_df.index.get_level_values("datetime").unique().sort_values()
    max_date = dates.max()
    
    cutoff_date = max_date
    if args.exclude_last_years > 0:
        cutoff_date = cutoff_date - pd.DateOffset(years=args.exclude_last_years)
    if args.exclude_last_months > 0:
        cutoff_date = cutoff_date - pd.DateOffset(months=args.exclude_last_months)
        
    end_date = pd.to_datetime(args.end_date) if args.end_date else None
    if end_date and end_date < cutoff_date:
        cutoff_date = end_date
        
    is_mask = norm_df.index.get_level_values("datetime") <= cutoff_date
    if start_date:
        is_mask &= norm_df.index.get_level_values("datetime") >= start_date
        
    is_norm_df = norm_df[is_mask]
    
    # OOS 是截止日之后的数据
    oos_mask = norm_df.index.get_level_values("datetime") > cutoff_date
    oos_norm_df = norm_df[oos_mask]
    
    print(f"\n=== 数据集划分 (In-Sample / Out-Of-Sample) ===")
    if not is_norm_df.empty:
        print(f"IS 期  : {is_norm_df.index.get_level_values('datetime').min().date()} ~ {is_norm_df.index.get_level_values('datetime').max().date()} (共 {len(is_norm_df.index.get_level_values('datetime').unique())} 天)")
    else:
        print("IS 期  : 无数据")
        
    if not oos_norm_df.empty:
        print(f"OOS 期 : {oos_norm_df.index.get_level_values('datetime').min().date()} ~ {oos_norm_df.index.get_level_values('datetime').max().date()} (共 {len(oos_norm_df.index.get_level_values('datetime').unique())} 天)")
    else:
        print("OOS 期 : 无数据")
        
    return is_norm_df, oos_norm_df


# ============================================================================
# 模型分组 & 组合生成
# ============================================================================

def extract_group_model_names(group_config_path):
    """Extract flat set of model names from a group config YAML.

    Unlike load_combo_groups, this does NOT validate against available_models.
    Intended for early filtering BEFORE predictions are loaded, so we know
    which models to load without loading everything first.

    Args:
        group_config_path: path to combo_groups.yaml

    Returns:
        set of model name strings (bare names, as written in YAML)
    """
    with open(group_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_groups = cfg.get("groups", {})
    if not raw_groups:
        raise ValueError(f"分组配置为空: {group_config_path}")

    all_models = set()
    for models in raw_groups.values():
        all_models.update(models)

    print(f"从分组配置中提取到 {len(all_models)} 个模型: {sorted(all_models)}")
    return all_models


def load_combo_groups(group_config_path, available_models):
    """
    加载分组配置，验证模型名，返回有效分组。

    Args:
        group_config_path: combo_groups.yaml 路径
        available_models: 当前加载到的模型列表 (norm_df.columns)

    Returns:
        groups: dict, group_name -> list of valid model names
    """
    with open(group_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_groups = cfg.get("groups", {})
    if not raw_groups:
        raise ValueError(f"分组配置为空: {group_config_path}")

    available_set = set(available_models)
    groups = {}
    skipped_models = []

    for gname, models in raw_groups.items():
        valid = [m for m in models if m in available_set]
        invalid = [m for m in models if m not in available_set]
        if invalid:
            skipped_models.extend(invalid)
            print(f"  ⚠️  组 [{gname}] 中以下模型不存在于预测数据中，已忽略: {invalid}")
        if valid:
            groups[gname] = valid
        else:
            print(f"  ⚠️  组 [{gname}] 无有效模型，已跳过")

    if skipped_models:
        print(f"  共忽略 {len(skipped_models)} 个无效模型")

    # 检查未分组的模型 (仅打印提示，不自动参与)
    grouped_models = set()
    for models in groups.values():
        grouped_models.update(models)
    ungrouped = available_set - grouped_models
    if ungrouped:
        print(f"  ℹ️  以下模型未在任何分组中，将被排除: {sorted(ungrouped)}")

    return groups


def generate_grouped_combinations(groups, min_combo_size=1, max_combo_size=0):
    """
    基于分组生成组合：从所有组的子集中，每组选一个模型，做笛卡尔积。

    为支持 min/max combo size，我们枚举组的子集（选哪些组参与），
    然后对参与的组做 itertools.product。

    Args:
        groups: dict, group_name -> list of models
        min_combo_size: 最小组合大小 (选几个组)
        max_combo_size: 最大组合大小 (0=全部组)

    Returns:
        list of tuples, 每个 tuple 是一个模型组合
    """
    group_names = list(groups.keys())
    n_groups = len(group_names)
    max_size = max_combo_size if max_combo_size > 0 else n_groups
    max_size = min(max_size, n_groups)

    all_combinations = []

    # 枚举选哪些组参与 (选 r 个组的组合)
    for r in range(min_combo_size, max_size + 1):
        for group_subset in itertools.combinations(group_names, r):
            # 对选中的组做笛卡尔积
            model_lists = [groups[g] for g in group_subset]
            for combo in itertools.product(*model_lists):
                all_combinations.append(combo)

    return all_combinations


# ============================================================================
# Process pool worker globals & helpers
# ============================================================================
_worker_norm_df = None
_worker_weight_df = None
_worker_trade_exchange = None
_worker_bt_start = None
_worker_bt_end = None
_worker_st_config = None
_worker_bt_config = None


def worker_init(norm_df, exchange_kwargs, all_codes, bt_start, bt_end,
                exchange_freq, st_config, bt_config, weight_df=None):
    """Initialize a ProcessPoolExecutor worker: fresh Qlib init + pre-create Exchange.

    Called once per worker process.  Because the pool uses ``spawn`` (not
    ``fork``), the worker is a brand-new Python interpreter — no inherited
    Qlib state, no inherited locks.  ``init_qlib()`` here is a genuine
    first-time init.

    The ``Exchange`` is created once per worker and reused for all subsequent
    backtests, avoiding the ~2s construction overhead per combo.
    """
    import quantpits.utils.env as _env
    import os as _os
    for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[_key] = "1"
    _env.init_qlib()

    from qlib.backtest.exchange import Exchange

    _exch_kwargs = dict(exchange_kwargs)
    trade_exchange = Exchange(
        freq=exchange_freq,
        start_time=bt_start,
        end_time=bt_end,
        codes=all_codes,
        **_exch_kwargs,
    )

    global _worker_norm_df, _worker_weight_df, _worker_trade_exchange
    global _worker_bt_start, _worker_bt_end
    global _worker_st_config, _worker_bt_config
    _worker_norm_df = norm_df
    _worker_weight_df = weight_df
    _worker_trade_exchange = trade_exchange
    _worker_bt_start = bt_start
    _worker_bt_end = bt_end
    _worker_st_config = st_config
    _worker_bt_config = bt_config


def run_backtest_in_worker(combo_models, top_k, drop_n, benchmark, freq):
    """Run a single backtest inside a ProcessPoolExecutor worker.

    Uses the pre-created ``Exchange`` (stored by :func:`worker_init`) and
    delegates to :func:`run_single_backtest`.
    """
    return run_single_backtest(
        combo_models, _worker_norm_df, top_k, drop_n, benchmark, freq,
        _worker_trade_exchange, _worker_bt_start, _worker_bt_end,
        _worker_st_config, _worker_bt_config,
        weight_df=_worker_weight_df,
    )


# ============================================================================
# Metadata & Config
# ============================================================================

def load_oos_config(workspace_root=None):
    """
    加载 OOS 配置 (config/oos_config.json)。
    
    优先级:
    1. workspace_root/config/oos_config.json
    2. env.ROOT_DIR/config/oos_config.json
    """
    from quantpits.utils import env
    search_dirs = []
    if workspace_root:
        search_dirs.append(os.path.join(workspace_root, "config"))
    search_dirs.append(os.path.join(env.ROOT_DIR, "config"))
    
    for d in search_dirs:
        path = os.path.join(d, "oos_config.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def save_run_metadata(ctx, metadata: dict):
    """
    保存运行元数据到 run_metadata.json，自动注入 OOS 配置。
    
    Args:
        ctx: RunContext 实例
        metadata: 原始元数据字典
    """
    # 注入 OOS 配置
    oos_cfg = load_oos_config()
    if oos_cfg:
        metadata["oos_params"] = oos_cfg
        
    metadata_path = ctx.run_path("run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 元数据已保存: {metadata_path}")
    return metadata_path
