"""
融合引擎 — 权重计算 & 信号融合

从 ensemble_fusion.py Stage 3 / Stage 4 抽取，供多处复用。
"""

import pandas as pd
import numpy as np


def calculate_weights(norm_df, model_metrics, method, model_config,
                      ensemble_config, manual_weights_str=None):
    """
    计算各模型权重。

    Args:
        norm_df: 归一化后的预测宽表
        model_metrics: {model_name: ICIR}
        method: 权重模式 ('equal', 'icir_weighted', 'manual', 'dynamic')
        model_config: 模型配置
        ensemble_config: ensemble 配置
        manual_weights_str: 手动权重字符串 "model1:w1,model2:w2"

    Returns:
        final_weights: 动态权重 DataFrame (dynamic 模式) 或 None
        static_weights: 静态权重 dict (非 dynamic 模式) 或 None
        is_dynamic: bool
    """
    model_names = list(norm_df.columns)

    top_k = model_config.get('topk', model_config.get('TopK', 20))
    min_ic = ensemble_config.get('min_model_ic', 0.01)

    print(f"\n{'='*60}")
    print(f"Stage 3: 权重计算 (Mode: {method})")
    print(f"{'='*60}")

    if method == 'dynamic':
        # ---- 滚动 TopK Sharpe 动态权重 ----
        from qlib.data import D

        ROLLING_WINDOW = 60
        MIN_SHARPE_THRESHOLD = 0.0
        EVAL_TOP_K = top_k
        LABEL_FIELD = ['Ref($close, -6)/$close - 1']

        print(f">>> 使用动态权重 (Rolling TopK Sharpe, Window={ROLLING_WINDOW})")

        # 加载真实 Label
        instruments = norm_df.index.get_level_values('instrument').unique().tolist()
        start_date = norm_df.index.get_level_values('datetime').min()
        end_date = norm_df.index.get_level_values('datetime').max()
        label_df = D.features(instruments, LABEL_FIELD, start_time=start_date, end_time=end_date)
        label_df.columns = ['label']
        eval_df = norm_df.join(label_df, how='inner')

        # 计算每个模型每天的 TopK 平均收益
        dates = eval_df.index.get_level_values('datetime').unique().sort_values()
        perf_dict = {m: [] for m in model_names}

        for date in dates:
            if date not in eval_df.index:
                for m in model_names:
                    perf_dict[m].append(0)
                continue
            day_data = eval_df.loc[date]
            for model in model_names:
                top_stocks = day_data.nlargest(EVAL_TOP_K, model)
                perf_dict[model].append(top_stocks['label'].mean())

        daily_topk_ret = pd.DataFrame(perf_dict, index=dates)

        # 滚动 Sharpe
        rolling_mean = daily_topk_ret.rolling(window=ROLLING_WINDOW, min_periods=20).mean()
        rolling_std = daily_topk_ret.rolling(window=ROLLING_WINDOW, min_periods=20).std()
        rolling_sharpe = (rolling_mean / (rolling_std + 1e-9)).fillna(0)

        # 熔断 + 归一化
        raw_weights = rolling_sharpe.copy()
        raw_weights[raw_weights < MIN_SHARPE_THRESHOLD] = 0
        weight_sum = raw_weights.sum(axis=1)
        equal_w = pd.DataFrame(1.0 / len(model_names), index=raw_weights.index, columns=raw_weights.columns)
        final_weights = raw_weights.div(weight_sum, axis=0).fillna(equal_w)
        final_weights = final_weights.shift(1).fillna(1.0 / len(model_names))

        print('\n平均权重分布:')
        for m, w in final_weights.mean().sort_values(ascending=False).items():
            print(f"  {m:<30}: {w:.4f} ({w*100:.1f}%)")

        return final_weights, None, True

    elif method == 'icir_weighted':
        # ---- 静态 ICIR 加权 ----
        print(f">>> 使用 ICIR 加权 (min_ic={min_ic})")
        valid = {m: max(0, v) for m, v in model_metrics.items() if m in model_names and v > min_ic}
        if not valid:
            print("Warning: 无有效 ICIR，使用等权")
            static_weights = {m: 1.0 / len(model_names) for m in model_names}
        else:
            total = sum(valid.values())
            static_weights = {m: valid.get(m, 0) / total if total > 0 else 1.0 / len(valid) for m in model_names}

        for m, w in sorted(static_weights.items(), key=lambda x: -x[1]):
            print(f"  {m:<30}: {w:.4f} ({w*100:.1f}%)  ICIR={model_metrics.get(m, 0):.4f}")

        return None, static_weights, False

    elif method == 'manual':
        # ---- 手动权重 ----
        print(">>> 使用手动权重")
        if manual_weights_str:
            manual_w = {}
            for item in manual_weights_str.split(','):
                parts = item.strip().split(':')
                if len(parts) == 2:
                    manual_w[parts[0].strip()] = float(parts[1].strip())
        else:
            manual_w = ensemble_config.get('manual_weights', {})

        total = sum(manual_w.get(m, 0) for m in model_names)
        if total == 0:
            print("Warning: 手动权重总和为 0，使用等权")
            static_weights = {m: 1.0 / len(model_names) for m in model_names}
        else:
            static_weights = {m: manual_w.get(m, 0) / total for m in model_names}

        for m, w in static_weights.items():
            print(f"  {m:<30}: {w:.4f} ({w*100:.1f}%)")

        return None, static_weights, False

    else:  # equal
        # ---- 等权 ----
        print(f">>> 使用等权 ({len(model_names)} 个模型)")
        static_weights = {m: 1.0 / len(model_names) for m in model_names}

        for m, w in static_weights.items():
            icir = model_metrics.get(m, 0)
            print(f"  {m:<30}: {w:.4f} ({w*100:.1f}%)  ICIR={icir:.4f}")

        return None, static_weights, False


def generate_ensemble_signal(norm_df, final_weights, static_weights, is_dynamic):
    """生成融合信号"""
    model_names = list(norm_df.columns)

    print(f"\n{'='*60}")
    print("Stage 4: 生成 Ensemble 融合信号")
    print(f"{'='*60}")

    final_score = pd.Series(0.0, index=norm_df.index, name='score')

    if is_dynamic:
        for model in model_names:
            w = final_weights[model]
            weighted_pred = norm_df[model].mul(w, level='datetime')
            final_score += weighted_pred
    else:
        for model in model_names:
            w = static_weights.get(model, 0)
            if w > 0:
                final_score += norm_df[model] * w

    # 统计检查
    print(f"\n=== Ensemble Signal 统计 ===")
    print(f"数据量: {len(final_score)}")
    print(f"Min: {final_score.min():.4f}, Max: {final_score.max():.4f}, Mean: {final_score.mean():.4f}")
    print(f"Std: {final_score.std():.4f}")

    if final_score.std() == 0:
        print("!!! 警告：最终结果标准差为0，加权可能失败 !!!")
    else:
        print("加权成功。")

    return final_score
