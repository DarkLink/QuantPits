import os
import json
import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
def zscore_norm(series: pd.Series) -> pd.Series:
    """按天 Z-Score 归一化 (减均值，除标准差)"""
    def _norm_func(x):
        std = x.std()
        if std == 0:
            return x - x.mean()
        return (x - x.mean()) / std
    return series.groupby(level="datetime", group_keys=False).apply(_norm_func)


def rank_norm(series: pd.Series) -> pd.Series:
    """按天 Cross-Sectional Percentile Rank 归一化 -> [0, 1].

    对每个交易日截面内所有股票按预测分排名，映射到 [0, 1]，
    使各模型在融合时获得完全平等的投票权。
    n=1 (仅一只股票) 时返回 0.5。
    """
    def _rank_func(x):
        n = len(x)
        if n <= 1:
            return pd.Series(0.5, index=x.index)
        ranked = x.rank(method="average")
        return (ranked - 1) / (n - 1)
    return series.groupby(level="datetime", group_keys=False).apply(_rank_func)


def load_predictions_from_recorder(
    train_records: dict,
    selected_models: Optional[List[str]] = None,
    norm_method: str = "rank",
) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """
    从 Qlib Recorder 加载所有模型或选定模型的预测值，归一化后返回宽表。

    Args:
        train_records: 包含 experiment_name 和 models 字典的记录信息。
        selected_models: 选定的模型列表。为 None 则加载全部。
        norm_method: 归一化方式，\"zscore\" (默认) 或 \"rank\"。

    Returns:
        norm_df: DataFrame, index=(datetime, instrument), columns=model_names
        model_metrics: dict, model_name -> ICIR
        loaded_models: 成功加载的模型列表
    """
    from qlib.workflow import R
    from quantpits.utils.train_utils import get_experiment_name_for_model
    models = train_records.get("models", {})

    if selected_models is None:
        selected_models = list(models.keys())
    
    all_preds = []
    model_metrics = {}
    loaded_models = []

    print(f"\n{'='*60}")
    print("加载模型预测数据...")
    print(f"{'='*60}")
    print(f"Models to load ({len(selected_models)}): {selected_models}")

    for model_name in selected_models:
        if model_name not in models:
            print(f"  [{model_name}] SKIPPED: 不在训练记录中")
            continue
            
        record_id = models[model_name]
        try:
            model_exp_name = get_experiment_name_for_model(train_records, model_name)
            recorder = R.get_recorder(
                recorder_id=record_id, experiment_name=model_exp_name
            )

            # 加载预测值
            pred = recorder.load_object("pred.pkl")
            if isinstance(pred, pd.Series):
                pred = pred.to_frame("score")
            # 多列 pred (如含 score+label) 只保留 score 列
            if isinstance(pred, pd.DataFrame) and 'score' in pred.columns and len(pred.columns) > 1:
                pred = pred[['score']]
            pred.columns = [model_name]
            all_preds.append(pred)
            loaded_models.append(model_name)

            # 读取 ICIR 指标
            raw_metrics = recorder.list_metrics()
            metric_val = 0.0
            for k, v in raw_metrics.items():
                if "ICIR" in k:
                    metric_val = v
                    break
            model_metrics[model_name] = metric_val

            print(f"  [{model_name}] OK. Preds={len(pred)}, ICIR={metric_val:.4f}")
        except Exception as e:
            print(f"  [{model_name}] FAILED: {e}")

    print(f"\n成功加载 {len(all_preds)}/{len(selected_models)} 个模型")

    if not all_preds:
        raise ValueError("未加载到任何预测数据！")

    # 合并 & 归一化 (注意：不要在这里提前 dropna，避免由于单模型标的变动殃及其他模型)
    merged_df = pd.concat(all_preds, axis=1)
    print(f"合并后数据维度: {merged_df.shape}")

    norm_func = rank_norm if norm_method == "rank" else zscore_norm
    norm_df = pd.DataFrame(index=merged_df.index)
    for col in merged_df.columns:
        # 独立依靠单模型自身非空范围进行归一化
        norm_df[col] = norm_func(merged_df[col].dropna())

    # rank 模式: 模型未覆盖的股票视为弃权 (0.5 中性分)，扩大可融合股票池
    if norm_method == "rank":
        norm_df = norm_df.fillna(0.5)

    return norm_df, model_metrics, loaded_models

def save_predictions_to_recorder(
    pred: pd.DataFrame, 
    experiment_name: str, 
    model_name: str, 
    tags: Optional[Dict[str, Any]] = None
) -> str:
    """
    保存预测结果 DataFrame 到 Qlib Recorder，并返回生成的 record_id。
    """
    tags = tags or {}
    tags.setdefault("model", model_name)
    tags.setdefault("mode", "fused_prediction")
    
    from qlib.workflow import R
    with R.start(experiment_name=experiment_name):
        R.set_tags(**tags)
        # 必须确保格式兼容原有读取接口（通常要求包含 score 字段或序列）
        R.save_objects(**{"pred.pkl": pred})
        record_id = R.get_recorder().info["id"]
        
    return record_id

def split_is_oos(norm_df: pd.DataFrame, cutoff_date: pd.Timestamp, start_date=None, end_date=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """简单根据日期切分 In-Sample / Out-of-Sample 的宽表数据。"""
    is_mask = norm_df.index.get_level_values("datetime") <= cutoff_date
    if start_date:
        is_mask &= norm_df.index.get_level_values("datetime") >= start_date
        
    is_norm_df = norm_df[is_mask]
    
    oos_mask = norm_df.index.get_level_values("datetime") > cutoff_date
    if end_date:
        oos_mask &= norm_df.index.get_level_values("datetime") <= end_date
    oos_norm_df = norm_df[oos_mask]
    
    return is_norm_df, oos_norm_df

def extract_report_df(metrics_result):
    """
    通用解析 Qlib 回测 metrics 返回结果，提取出 report_normal 的 DataFrame。
    支持 Qlib 的多种返回格式：字典、元组、嵌套元组等。
    """
    if metrics_result is None:
        return None
    if isinstance(metrics_result, pd.DataFrame):
        return metrics_result
    
    # metrics 可能是个字典，例如 {"fold1": (report_norm, report_pos)}
    if isinstance(metrics_result, dict):
        first_fold = list(metrics_result.values())[0]
        if isinstance(first_fold, tuple) and len(first_fold) > 0:
            return first_fold[0]
        elif isinstance(first_fold, pd.DataFrame):
            return first_fold
            
    # metrics 也可能是个元组，例如 (report_norm, report_pos)
    if isinstance(metrics_result, tuple):
        first_item = metrics_result[0]
        # 有时可能是嵌套元组 ((report, pos), algo)
        if isinstance(first_item, tuple) and len(first_item) > 0:
            return first_item[0]
        elif isinstance(first_item, pd.DataFrame):
            return first_item
            
    return metrics_result

