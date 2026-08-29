"""RandomModel — 随机排序猴子基准模型。

fit() 是空操作，predict() 对每只股票每天生成均匀随机分数。
用于量化检验：真实模型是否显著优于随机选股。

单次使用（标准 pipeline）：
    注册到 model_registry.yaml，通过 static_train.py --models random_monkey_Alpha158 训练。

批量统计分析：
    由 monkey_benchmark.py 直接调用，跳过 MLflow 开销。
"""

import numpy as np
import pandas as pd

from qlib.model.base import Model
from qlib.data.dataset.handler import DataHandlerLP


class RandomModel(Model):
    """对每只股票生成均匀随机分数的伪模型。

    Args:
        seed: 随机种子。None 则每次运行产生不同结果。
    """

    def __init__(self, seed=None, **kwargs):
        self.seed = seed
        self.fitted = True

    def fit(self, dataset, evals_result=None, save_path=None):
        """猴子不需要学习。"""
        if evals_result is not None and isinstance(evals_result, dict):
            evals_result["train"] = [0.0]
            evals_result["valid"] = [0.0]

    def predict(self, dataset, segment="test"):
        """为 test segment 中每个 (datetime, instrument) 生成随机分数。

        Returns:
            pd.Series: index=(datetime, instrument), values=U(0,1) 随机分数。
        """
        dl_test = dataset.prepare(
            segment, col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_I,
        )
        dl_test.config(fillna_type="ffill+bfill")
        index = dl_test.get_index()

        rng = np.random.default_rng(self.seed)
        scores = rng.uniform(0.0, 1.0, size=len(index))

        return pd.Series(scores, index=index, name="score")
