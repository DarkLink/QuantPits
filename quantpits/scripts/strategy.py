#!/usr/bin/env python
"""
Strategy Provider 策略工厂模块

提供统一的回测策略和订单生成策略管理，解耦代码与具体策略实现。

暴露的核心接口：
1. load_strategy_config(): 加载策略配置文件（包含回退逻辑）
2. create_backtest_strategy(signal, config_dict=None): 生产 Qlib 回测策略实例
3. create_order_generator(config_dict=None): 生产与回测策略匹配的订单生成器
4. get_backtest_config(config_dict=None): 获取回测环境参数
5. get_strategy_params(config_dict=None): 获取策略自身参数
6. generate_port_analysis_config(config_dict=None, freq='day'): 生成给 workflow_config.yaml 注入用的字典
"""

import os
import yaml
import json
import importlib
import pandas as pd
import numpy as np
import env


# ==============================================================================
# 配置文件加载
# ==============================================================================
def load_strategy_config():
    """
    加载策略配置，首选 config/strategy_config.yaml
    如果不存在，Fallback 到遗留的 model_config.json 和 prod_config.json（将来弃用）
    """
    strategy_yaml = os.path.join(env.ROOT_DIR, "config", "strategy_config.yaml")
    
    # 默认值
    config = {
        "strategy": {
            "name": "topk_dropout",
            "params": {
                "topk": 20,
                "n_drop": 3,
                "only_tradable": True,
                "buy_suggestion_factor": 2
            }
        },
        "backtest": {
            "account": 100_000_000,
            "exchange_kwargs": {
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5
            }
        }
    }
    
    if os.path.exists(strategy_yaml):
        with open(strategy_yaml, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                if "strategy" in yaml_config:
                    config["strategy"].update(yaml_config["strategy"])
                if "backtest" in yaml_config:
                    config["backtest"].update(yaml_config["backtest"])
    else:
        print("⚠️  [Strategy Provider] Warning: `config/strategy_config.yaml` not found.")
        print("   Falling back to legacy JSON configs (`model_config.json` / `prod_config.json`).")
        print("   Please consider migrating to `strategy_config.yaml` as JSON parameters are deprecated.")
        
        prod_json = os.path.join(env.ROOT_DIR, "config", "prod_config.json")
        model_json = os.path.join(env.ROOT_DIR, "config", "model_config.json")
        
        topk, ndrop, bs_factor = 20, 3, 2
        
        if os.path.exists(model_json):
            with open(model_json, "r") as f:
                mc = json.load(f)
                if "TopK" in mc: topk = mc["TopK"]
                if "DropN" in mc: ndrop = mc["DropN"]
                
        if os.path.exists(prod_json):
            with open(prod_json, "r") as f:
                pc = json.load(f)
                if "TopK" in pc: topk = pc["TopK"]
                if "DropN" in pc: ndrop = pc["DropN"]
                if "buy_suggestion_factor" in pc: bs_factor = pc["buy_suggestion_factor"]
                
        config["strategy"]["params"]["topk"] = topk
        config["strategy"]["params"]["n_drop"] = ndrop
        config["strategy"]["params"]["buy_suggestion_factor"] = bs_factor

    return config


# ==============================================================================
# 订单生成器基类和具体实现
# ==============================================================================
class OrderGenerator:
    """与 Qlib 回测 Strategy 逻辑对齐的订单生成器基类"""
    
    def analyze_positions(self, pred_df, price_df, current_holding):
        """
        根据预测和价格数据，确定继续持有/卖出/买入候选
        返回: hold_final, sell_candidates, buy_candidates, merged_df, buy_count
        """
        raise NotImplementedError
        
    def generate_sell_orders(self, sell_candidates, current_holding, next_trade_date_string):
        """生成具体的卖出定单列表"""
        raise NotImplementedError
        
    def generate_buy_orders(self, buy_candidates, buy_count, available_cash, next_trade_date_string):
        """生成具体的买入定单列表"""
        raise NotImplementedError


class TopkDropoutOrderGenerator(OrderGenerator):
    """对应 TopkDropoutStrategy 的订单生成"""
    
    def __init__(self, topk=20, n_drop=3, buy_suggestion_factor=3, **kwargs):
        self.topk = topk
        self.drop_n = n_drop
        self.buy_suggestion_factor = buy_suggestion_factor
    
    def analyze_positions(self, pred_df, price_df, current_holding):
        top_k = self.topk
        drop_n = self.drop_n
        buy_suggestion_factor = self.buy_suggestion_factor

        latest_date = pred_df.index.get_level_values('datetime').max()
        if len(pred_df.index.get_level_values('datetime').unique()) > 1:
            daily_pred = pred_df.xs(latest_date, level='datetime')
        else:
            daily_pred = pred_df

        if 'instrument' in daily_pred.columns:
            pred_reset = daily_pred
        else:
            pred_reset = daily_pred.reset_index()

        price_reset = price_df.reset_index()

        merged_df = pd.merge(
            pred_reset, price_reset,
            on='instrument', how='inner'
        )

        sorted_df = merged_df.sort_values(by='score', ascending=False).set_index('instrument')

        current_holding_instruments = [h['instrument'] for h in current_holding]

        top_k_candidates = sorted_df.head(top_k + drop_n * buy_suggestion_factor)

        hold_candidates = top_k_candidates[
            top_k_candidates.index.isin(current_holding_instruments)
        ]

        current_holding_in_df = sorted_df[sorted_df.index.isin(current_holding_instruments)]
        sell_candidates = current_holding_in_df[
            ~current_holding_in_df.index.isin(hold_candidates.index)
        ].tail(drop_n)

        hold_final = current_holding_in_df[
            ~current_holding_in_df.index.isin(sell_candidates.index)
        ]

        buy_count = top_k - len(hold_final)
        buy_candidates = top_k_candidates[
            ~top_k_candidates.index.isin(hold_final.index)
        ].head(max(0, buy_count * buy_suggestion_factor))

        return hold_final, sell_candidates, buy_candidates, sorted_df, buy_count

    def generate_sell_orders(self, sell_candidates, current_holding, next_trade_date_string):
        holdings_dict = {h['instrument']: float(h['value']) for h in current_holding}
        sell_orders = []
        sell_amount = 0

        for instrument, row in sell_candidates.iterrows():
            if instrument in holdings_dict:
                value = holdings_dict[instrument]
                amount = value * row['possible_min']
                sell_amount += amount
                sell_orders.append({
                    'instrument': instrument,
                    'datetime': next_trade_date_string,
                    'value': int(value),
                    'estimated_amount': round(amount, 2),
                    'score': round(row['score'], 6),
                    'current_close': round(row['current_close'], 2),
                })

        return sell_orders, sell_amount

    def generate_buy_orders(self, buy_candidates, buy_count, available_cash, next_trade_date_string):
        avg_cash = available_cash / buy_count if buy_count > 0 else 0
        buy_orders = []
        
        for instrument, row in buy_candidates.iterrows():
            value = int(np.floor(avg_cash / row['possible_max'] / 100) * 100)
            if value >= 100:
                amount = value * row['possible_max']
                buy_orders.append({
                    'instrument': instrument,
                    'datetime': next_trade_date_string,
                    'value': value,
                    'estimated_amount': round(amount, 2),
                    'score': round(row['score'], 6),
                    'current_close': round(row['current_close'], 2),
                })

        return buy_orders


# ==============================================================================
# 策略注册表和 Provider 工厂方法
# ==============================================================================
STRATEGY_REGISTRY = {
    "topk_dropout": {
        "backtest_class": "qlib.contrib.strategy.TopkDropoutStrategy",
        "order_generator_class": TopkDropoutOrderGenerator,
    }
}


def _get_strategy_config(config_dict=None):
    if config_dict is None:
        return load_strategy_config()
    return config_dict


def get_strategy_params(config_dict=None):
    """获取策略内部参数"""
    config = _get_strategy_config(config_dict)
    return config["strategy"]["params"].copy()


def get_backtest_config(config_dict=None, fallback_freq="day"):
    """获取回测账号、手续费等与策略逻辑无关的设置"""
    config = _get_strategy_config(config_dict)
    bt_conf = config["backtest"].copy()
    if "freq" not in bt_conf.get("exchange_kwargs", {}):
        bt_conf.setdefault("exchange_kwargs", {})["freq"] = fallback_freq
    return bt_conf


def create_order_generator(config_dict=None) -> OrderGenerator:
    """实例化订单生成器"""
    config = _get_strategy_config(config_dict)
    name = config["strategy"]["name"]
    params = config["strategy"]["params"]
    
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{name}' not found in STRATEGY_REGISTRY")
        
    cls = STRATEGY_REGISTRY[name]["order_generator_class"]
    return cls(**params)


def create_backtest_strategy(signal, config_dict=None):
    """实例化 Qlib 回测策略，自动剔除 Qlib 不支持的自用参数"""
    config = _get_strategy_config(config_dict)
    name = config["strategy"]["name"]
    params = config["strategy"]["params"].copy()
    
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{name}' not found in STRATEGY_REGISTRY")
        
    # 移除 order generator 专用的参数避免传给 Qlib 报错
    params.pop("buy_suggestion_factor", None)
    
    class_path = STRATEGY_REGISTRY[name]["backtest_class"]
    module_name, class_name = class_path.rsplit(".", 1)
    
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    
    # 将信号传入
    return cls(signal=signal, **params)


def generate_port_analysis_config(config_dict=None, freq="day"):
    """
    生成供 workflow_config.yaml 注入用的 port_analysis_config 字典。
    用于在 train_single_model 时生成相同的配套测试环境。
    """
    config = _get_strategy_config(config_dict)
    st_conf = config["strategy"]
    bt_conf = config["backtest"]
    
    name = st_conf["name"]
    params = st_conf["params"].copy()
    params.pop("buy_suggestion_factor", None)
    
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{name}' not found in STRATEGY_REGISTRY")
        
    strategy_class_path = STRATEGY_REGISTRY[name]["backtest_class"]
    strategy_module, strategy_class = strategy_class_path.rsplit(".", 1)
    
    # 构造与 Qlib YAML 格式匹配的 port_analysis_config
    pa_config = {
        "strategy": {
            "class": strategy_class,
            "module_path": strategy_module,
            "kwargs": {
                "signal": "<PRED>",
                **params
            }
        },
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": freq,
                "generate_portfolio_metrics": True,
                "verbose": False
            }
        },
        "backtest": {
            "start_time": "<TBD>",
            "end_time": "<TBD>",
            "account": bt_conf["account"],
            "benchmark": "<TBD>",
            "exchange_kwargs": bt_conf["exchange_kwargs"]
        }
    }
    
    return pa_config
