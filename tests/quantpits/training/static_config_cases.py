"""Synthetic inputs for old/new static configuration comparisons."""
import copy


def configuration_cases():
    handler = {
        "start_time": "2010-01-01", "end_time": "2015-12-31",
        "fit_start_time": "2010-01-01", "fit_end_time": "2013-12-31",
        "instruments": "original", "label": ["original-label"],
        "infer_processors": [{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature"}}],
        "learn_processors": [{"class": "DropnaLabel"}, {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}],
        "feature_extension": {"fields": ["$close", "Ref($close, 1)"], "names": ["CLOSE", "LAG"]},
    }
    portfolio = {}
    template = {
        "market": "original", "benchmark": "original-benchmark", "custom_root": {"keep": [1, 2]},
        "strategy": {"params": {"topk": 8, "n_drop": 1, "buy_suggestion_factor": 3, "custom_strategy": True}},
        "data_handler_config": handler, "port_analysis_config": portfolio,
        "task": {
            "model": {"class": "SyntheticModel", "module_path": "synthetic.models", "kwargs": {
                "seed": 17, "n_epochs": 20, "early_stop": 5, "metric": "ir", "topk": 7, "n_drop": 2,
                "unknown_model_option": {"nested": ["unchanged"]}}},
            "dataset": {"class": "DatasetH", "module_path": "qlib.data.dataset", "kwargs": {
                "handler": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler", "kwargs": handler},
                "segments": {"train": [], "valid": [], "test": [], "pretrain": [], "pretrain_validation": []},
                "unknown_dataset_option": ["preserved"],
            }},
            "custom_task": {"keep": True},
            "record": [
                {"class": "SignalRecord", "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"}},
                {"class": "SigAnaRecord"},
                {"class": "PortAnaRecord", "kwargs": {"config": portfolio}},
                {"class": "CustomRecord", "kwargs": {"extension": [1, 2]}},
            ],
        },
    }
    params = {"freq": "week", "market": "selected", "benchmark": "SH000300",
              "start_time": "2024-01-02", "end_time": "2024-05-17",
              "fit_start_time": "2024-01-02", "fit_end_time": "2024-03-29",
              "valid_start_time": "2024-04-01", "valid_end_time": "2024-04-12",
              "test_start_time": "2024-04-15", "test_end_time": "2024-05-17",
              "anchor_date": "2024-05-17", "account": 123456, "topk": 33, "n_drop": 4,
              "buy_suggestion_factor": 5}
    cases = []
    for name in ("week_alias", "day_override", "multi_label", "non_alias", "pretrain_found", "pretrain_missing", "no_pretrain", "configured_path"):
        item = {"name": name, "template": copy.deepcopy(template), "params": copy.deepcopy(params),
                "model_name": "synthetic", "no_pretrain": False, "resolved_path": "/synthetic/pretrain/base_latest.pkl"}
        if name == "day_override":
            item["params"].update(freq="day", label_formula="custom-label")
            item["params"].pop("account")
        if name == "multi_label":
            item["template"]["data_handler_config"]["label"] = ["label-one", "label-two"]
        if name == "non_alias":
            item["template"]["task"]["dataset"]["kwargs"]["handler"]["kwargs"] = copy.deepcopy(handler)
        if name.startswith("pretrain") or name in ("no_pretrain", "configured_path"):
            item["template"]["task"]["model"]["kwargs"].update(base_model="base", model_path="configured.pkl", d_feat=20)
        if name == "pretrain_missing": item["resolved_path"] = None
        if name == "no_pretrain": item["no_pretrain"] = True
        if name == "configured_path": item["model_name"] = None
        cases.append(item)
    return cases


def explicit_portfolio(freq):
    return {"strategy": {"class": "TopkDropoutStrategy", "kwargs": {"signal": "<PRED>", "topk": 22, "n_drop": 3}},
            "executor": {"class": "SimulatorExecutor", "kwargs": {"time_per_step": freq}},
            "backtest": {"account": 700000, "exchange_kwargs": {"deal_price": "close", "open_cost": .0005, "close_cost": .0015, "min_cost": 5}}}
