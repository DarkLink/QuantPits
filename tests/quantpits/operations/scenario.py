"""Three independent accounts, two signal producers, temporary local backends."""
import json
import struct

import pandas as pd
import yaml

from quantpits.research.paper_workflow import PaperWorkflow
from quantpits.utils.workspace import WorkspaceContext


SESSIONS = ["2024-05-10", "2024-05-13", "2024-05-17", "2024-05-20", "2024-05-21"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def broker(root):
    from quantpits.scripts.brokers.base import REQUIRED_COLUMNS
    frame = pd.DataFrame([{ "证券代码": "", "交易类别": "利息归本", "成交价格": 0,
        "成交数量": 0, "成交金额": 0, "资金发生数": 7, "交收日期": "2024-05-13", "费用合计": 0}], columns=REQUIRED_COLUMNS + ["费用合计"])
    for selected in SESSIONS[1:3]:
        (frame if selected == SESSIONS[1] else frame.iloc[:0]).to_excel(
            root / "data" / (selected + "-table.xlsx"), sheet_name="Sheet1", startrow=5, index=False)


def scenario(tmp_path, *, with_broker=True):
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    (provider / "calendars/day.txt").write_text("\n".join(SESSIONS[:3]) + "\n")
    (provider / "calendars/day_future.txt").write_text("\n".join(SESSIONS) + "\n")
    (provider / "instruments/selected.txt").write_text("SH600001\t2024-01-01\t2024-12-31\nSH600002\t2024-01-01\t2024-12-31\n")
    for instrument in ("sh600001", "sh600002"):
        directory = provider / "features" / instrument
        directory.mkdir(parents=True)
        for field in ("open", "close", "factor"):
            (directory / (field + ".day.bin")).write_bytes(struct.pack("<6f", 0, *([1 if field == "factor" else 10] * 5)))
    members = []
    for name, cash in (("live", 3000), ("paper_a", 2100), ("paper_b", 4100)):
        root = tmp_path / name
        (root / "config").mkdir(parents=True)
        (root / "data").mkdir()
        (root / "output").mkdir()
        write_json(root / "config/model_config.json", {"market": "selected", "benchmark": "SH000300", "freq": "week",
            "train_date_mode": "fixed", "current_date": SESSIONS[0]})
        (root / "config/strategy_config.yaml").write_text(
            "strategy:\n  name: topk_dropout\n  params:\n    topk: 1\n    n_drop: 0\n    buy_suggestion_factor: 2\n")
        write_json(root / "config/prod_config.json", {"current_date": SESSIONS[0], "last_processed_date": SESSIONS[0],
            "current_cash": cash, "current_holding": [], "initial_cash": cash, "initial_holding": [], "broker": "gtja"})
        write_json(root / "config/cashflow.json", {"cashflows": {}})
        (root / "config/model_registry.yaml").write_text("models:\n  small:\n    enabled: true\n    yaml_file: small.yaml\n  unused:\n    enabled: true\n    yaml_file: unused.yaml\n")
        (root / "config/small.yaml").write_text("model: {}\n")
        (root / "config/unused.yaml").write_text("model: {}\n")
        write_json(root / "fit.json", {"experiment_name": "fit", "anchor_date": SESSIONS[0], "models": {"small@static": "original-fit"}})
        write_json(root / "config/ensemble_config.json", {"default_combo": "chosen", "combos": {"chosen": {"models": ["small"], "method": "equal", "enabled": True}}})
        members.append({"id": name, "kind": "actual" if name == "live" else "paper", "workspace": str(root),
            "mlflow_uri": (root / "mlruns").as_uri(), "qlib_data_dir": str(provider)})
        if name == "live":
            pd.DataFrame([{ "成交日期": SESSIONS[0], "收盘价值": cash, "CASHFLOW": 0}]).to_csv(root / "data/daily_amount_log_full.csv", index=False)
            pd.DataFrame([{ "成交日期": SESSIONS[0], "证券代码": "CASH", "持仓数量": cash, "收盘价值": cash}]).to_csv(root / "data/holding_log_full.csv", index=False)
            pd.DataFrame(columns=["成交日期", "费用合计", "交易类别"]).to_csv(root / "data/trade_log_full.csv", index=False)
            if with_broker:
                broker(root)
        else:
            signal = root / "old_signal.csv"
            signal.write_text("datetime,instrument,score\n2024-05-10,SH600001,2\n2024-05-10,SH600002,1\n")
            write_json(root / "config/paper_config.json", {"account_id": name,
                "signal_source": {"prediction": str(signal)},
                "execution": {"base_fee": "4", "base_rate": "0", "sell_tax_rate": "0.0007",
                              "transfer_rates": {"SH": "0.00002", "SZ": "0"}, "lot_size": 100},
                "market_rules": {"source": "synthetic explicit normal-day assumption", "instruments": {},
                                 "boards": {"SH_MAIN": {"normal_limit": True, "limit_rate": "0.1"}}}})
            workflow = PaperWorkflow(WorkspaceContext.from_root(root, mlflow_uri=(root / "mlruns").as_uri(), qlib_data_dir=provider))
            workflow.bootstrap(root / "config/prod_config.json", SESSIONS[0])
            workflow.plan("old_native_id", SESSIONS[0], SESSIONS[1])
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    group = project_dir / "group.yaml"
    group.write_text(yaml.safe_dump({"name": "Synthetic three accounts", "members": members}))
    path = project_dir / "project.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "group": "group.yaml",
        "signals": {"shared": {"producer": "live", "models": ["small"], "fit_records": "fit.json", "combo": "chosen"},
                    "candidate": {"producer": "paper_b", "models": ["small"], "fit_records": "fit.json", "combo": "chosen"}},
        "accounts": {"live": {"signal": "shared"}, "paper_a": {"signal": "shared"}, "paper_b": {"signal": "candidate"}}}))
    return path
