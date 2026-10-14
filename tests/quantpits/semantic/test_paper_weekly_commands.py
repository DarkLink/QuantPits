"""Hermetic cross-command paper contract, with real temporary provider files."""
import json
import os
import struct
import subprocess
import sys


def test_paper_commands_preserve_cycle_and_account_across_processes(tmp_path):
    root, provider = tmp_path / "account", tmp_path / "market"
    for folder in (root / "config", provider / "calendars", provider / "instruments"):
        folder.mkdir(parents=True)
    sessions = ["2024-05-10", "2024-05-13", "2024-05-17"]
    (provider / "calendars/day.txt").write_text(sessions[0] + "\n")
    (provider / "calendars/day_future.txt").write_text("\n".join(sessions) + "\n")
    (provider / "instruments/selected.txt").write_text("SH600001\t2024-01-01\t2024-12-31\n")
    features = provider / "features/sh600001"
    features.mkdir(parents=True)
    for field, values in {"close": [10, 10, 11], "open": [10, 10.3, 10], "factor": [1, 1, 1]}.items():
        (features / (field + ".day.bin")).write_bytes(struct.pack("<4f", 0, *values))
    source = tmp_path / "actual.json"
    source.write_text(json.dumps({"current_date": sessions[0], "current_cash": 2100, "current_holding": []}))
    signal = tmp_path / "signal.csv"
    signal.write_text("datetime,instrument,score\n2024-05-10,SH600001,1\n")
    (root / "config/model_config.json").write_text(json.dumps({"market": "selected"}))
    (root / "config/strategy_config.yaml").write_text(
        "strategy:\n  name: topk_dropout\n  params:\n    topk: 1\n    n_drop: 0\n    buy_suggestion_factor: 2\n")
    (root / "config/paper_config.json").write_text(json.dumps({
        "account_id": "example", "signal_source": {"workspace": str(tmp_path), "prediction": str(signal)},
        "execution": {"base_fee": "4", "base_rate": "0", "sell_tax_rate": "0.0007",
                      "transfer_rates": {"SH": "0.00002", "SZ": "0"}, "lot_size": 100},
        "market_rules": {"source": "synthetic dated declaration", "instruments": {},
                         "boards": {"SH_MAIN": {"normal_limit": True, "limit_rate": "0.10"}}}}))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("QLIB_WORKSPACE_DIR", None)
    environment.pop("MLFLOW_TRACKING_URI", None)
    def command(*arguments, exit_code=0):
        result = subprocess.run([sys.executable, "-m", "quantpits.scripts.paper_trade", "--workspace", str(root),
                                 "--qlib-data-dir", str(provider)] + list(arguments),
                                env=environment, capture_output=True, text=True, timeout=30)
        assert result.returncode == exit_code, result.stderr
        return json.loads(result.stdout if exit_code != 1 else result.stderr)
    command("bootstrap", "--source-config", str(source), "--date", sessions[0])
    plan = command("plan", "--cycle", "same", "--anchor", sessions[0], "--trade-date", sessions[1])["plan"]
    journal = root / "data/paper/account.json"
    before = journal.read_bytes()
    assert command("settle", "--cycle", "same", exit_code=2)["status"] == "WAITING_FOR_DATA"
    assert journal.read_bytes() == before
    command("plan", "--cycle", "same", "--anchor", sessions[0], "--trade-date", sessions[2], exit_code=1)
    assert journal.read_bytes() == before
    # Frozen signal content survives a later source file update.
    signal.write_text("datetime,instrument,score\n2024-05-10,SH600001,-999\n")
    (provider / "calendars/day.txt").write_text("\n".join(sessions) + "\n")
    result = command("settle", "--cycle", "same")["settlement"]
    assert result["decisions"][0]["quantity"] == 200
    assert result["decisions"][0]["suggested_quantity"] == plan["buys"][0]["suggested_quantity"] == 100
    assert command("settle", "--cycle", "same")["settlement"] == result
    assert command("value", "--date", sessions[2])["valuation"]["nav"] == "2235.96"
    before_report = journal.read_bytes()
    report = command("report", "--date", sessions[2], "--since", sessions[0], "--save")
    assert report["paper_period"]["pnl_excluding_external"] == "135.96"
    assert journal.read_bytes() == before_report
    assert (root / "output/paper/report_2024-05-17.md").exists()
    assert not (root / "mlflow.db").exists()
