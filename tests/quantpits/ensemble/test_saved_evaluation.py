"""Saved signals remain immutable; diagnostic accounts and conditions are fresh."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantpits.ensemble.saved_evaluation import evaluate_saved_group, evaluation_options, summarize_report
from quantpits.training.records import ModelRecordEntry, TrainingRecordSnapshot
from quantpits.training.record_repository import TrainingRecordRepository
from quantpits.utils.workspace import WorkspaceContext, fingerprint_file


def signal(days, instruments=("SH600001", "SH600002"), offset=0):
    index = pd.MultiIndex.from_product([pd.to_datetime(days), instruments], names=["datetime", "instrument"])
    return pd.DataFrame({"score": np.arange(len(index), dtype=float) + offset}, index=index)


@pytest.fixture
def saved_case(tmp_path, monkeypatch):
    # Backtests/initialization are replaced in these unit cases; provide the
    # market defaults normally established by the real initialize_market call.
    from qlib.config import C
    for key, value in {"deal_price": "close", "limit_threshold": .095, "trade_unit": 100}.items():
        monkeypatch.setitem(C.__dict__["_config"], key, value)
    root = tmp_path / "account"
    (root / "config").mkdir(parents=True)
    (root / "config/model_config.json").write_text(json.dumps({"freq": "day", "benchmark": "SH000905"}))
    (root / "config/prod_config.json").write_text(json.dumps({"current_cash": 1234, "current_holding": []}))
    (root / "config/strategy_config.yaml").write_text(yaml.safe_dump({
        "strategy": {"name": "topk_dropout", "params": {"topk": 1, "n_drop": 0}},
        "backtest": {"account": 987654, "exchange_kwargs": {"deal_price": "open", "min_cost": 7}}}))
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    days = pd.bdate_range("2024-05-13", periods=5).strftime("%Y-%m-%d").tolist()
    for name in ("day.txt", "day_future.txt"):
        (provider / "calendars" / name).write_text("\n".join(days) + "\n")
    ctx = WorkspaceContext.from_root(root, mlflow_uri=(root / "mlruns").as_uri(), qlib_data_dir=provider)
    directory = root / "output/fixed"
    directory.mkdir(parents=True)
    request = {"signal": "shared", "models": ["one@static", "two@static"], "combo": "combo",
        "anchor": days[-1], "source_directory": str(directory),
        "options": evaluation_options(), "prediction_proof": {}, "ensemble_proof": {}}

    def prepare(frames=None):
        frames = frames or [signal(days), signal(days, ("SH600002", "SH600003"), 100),
                            signal(days, ("SH600001", "SH600002", "SH600003"), 1000)]
        entries, proofs = [], []
        for number, frame in enumerate(frames):
            artifacts = root / "artifacts" / str(number)
            artifacts.mkdir(parents=True, exist_ok=True)
            frame.to_pickle(artifacts / "pred.pkl")
            (artifacts / "model.pkl").write_bytes(b"MODEL MUST NEVER BE LOADED")
            (artifacts / "label.pkl").write_bytes(b"UNKNOWN LABELS ARE NOT NEEDED")
            (artifacts / "portfolio_analysis").mkdir(exist_ok=True)
            pd.DataFrame({"account": [9999999], "holding": ["different old position"]},
                index=pd.to_datetime(["2000-01-01"])).to_pickle(artifacts / "portfolio_analysis/port_analysis_1day.pkl")
            metadata = root / "mlruns/1" / ("rid" + str(number))
            metadata.mkdir(parents=True, exist_ok=True)
            (metadata.parent / "meta.yaml").write_text(yaml.safe_dump({"experiment_id": "1", "name": "saved", "lifecycle_stage": "active"}))
            (metadata / "meta.yaml").write_text(yaml.safe_dump({"run_id": metadata.name,
                "experiment_id": "1", "status": 3, "lifecycle_stage": "active", "artifact_uri": artifacts.as_uri()}))
            if number < 2:
                key = request["models"][number]
                entry = ModelRecordEntry(key, key.split("@")[0], "static", "predict_only", "ready", metadata.name,
                    "saved", experiment_id="1", artifact_path=str(artifacts.relative_to(root)),
                    requested_anchor=days[-1], prediction_start=frame.index.get_level_values("datetime").min().strftime("%Y-%m-%d"),
                    prediction_end=days[-1], prediction_rows=len(frame), dataset_test_end=days[-1],
                    source_recorder_id="fit-" + str(number), source_experiment_name="fit", source_operation="train")
                entries.append(entry)
                proofs.append({"key": key, "recorder_id": metadata.name, "sha256": fingerprint_file(artifacts / "pred.pkl")})
        (directory / "predictions.json").write_bytes(TrainingRecordRepository.serialize(TrainingRecordSnapshot(tuple(entries))))
        snapshot = json.loads((directory / "predictions.json").read_text())
        ensemble = {"source_recorders": snapshot["models"], "combos": {"combo": "rid2"},
            "anchor_date": days[-1],
            "combo_meta": {"combo": {"experiment_id": "1", "recorder_id": "rid2"}},
            "native_prediction": {"sha256": fingerprint_file(root / "artifacts/2/pred.pkl")}}
        (directory / "ensemble.json").write_text(json.dumps(ensemble))
        for proof, filename in (("prediction_proof", "predictions.json"), ("ensemble_proof", "ensemble.json")):
            path = directory / filename
            request[proof] = {"source": str(path), "files": [{"path": str(path), "sha256": fingerprint_file(path)}]}
        request["prediction_proof"]["members"] = proofs
        return frames

    frames = prepare()
    return {"root": root, "ctx": ctx, "request": request, "days": days, "prepare": prepare,
            "frames": frames, "output": tmp_path / "evaluation"}


def report_for(score, conditions, start, end):
    days = sorted(score.index.get_level_values("datetime").unique())
    amounts = conditions["initial_cash"] * .99 * np.power(1.02, np.arange(len(days)))
    return pd.DataFrame({"account": amounts, "return": [0] + [.02] * (len(days) - 1),
        "cost": [.01] + [0] * (len(days) - 1), "turnover": [.5] + [0] * (len(days) - 1)}, index=days)


def tree(root):
    return {str(p.relative_to(root)): fingerprint_file(p) for p in root.rglob("*") if p.is_file()}


def run(case, **kwargs):
    return evaluate_saved_group(case["ctx"], case["request"], case["output"],
        initialize=lambda ctx: None, backtest=kwargs.pop("backtest", report_for), **kwargs)


def test_comparison_uses_saved_raw_signals_common_cash_and_ignores_old_reports(saved_case, monkeypatch):
    case = saved_case
    (case["root"] / "latest_train_records.json").write_text('{"models":{"other@static":"later-model"}}')
    (case["root"] / "config/ensemble_records.json").write_text('{"combos":{"combo":"later-ensemble"}}')
    before = tree(case["root"])
    class NoRecorder:
        def get_recorder(self, *args, **kwargs):
            pytest.fail("no recorder/model load")
    monkeypatch.setattr("qlib.workflow.R", NoRecorder())
    monkeypatch.setattr("quantpits.utils.fusion_engine.generate_ensemble_signal", lambda *a, **kw: pytest.fail("no re-fusion"))
    observed = []
    def backtest(score, conditions, start, end):
        observed.append((score.copy(), copy.deepcopy(conditions)))
        result = report_for(score, conditions, start, end)
        conditions["strategy"]["params"]["topk"] = 999
        return result
    result = run(case, backtest=backtest)
    assert result["status"] == "complete", result
    assert [r["name"] for r in result["members"]] == ["one@static", "two@static", "combo"]
    for (score, config), original in zip(observed, case["frames"]):
        pd.testing.assert_series_equal(score, original["score"])
        assert config["initial_cash"] == 100000 and config["initial_positions"] == {}
        assert config["benchmark"] == "SH000905" and config["strategy"]["params"]["topk"] == 1
        assert config["backtest"]["exchange_kwargs"]["deal_price"] == "open"
    assert tree(case["root"]) == before
    assert result["ic"] is None and result["ic_status"] == "not_recomputed"
    exchange = result["conditions"]["backtest"]["exchange_kwargs"]
    assert {"open_cost", "close_cost", "min_cost", "deal_price", "trade_unit"} <= set(exchange)


def test_initial_nav_and_first_bar_fee_are_not_normalized_away(saved_case):
    result = run(saved_case)
    for row in result["members"]:
        m = row["metrics"]
        assert m["initial_nav"] == 1
        assert m["first_close_nav"] == pytest.approx(.99)
        assert m["first_period_fees"] == pytest.approx(1000)
        assert m["max_drawdown"] == pytest.approx(-.01)
        assert m["final_nav"] == pytest.approx(.99 * 1.02 ** 4)


def test_default_common_window_and_explicit_missing_coverage(saved_case):
    case = saved_case; days = case["days"]
    case["prepare"]([signal(days), signal(days[2:]), signal(days[1:])])
    result = run(case)
    assert result["window"] == {"start": days[2], "end": days[-1], "trading_sessions": 3}
    assert all(r["metrics"]["report_periods"] == 3 for r in result["members"])
    case["output"] = case["output"].parent / "explicit"
    case["request"]["options"] = evaluation_options(days[0], days[-1], 2000)
    result = run(case)
    assert result["status"] == "partial"
    assert result["members"][1]["missing_dates"] == days[:2]
    assert result["members"][1]["source"]["start"] == days[2]
    assert "metrics" not in result["members"][1]
    assert result["members"][0]["metrics"]["initial_cash"] == 2000
    table = (case["output"] / "comparison.md").read_text()
    assert "missing dates=" + ", ".join(days[:2]) in table
    assert "Source coverage" in table


def test_interior_nan_session_is_a_coverage_gap(saved_case):
    case = saved_case; days = case["days"]
    frames = [signal(days), signal(days), signal(days)]
    frames[1].loc[frames[1].index.get_level_values("datetime") == days[2], "score"] = np.nan
    case["prepare"](frames)
    result = run(case)
    assert result["status"] == "partial", result
    assert result["members"][1]["missing_dates"] == [days[2]]


def test_no_common_finite_window_is_disclosed(saved_case):
    case = saved_case; days = case["days"]
    frames = [signal(days), signal(days[1:]), signal(days)]
    frames[0].loc[frames[0].index.get_level_values("datetime") >= days[1], "score"] = np.nan
    case["prepare"](frames)
    result = run(case)
    assert result["status"] == "failed"
    assert "no common" in result["error"]


def test_missing_model_keeps_member_scope_and_does_not_select_latest(saved_case):
    case = saved_case
    (case["root"] / "artifacts/0/pred.pkl").unlink()
    (case["root"] / "latest_train_records.json").write_text('{"models": {"replacement": "later"}}')
    result = run(case)
    assert result["status"] == "failed"
    assert [r["name"] for r in result["members"]] == ["one@static", "two@static", "combo"]
    assert result["members"][0]["status"] == "failed"


def test_changed_snapshot_digest_is_rejected(saved_case):
    case = saved_case
    path = Path(case["request"]["source_directory"]) / "ensemble.json"
    path.write_text(path.read_text() + "\n")
    result = run(case)
    assert result["status"] == "failed" and "digest mismatch" in result["error"]


def test_changed_prediction_cannot_be_adopted_by_same_recorder_id(saved_case):
    case = saved_case
    path = case["root"] / "artifacts/0/pred.pkl"
    frame = pd.read_pickle(path); frame["score"] += 10
    frame.to_pickle(path)
    result = run(case)
    assert result["status"] == "failed" and "digest" in result["members"][0]["error"]


def test_backtest_failure_does_not_suppress_other_members(saved_case):
    calls = []
    def backtest(score, conditions, start, end):
        calls.append(score)
        if len(calls) == 1:
            raise RuntimeError("one backtest failed")
        return report_for(score, conditions, start, end)
    result = run(saved_case, backtest=backtest)
    assert result["status"] == "partial" and len(calls) == 3
    assert [r["status"] for r in result["members"]] == ["failed", "complete", "complete"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_backtest_control_interrupt_propagates(saved_case, interrupt):
    def stop(*args):
        raise interrupt("stop")
    with pytest.raises(interrupt):
        run(saved_case, backtest=stop)
    result = json.loads((saved_case["output"] / "result.json").read_text())
    assert result["status"] == "interrupted"
    assert [r["status"] for r in result["members"]] == ["interrupted", "pending", "pending"]


@pytest.mark.parametrize("account", [0, -1, float("nan"), float("inf"), True])
def test_invalid_diagnostic_cash_is_rejected(account):
    with pytest.raises(ValueError):
        evaluation_options(account=account)


def test_output_cannot_write_inside_source_account(saved_case):
    saved_case["output"] = saved_case["root"] / "evaluation"
    with pytest.raises(ValueError, match="outside"):
        run(saved_case)


def test_unreconciled_report_cannot_be_complete(saved_case):
    config = {"initial_cash": 1000, "freq": "day"}
    report = pd.DataFrame({"account": [999], "return": [0], "cost": [0], "turnover": [0]},
                          index=pd.to_datetime(["2024-05-13"]))
    with pytest.raises(ValueError, match="reconcile"):
        summarize_report(report, config, "2024-05-13", "2024-05-13")
