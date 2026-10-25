"""Diagnostic backtests of fixed, already saved model and ensemble signals."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from quantpits.research.paper_market import PaperMarket, load_signal, strict_json
from quantpits.research.replay import _prediction_frame
from quantpits.training.persistence import atomic_write_bytes
from quantpits.training.records import parse_training_record_document
from quantpits.utils.workspace import fingerprint_file


def evaluation_options(start=None, end=None, account=100000):
    for value in (start, end):
        if value is not None and pd.Timestamp(value).strftime("%Y-%m-%d") != value:
            raise ValueError("evaluation dates must use YYYY-MM-DD")
    if start and end and start > end:
        raise ValueError("evaluation start is after end")
    if isinstance(account, bool) or not math.isfinite(float(account)) or float(account) <= 0:
        raise ValueError("diagnostic initial cash must be finite and positive")
    return {"start_date": start, "end_date": end, "account": float(account)}


def _inside(root, value):
    path = Path(value).resolve()
    path.relative_to(root.resolve())
    return path


def _json(path, value):
    atomic_write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def _verify_proof(ctx, proof, target):
    if proof.get("source") != str(target):
        raise ValueError("fixed source proof selects a different snapshot")
    if str(target) not in {item["path"] for item in proof["files"]}:
        raise ValueError("fixed snapshot has no original digest")
    for item in proof["files"]:
        path = _inside(ctx.root, item["path"])
        if fingerprint_file(path) != item["sha256"]:
            raise ValueError("fixed source proof digest mismatch: %s" % path)


def _read_prediction(ctx, source, anchor, checksum):
    _, identity = load_signal(ctx, source, anchor)
    path = _inside(ctx.root, identity["path"])
    raw = path.read_bytes()
    from quantpits.training.persistence import sha256_bytes
    if sha256_bytes(raw) != identity["sha256"] or identity["sha256"] != checksum:
        raise ValueError("saved prediction digest differs from the fixed cycle")
    decoded = _prediction_frame(raw, "saved signal")
    pairs = [(pd.Timestamp(day), instrument, value) for day, rows in decoded.items() for instrument, value in rows]
    index = pd.MultiIndex.from_tuples([(d, i) for d, i, _ in pairs], names=["datetime", "instrument"])
    score = pd.Series(pd.to_numeric([v for _, _, v in pairs], errors="raise"), index=index, name="score")
    if np.isinf(score.to_numpy()).any():
        raise ValueError("saved prediction contains infinite scores")
    dates = sorted(set(score.dropna().index.get_level_values("datetime").strftime("%Y-%m-%d")))
    if not dates:
        raise ValueError("saved prediction has no finite scores")
    facts = {key: identity[key] for key in ("recorder_id", "experiment_id", "path", "sha256")}
    facts.update(start=dates[0], end=dates[-1], rows=len(score), sessions=len(dates),
                 instruments=int(score.index.get_level_values("instrument").nunique()),
                 prediction_start=min(decoded), prediction_end=max(decoded))
    return score, dates, facts


def research_conditions(ctx, account):
    from quantpits.utils.strategy import load_strategy_config, get_backtest_config
    config = load_strategy_config(workspace_path=str(ctx.root))
    backtest = get_backtest_config(config)
    backtest["account"] = account
    model = strict_json(ctx.config_path("model_config.json").read_bytes())
    return {"initial_cash": account, "initial_positions": {}, "initial_nav": 1.0,
            "strategy": copy.deepcopy(config["strategy"]), "backtest": copy.deepcopy(backtest),
            "freq": model.get("freq", "day"), "benchmark": model.get("benchmark", "SH000300")}


def initialize_market(ctx):
    import qlib
    qlib.init(provider_uri=str(ctx.qlib_data_dir), region=ctx.qlib_region, exp_manager={
        "class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
        "kwargs": {"uri": ctx.mlflow_uri, "default_exp_name": "Experiment"}})


def resolve_exchange_conditions(conditions):
    """Record native defaults explicitly after the selected market is initialized."""
    import inspect
    from qlib.config import C
    from qlib.backtest.exchange import Exchange
    exchange = conditions["backtest"]["exchange_kwargs"]
    parameters = inspect.signature(Exchange.__init__).parameters
    for name in ("open_cost", "close_cost", "min_cost", "impact_cost", "volume_threshold"):
        if name in parameters:
            exchange.setdefault(name, copy.deepcopy(parameters[name].default))
    for name in ("deal_price", "limit_threshold"):
        if exchange.get(name) is None:
            exchange[name] = copy.deepcopy(getattr(C, name))
    exchange.setdefault("trade_unit", C.trade_unit)


def backtest_saved_signal(score, conditions, start, end):
    """Create a fresh strategy, exchange and empty Account for each saved signal."""
    from qlib.backtest.exchange import Exchange
    from quantpits.utils.strategy import create_backtest_strategy
    from quantpits.utils.backtest_utils import run_backtest_with_strategy
    config = copy.deepcopy(conditions)
    strategy = create_backtest_strategy(score, {"strategy": config["strategy"]})
    exchange = config["backtest"]["exchange_kwargs"]
    exchange_freq = exchange.pop("freq", "day")
    trade_exchange = Exchange(freq=exchange_freq, start_time=start, end_time=end,
        codes=sorted(score.index.get_level_values("instrument").unique()), **exchange)
    report, _ = run_backtest_with_strategy(strategy, trade_exchange, config["freq"],
        config["initial_cash"], start, end, benchmark=config["benchmark"])
    return report


def summarize_report(report, conditions, start, end):
    """Keep initial cash and the first bar's costs in NAV and drawdown."""
    from quantpits.ensemble.risk_report import calculate_safe_risk
    if not isinstance(report, pd.DataFrame) or report.empty or not {"account", "return", "cost", "turnover"} <= set(report):
        raise ValueError("backtest returned no usable account/cost/turnover trajectory")
    report = report.copy()
    report.index = pd.to_datetime(report.index)
    if not report.index.is_unique or not report.index.is_monotonic_increasing:
        raise ValueError("backtest report dates must be unique and ordered")
    if report.index.min() < pd.Timestamp(start) or report.index.max() > pd.Timestamp(end):
        raise ValueError("backtest report is outside the requested window")
    if not np.isfinite(report[["account", "return", "cost", "turnover"]].to_numpy(dtype=float)).all():
        raise ValueError("backtest report contains nonfinite account metrics")
    initial = conditions["initial_cash"]
    previous = report["account"].shift(1)
    previous.iloc[0] = initial
    report["net_return"] = report["account"] / previous - 1
    if not np.allclose(report["net_return"], report["return"] - report["cost"], atol=1e-9, rtol=1e-7):
        raise ValueError("account returns do not reconcile with gross returns minus costs")
    report["nav"] = report["account"] / initial
    report["drawdown"] = report["nav"] / report["nav"].cummax().clip(lower=1.0) - 1
    report["fees"] = previous * report["cost"]
    risk = calculate_safe_risk(report["net_return"], conditions["freq"])
    def finite(value):
        return float(value) if value is not None and math.isfinite(float(value)) else None
    return report, {"initial_cash": initial, "initial_nav": 1.0,
        "first_close_nav": float(report["nav"].iloc[0]), "first_period_fees": float(report["fees"].iloc[0]),
        "final_account": float(report["account"].iloc[-1]), "final_nav": float(report["nav"].iloc[-1]),
        "net_return": float(report["nav"].iloc[-1] - 1), "max_drawdown": float(report["drawdown"].min()),
        "fees": float(report["fees"].sum()), "sum_period_turnover": float(report["turnover"].sum()),
        "benchmark_return": (finite((1 + report["bench"]).prod() - 1) if "bench" in report
                             and np.isfinite(report["bench"].to_numpy(dtype=float)).all() else None),
        "annualized_arithmetic_return": finite(risk.get("annualized_return")),
        "first_report_bar": report.index.min().strftime("%Y-%m-%d"),
        "last_report_bar": report.index.max().strftime("%Y-%m-%d"), "report_periods": len(report)}


def save_group_result(output, result):
    _json(output / "result.json", result)
    rows = []
    for row in result["members"]:
        metrics = row.get("metrics", {})
        source = row.get("source", {})
        rows.append({"name": row["name"], "kind": row["kind"], "status": row["status"],
                     "error": row.get("error"), "source_recorder": source.get("recorder_id"),
                     "source_start": source.get("start"), "source_end": source.get("end"),
                     "source_instruments": source.get("instruments"),
                     "selected_signal_sessions": row.get("selected_signal_sessions"),
                     "missing_dates": ",".join(row.get("missing_dates", [])), **metrics})
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    lines = ["# Saved-signal diagnostic comparison", "", "Group: " + result["signal"],
        "Status: " + result["status"], "", "Conditions (fresh empty account per signal):", "",
        "```json", json.dumps(result.get("conditions"), ensure_ascii=False, indent=2), "```", "",
        "Window: " + json.dumps(result.get("window")), "",
        "| Signal | Status | Source coverage | Instruments | Selected sessions | Report periods | Final NAV | Net return | Max drawdown | Fees |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in result["members"]:
        m = row.get("metrics", {})
        source = row.get("source", {})
        lines.append("| %s | %s | %s..%s | %s | %s | %s | %s | %s | %s | %s |" % (
            row["name"].replace("|", "\\|").replace("\n", " "), row["status"], source.get("start", "—"), source.get("end", "—"),
            source.get("instruments", "—"), row.get("selected_signal_sessions", "—"), m.get("report_periods", "—"),
            m.get("final_nav", "—"), m.get("net_return", "—"), m.get("max_drawdown", "—"), m.get("fees", "—")))
    for row in result["members"]:
        source = row.get("source", {})
        lines += ["", "%s: recorder=%s; missing dates=%s; error=%s" % (row["name"].replace("\n", " "),
            source.get("recorder_id", "—"), ", ".join(row.get("missing_dates", [])) or "—", row.get("error", "—"))]
    lines += ["", "Historical diagnostic replay, separate from continuous actual/paper NAV.",
        "No cross-group ranking. Qlib uses its native signal lag and report-bar labels.",
        "Only the selected saved signal dates are supplied; the first bar can precede the first trade.",
        "IC/RankIC and labels are not recomputed; source record metrics remain their original observations.", ""]
    atomic_write_bytes(output / "comparison.md", "\n".join(lines).encode())


def evaluate_saved_group(ctx, request, output, *, backtest=None, initialize=None):
    """Preserve the full requested member list, including failures and gaps."""
    output = Path(output).resolve()
    for root in (ctx.root, ctx.qlib_data_dir):
        if output == root.resolve() or root.resolve() in output.parents:
            raise ValueError("evaluation output must be outside account/provider trees")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise ValueError("evaluation output already exists; select a new directory")
    options = evaluation_options(request["options"].get("start_date"), request["options"].get("end_date"),
                                 request["options"].get("account", 100000))
    members = [{"name": name, "kind": "model", "status": "pending"} for name in request["models"]]
    members.append({"name": request["combo"], "kind": "ensemble", "status": "pending"})
    result = {"schema_version": 1, "signal": request["signal"], "status": "failed", "members": members,
              "options": options, "ic": None, "ic_status": "not_recomputed"}
    signals, dates = {}, {}
    try:
        directory = _inside(ctx.root, request["source_directory"])
        _verify_proof(ctx, request["prediction_proof"], directory / "predictions.json")
        _verify_proof(ctx, request["ensemble_proof"], directory / "ensemble.json")
        predictions = strict_json((directory / "predictions.json").read_bytes())
        parse_training_record_document(predictions)
        ensemble = strict_json((directory / "ensemble.json").read_bytes())
        if set(predictions["models"]) != set(request["models"]) or ensemble["source_recorders"] != predictions["models"]:
            raise ValueError("fixed prediction/ensemble membership differs from the requested group")
        if predictions["anchor_date"] != request["anchor"] or ensemble["anchor_date"] != request["anchor"]:
            raise ValueError("fixed prediction/ensemble anchor differs from the cycle")
        proofs = {m["key"]: m for m in request["prediction_proof"]["members"]}
        for index, row in enumerate(members):
            try:
                if row["kind"] == "model":
                    record = predictions["model_records"][row["name"]]
                    if proofs[row["name"]]["recorder_id"] != record["recorder_id"]:
                        raise ValueError("fixed model proof selects a different recorder")
                    source = {"recorder_id": record["recorder_id"], "experiment_id": record["experiment_id"]}
                    checksum = proofs[row["name"]]["sha256"]
                else:
                    record = ensemble["combo_meta"][request["combo"]]
                    source = {"recorder_id": ensemble["combos"][request["combo"]], "experiment_id": record["experiment_id"]}
                    if record["recorder_id"] != source["recorder_id"]:
                        raise ValueError("fixed ensemble recorder identity differs")
                    checksum = ensemble["native_prediction"]["sha256"]
                score, sessions, facts = _read_prediction(ctx, source, request["anchor"], checksum)
                if row["kind"] == "model" and (facts["prediction_start"], facts["prediction_end"], facts["rows"]) != (
                    record["prediction_start"], record["prediction_end"], record["prediction_rows"]
                ):
                    raise ValueError("saved model coverage differs from its fixed record")
                row.update(status="loaded", source=facts)
                signals[index], dates[index] = score, sessions
            except Exception as exc:
                row.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        if len(signals) != len(members):
            raise ValueError("a requested signal is unavailable; common group window cannot be established")
        start = options["start_date"] or max(values[0] for values in dates.values())
        end = options["end_date"] or min(values[-1] for values in dates.values())
        if start > end:
            raise ValueError("requested signals have no common usable interval")
        calendar = PaperMarket(ctx.qlib_data_dir).calendar
        sessions = [day for day in calendar if start <= day <= end]
        result["window"] = {"start": start, "end": end, "trading_sessions": len(sessions)}
        if not sessions or start not in calendar or end not in calendar:
            raise ValueError("evaluation boundaries are outside the actual trading calendar")
        conditions = research_conditions(ctx, options["account"])
        result["conditions"] = conditions
        ready = []
        for index, row in enumerate(members):
            missing = sorted(set(sessions) - set(dates[index]))
            row["missing_dates"] = missing
            row["selected_signal_sessions"] = len(set(sessions) & set(dates[index]))
            if missing:
                row.update(status="unavailable", error="saved signal does not cover the requested dates")
            else:
                ready.append(index)
        if ready:
            (initialize or initialize_market)(ctx)
            resolve_exchange_conditions(conditions)
        for index in ready:
            row = members[index]
            try:
                row["status"] = "running"
                score = signals[index]
                selected = score.loc[(score.index.get_level_values("datetime") >= start) &
                                     (score.index.get_level_values("datetime") <= end)].copy()
                report = (backtest or backtest_saved_signal)(selected, copy.deepcopy(conditions), start, end)
                report, metrics = summarize_report(report, conditions, start, end)
                filename = "signal-%03d.csv" % index
                report.to_csv(output / filename, index_label="datetime")
                row.update(status="complete", metrics=metrics, trajectory=filename)
            except Exception as exc:
                row.update(status="failed", error=str(exc), error_type=type(exc).__name__)
    except (KeyboardInterrupt, SystemExit):
        result["status"] = "interrupted"
        for row in members:
            if row["status"] == "running":
                row["status"] = "interrupted"
            elif row["status"] == "loaded":
                row["status"] = "pending"
        save_group_result(output, result)
        raise
    except Exception as exc:
        result["error"] = str(exc)
        for row in members:
            if row["status"] in ("pending", "loaded"):
                row.update(status="blocked", error=str(exc))
    result["status"] = ("complete" if all(r["status"] == "complete" for r in members) else
                        "partial" if any(r["status"] == "complete" for r in members) else "failed")
    save_group_result(output, result)
    return result
