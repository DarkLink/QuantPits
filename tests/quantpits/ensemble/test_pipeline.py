from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quantpits.ensemble.pipeline import (
    SingleComboPipelineHooks,
    SingleComboPipelineRequest,
    run_single_combo_pipeline,
)


def _norm_frame(columns=("M1", "M2")):
    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    index = pd.MultiIndex.from_product(
        [dates, ["A"]],
        names=["datetime", "instrument"],
    )
    values = {
        name: [0.5 + idx * 0.1, 0.6 + idx * 0.1]
        for idx, name in enumerate(columns)
    }
    return pd.DataFrame(values, index=index)


def _args(**overrides):
    values = {
        "output_dir": "/tmp/ensemble-out",
        "prediction_dir": None,
        "workspace_root": "/tmp/workspace",
        "save_csv": False,
        "no_backtest": False,
        "verbose_backtest": False,
        "detailed_analysis": False,
        "freq": "day",
        "no_charts": False,
        "only_last_years": 0,
        "only_last_months": 0,
        "cli_args": ["--from-config"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _request(**overrides):
    values = {
        "combo_name": "combo_a",
        "selected_models": ["M1", "M2"],
        "method": "equal",
        "manual_weights_str": None,
        "norm_df": _norm_frame(),
        "model_metrics": {"M1": 0.4, "M2": 0.3},
        "loaded_models": ["M1", "M2"],
        "train_records": {"experiment_name": "exp"},
        "model_config": {"TopK": 20, "DropN": 2, "benchmark": "SH000905"},
        "ensemble_config": {},
        "anchor_date": "2020-01-02",
        "experiment_name": "exp",
        "args": _args(),
        "is_default": True,
    }
    values.update(overrides)
    return SingleComboPipelineRequest(**values)


def _leaderboard():
    return pd.DataFrame(
        {
            "annualized_return": [0.1255555, 0.1],
            "max_drawdown": [-0.0522222, -0.07],
            "information_ratio": [1.23456789, 0.8],
        },
        index=["Ensemble", "M1"],
    )


def _hooks(calls=None, *, report_df=None, leaderboard_df=None, contributions=None):
    calls = calls if calls is not None else []
    signal = pd.Series([0.2, 0.3], index=_norm_frame().index)
    report = report_df if report_df is not None else pd.DataFrame(
        {"account": [100.0, 110.0], "return": [0.0, 0.1], "bench": [0.0, 0.05]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )
    leaderboard = leaderboard_df if leaderboard_df is not None else _leaderboard()
    contribution_map = contributions if contributions is not None else {
        "M1": {"delta": 0.1111119, "loo_ic": 0.5, "full_ic": 0.6},
        "M2": {"delta": -0.05, "loo_ic": 0.7, "full_ic": 0.65},
    }

    def mark(name, value=None):
        def _inner(*args, **kwargs):
            calls.append((name, args, kwargs))
            return value

        return _inner

    return SingleComboPipelineHooks(
        correlation_analysis=mark("correlation", pd.DataFrame()),
        calculate_weights=mark("weights", (None, {"M1": 0.6, "M2": 0.4}, False)),
        generate_ensemble_signal=mark("signal", signal),
        save_predictions=mark("save", "pred.csv"),
        run_backtest=mark("backtest", (report, object())),
        run_detailed_backtest_analysis=mark("detailed", None),
        risk_analysis_and_leaderboard=mark("risk", ({"Ensemble": report}, leaderboard)),
        generate_charts=mark("charts", None),
        calculate_loo_contribution=mark("loo", contribution_map),
        save_model_contribution_snapshot=mark(
            "save_contribution",
            SimpleNamespace(path="/tmp/ensemble-out/model_contribution_combo_a_2020-01-02.json"),
        ),
        append_to_fusion_ledger=mark("ledger", None),
        get_workspace_root=lambda: "/tmp/workspace",
    )


def test_run_single_combo_pipeline_skips_when_no_valid_models():
    calls = []
    hooks = _hooks(calls)
    result = run_single_combo_pipeline(
        _request(selected_models=["missing"]),
        hooks=hooks,
    )

    assert result is None
    assert calls == []


def test_run_single_combo_pipeline_runs_full_stage_sequence():
    calls = []
    result = run_single_combo_pipeline(_request(), hooks=_hooks(calls))

    assert result is not None
    assert result.name == "combo_a"
    assert result.models == ("M1", "M2")
    assert result.to_legacy_dict()["models"] == ["M1", "M2"]
    assert [name for name, _args, _kwargs in calls] == [
        "correlation",
        "weights",
        "signal",
        "save",
        "backtest",
        "risk",
        "charts",
        "loo",
        "save_contribution",
        "ledger",
    ]


def test_run_single_combo_pipeline_skips_backtest_dependent_outputs():
    calls = []
    result = run_single_combo_pipeline(
        _request(args=_args(no_backtest=True)),
        hooks=_hooks(calls, contributions={}),
    )

    assert result is not None
    assert result.report_df is None
    assert result.leaderboard_df is None
    names = [name for name, _args, _kwargs in calls]
    assert "backtest" not in names
    assert "risk" not in names
    assert "charts" not in names
    assert "ledger" not in names
    assert "loo" in names


def test_run_single_combo_pipeline_runs_detailed_analysis_when_executor_exists():
    calls = []
    run_single_combo_pipeline(
        _request(args=_args(detailed_analysis=True)),
        hooks=_hooks(calls, contributions={}),
    )

    detailed = [call for call in calls if call[0] == "detailed"]
    assert len(detailed) == 1
    assert detailed[0][2]["benchmark"] == "SH000905"


def test_run_single_combo_pipeline_skips_detailed_analysis_without_executor():
    calls = []
    report = pd.DataFrame({"account": [1.0], "return": [0.0], "bench": [0.0]})
    hooks = _hooks(calls, report_df=report, contributions={})
    hooks = SingleComboPipelineHooks(
        **{
            **hooks.__dict__,
            "run_backtest": lambda *args, **kwargs: calls.append(("backtest", args, kwargs)) or (report, None),
        }
    )

    run_single_combo_pipeline(
        _request(args=_args(detailed_analysis=True)),
        hooks=hooks,
    )

    assert "detailed" not in [name for name, _args, _kwargs in calls]


def test_run_single_combo_pipeline_ledger_payload_matches_legacy():
    calls = []
    run_single_combo_pipeline(_request(), hooks=_hooks(calls))

    ledger = [call for call in calls if call[0] == "ledger"][0][2]
    assert ledger["workspace_root"] == "/tmp/workspace"
    assert ledger["eval_window"] == {
        "start": "2020-01-01",
        "end": "2020-01-02",
        "only_last_years": 0,
        "only_last_months": 0,
    }
    assert ledger["metrics"]["annualized_return"] == 0.125555
    assert ledger["metrics"]["calmar"] == pytest.approx(2.4043)
    assert ledger["sub_model_metrics"]["M1"]["max_drawdown"] == -0.07
    assert ledger["loo_contributions"] == {
        "M1": {"delta": 0.111112},
        "M2": {"delta": -0.05},
    }
    assert ledger["cli_args"] == ["--from-config"]


def test_run_single_combo_pipeline_ledger_failure_is_non_fatal(capsys):
    calls = []
    hooks = _hooks(calls, contributions={})
    hooks = SingleComboPipelineHooks(
        **{
            **hooks.__dict__,
            "append_to_fusion_ledger": lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        }
    )

    result = run_single_combo_pipeline(_request(), hooks=hooks)

    assert result is not None
    assert "写入失败（非致命）: boom" in capsys.readouterr().out


def test_script_run_single_combo_preserves_patch_points(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    import importlib
    from quantpits.scripts import ensemble_fusion as ef

    ef = importlib.reload(ef)

    norm_df = _norm_frame(("M1",))
    final_score = pd.Series([0.2, 0.3], index=norm_df.index)
    args = _args(output_dir=str(tmp_path / "out"), no_charts=True)

    monkeypatch.setattr(ef, "correlation_analysis", MagicMock(return_value=pd.DataFrame()))
    monkeypatch.setattr(ef, "calculate_weights", MagicMock(return_value=(None, {"M1": 1.0}, False)))
    monkeypatch.setattr(ef, "generate_ensemble_signal", MagicMock(return_value=final_score))
    monkeypatch.setattr(ef, "save_predictions", MagicMock(return_value="pred.csv"))
    monkeypatch.setattr(ef, "run_backtest", MagicMock(return_value=(pd.DataFrame({"account": [1.0]}), object())))
    monkeypatch.setattr(ef, "risk_analysis_and_leaderboard", MagicMock(return_value=({}, None)))
    monkeypatch.setattr(ef, "calculate_loo_contribution", MagicMock(return_value={}))

    result = ef.run_single_combo(
        "combo_a",
        ["M1"],
        "equal",
        None,
        norm_df,
        {"M1": 0.5},
        ["M1"],
        {"experiment_name": "exp"},
        {"TopK": 20},
        {},
        "2020-01-02",
        "exp",
        args,
    )

    assert result["pred_file"] == "pred.csv"
    ef.run_backtest.assert_called_once()
