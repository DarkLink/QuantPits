"""Optional static research inputs must reach computation without account reads."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from quantpits.training.command import TrainingRunOptions, prepare_training_run, prepared_plan_json
from quantpits.training.resolved import resolve_training_run
from quantpits.training.errors import TrainingExecutionError, TrainingPlanError, TrainingStateConflictError
from quantpits.training.service import TrainingExecutionHooks, TrainingExecutionService
from quantpits.utils.workspace import WorkspaceContext
from tests.quantpits.utils.test_train_utils_pipeline import finite_signal_training


MODEL = {
    "market": "finite", "benchmark": "SH000300", "freq": "week",
    "train_date_mode": "fixed", "current_date": "2024-05-17", "data_slice_mode": "fixed",
    "start_time": "2024-01-02", "fit_start_time": "2024-01-02", "fit_end_time": "2024-03-29",
    "valid_start_time": "2024-04-01", "valid_end_time": "2024-04-12",
    "test_start_time": "2024-04-15", "test_end_time": "2024-05-17",
}
RESEARCH = {
    "strategy": {"name": "topk_dropout", "params": {
        "topk": 2, "n_drop": 1, "only_tradable": True, "buy_suggestion_factor": 2,
    }},
    "backtest": {"account": 123000, "exchange_kwargs": {
        "limit_threshold": .095, "deal_price": "close", "open_cost": .002,
        "close_cost": .003, "min_cost": 7,
    }},
}


def workspace(tmp_path, cash=777000):
    root = tmp_path / "Demo_Workspace"
    (root / "config").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "output").mkdir()
    (root / "config/model_config.json").write_text(json.dumps(MODEL))
    (root / "config/model_registry.yaml").write_text(
        "models:\n  finite: {enabled: true, yaml_file: config/finite.yaml}\n")
    (root / "config/finite.yaml").write_text("model: {}\n")
    (root / "config/strategy_config.yaml").write_text(
        "strategy:\n  name: topk_dropout\n  params: {topk: 20, n_drop: 3, buy_suggestion_factor: 2}\n"
        "backtest:\n  account: 999000\n  exchange_kwargs: {deal_price: close}\n")
    prod = {"current_date": "2020-01-01"}
    if cash is not None:
        prod["current_full_cash"] = cash
    (root / "config/prod_config.json").write_text(json.dumps(prod))
    (root / "latest_train_records.json").write_text(json.dumps({
        "experiment_name": "source-experiment", "models": {"finite@static": "source-finite"},
    }))
    return root


def legacy_observation(root):
    from quantpits.utils.train_utils import calculate_dates
    prepared = prepare_training_run(ctx=WorkspaceContext.from_root(root), options=TrainingRunOptions(
        family="static", action="predict_only", all_enabled=True,
        run_id="research-baseline", cache_size_mb=0,
    ))
    params = calculate_dates(ctx=prepared.ctx)
    run = resolve_training_run(prepared, params)
    return {
        "plan": prepared_plan_json(prepared), "params": params,
        "execution_fingerprint": run.execution_fingerprint,
        "resume_fingerprint": run.resume_fingerprint,
        "resolved_params_fingerprint": run.resolved_params_fingerprint,
    }


@pytest.mark.parametrize("cash", [777000, None], ids=["account", "fallback"])
def test_default_request_matches_prechange_identity(tmp_path, cash):
    baseline = json.loads((Path(__file__).parent / "fixtures/static_research_legacy_baseline.json").read_text())
    observed = legacy_observation(workspace(tmp_path, cash))
    assert observed == baseline["account" if cash is not None else "fallback"]
    assert "research" not in json.dumps(observed["plan"]["plan"]["metadata"]).lower()


def prepare(root, **changes):
    options = dict(family="static", action="incremental", all_enabled=True,
                   run_id="research-test", research_config="config/research.yaml", cache_size_mb=0)
    options.update(changes)
    return prepare_training_run(ctx=WorkspaceContext.from_root(root), options=TrainingRunOptions(**options))


def research_file(root, config=None):
    path = root / "config/research.yaml"
    path.write_text(yaml.safe_dump(RESEARCH if config is None else config))
    return path


def hooks(**changes):
    options = dict(activate_workspace=lambda _: None, init_qlib=lambda: None, calculate_dates=None)
    options.update(changes)
    return TrainingExecutionHooks(**options)


def tree(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_explicit_plan_is_readonly_and_uses_an_immutable_selected_input(tmp_path, monkeypatch):
    from quantpits.training.command import render_prepared_plan
    root = workspace(tmp_path)
    path = research_file(root)
    (root / "config/prod_config.json").write_text("{broken account")
    (root / "config/strategy_config.yaml").write_text("[broken default")
    def forbidden(*a, **k):
        pytest.fail("explicit research must not load merged account/strategy settings")
    monkeypatch.setattr("quantpits.utils.config_loader.load_workspace_config", forbidden)
    monkeypatch.setattr("quantpits.utils.strategy.load_strategy_config", forbidden)
    before = tree(root)
    prepared = prepare(root)
    assert tree(root) == before
    assert not (root / "mlflow.db").exists() and not (root / "mlruns").exists()
    assert prepared.plan.metadata["research_config"]["backtest"]["account"] == 123000
    assert "Research conditions:" in render_prepared_plan(prepared)
    assert [ref.path for ref in prepared.plan.inputs].count("config/research.yaml") == 1
    # Canonical workspace-relative identity, including for absolute/link spelling.
    alias = root / "selected.yaml"
    alias.symlink_to(path)
    for spelling in (str(path), "selected.yaml"):
        assert prepare(root, research_config=spelling).plan_fingerprint == prepared.plan_fingerprint
    copy_of_input = prepared.research_config.to_dict()
    copy_of_input["backtest"]["account"] = 1
    assert prepared.research_config.to_dict() == RESEARCH
    params = TrainingExecutionService(hooks())._resolve(prepared).params
    assert params["account"] == 123000 and params["topk"] == 2 and params["n_drop"] == 1
    assert params["anchor_date"] == MODEL["current_date"]
    assert params["market"] == MODEL["market"] and params["freq"] == MODEL["freq"]


@pytest.mark.parametrize("cash", [True, False, 0, -1, float("nan"), float("inf"), -float("inf"),
                                 "1000", None, [], {"cash": 100}, 10 ** 1000])
def test_invalid_research_cash_never_falls_back_to_account(tmp_path, cash):
    root = workspace(tmp_path)
    config = copy.deepcopy(RESEARCH)
    config["backtest"]["account"] = cash
    research_file(root, config)
    before = tree(root)
    with pytest.raises(TrainingPlanError, match="backtest.account"):
        prepare(root)
    assert tree(root) == before


@pytest.mark.parametrize("case", ["missing", "yaml", "root", "params", "exchange", "name", "topk",
                                  "date", "account_cursor", "backtest_date", "infinite_fee"])
def test_invalid_research_structure_fails_during_plan(tmp_path, case):
    root = workspace(tmp_path)
    config = copy.deepcopy(RESEARCH)
    if case == "root":
        config = []
    elif case == "params":
        config["strategy"]["params"] = []
    elif case == "exchange":
        config["backtest"]["exchange_kwargs"] = []
    elif case == "name":
        config["strategy"]["name"] = "unsupported"
    elif case == "topk":
        config["strategy"]["params"]["topk"] = 2.5
    elif case == "date":
        config["current_date"] = "2030-01-01"
    elif case == "account_cursor":
        config["current_full_cash"] = 1
    elif case == "backtest_date":
        config["backtest"]["start_time"] = "2030-01-01"
    elif case == "infinite_fee":
        config["backtest"]["exchange_kwargs"]["open_cost"] = float("inf")
    path = research_file(root, config)
    if case == "missing":
        path.unlink()
    elif case == "yaml":
        path.write_text("[broken yaml")
    before = tree(root)
    with pytest.raises(TrainingPlanError):
        prepare(root)
    assert tree(root) == before


@pytest.mark.parametrize("spelling", ["relative", "absolute", "symlink"])
def test_research_path_must_resolve_inside_workspace(tmp_path, spelling):
    root = workspace(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_text(yaml.safe_dump(RESEARCH))
    value = "../outside.yaml" if spelling == "relative" else str(outside)
    if spelling == "symlink":
        (root / "selected.yaml").symlink_to(outside)
        value = "selected.yaml"
    with pytest.raises(TrainingPlanError, match="inside workspace"):
        prepare(root, research_config=value)


def test_research_family_and_legacy_hook_fail_before_execution_side_effects(tmp_path):
    root = workspace(tmp_path)
    research_file(root)
    with pytest.raises(TrainingPlanError, match="only for static"):
        prepare(root, family="cpcv")
    prepared = prepare(root)
    before = tree(root)
    def forbidden(*args):
        pytest.fail("unsupported date hook must fail before execution")
    with pytest.raises(TrainingExecutionError, match="legacy calculate_dates hooks"):
        TrainingExecutionService(hooks(activate_workspace=forbidden, init_qlib=forbidden,
                                       calculate_dates=forbidden)).execute(prepared)
    assert tree(root) == before


@pytest.mark.parametrize("when", ["before", "during_init", "symlink_escape"])
def test_prepared_research_changes_stop_before_model_work(tmp_path, when):
    root = workspace(tmp_path)
    path = research_file(root)
    prepared = prepare(root)
    def change():
        if when == "symlink_escape":
            outside = tmp_path / "same_bytes.yaml"
            outside.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(outside)
        else:
            path.write_text(path.read_text() + "\n# changed after prepare\n")
    if when != "during_init":
        change()
    def forbidden(*args):
        pytest.fail("changed research input must stop before cache/model work")
    with pytest.raises(TrainingStateConflictError):
        TrainingExecutionService(hooks(init_qlib=change if when == "during_init" else lambda: None,
                                       prepare_cache=forbidden, run_static_target=forbidden)).execute(prepared)
    assert not (root / "data/training_transactions").exists()
    assert not (root / "data/training_history.jsonl").exists()


def test_resume_rejects_new_research_conditions_before_model_work(tmp_path):
    from quantpits.training.state import TrainingRunState, TrainingStateRepository
    root = workspace(tmp_path)
    path = research_file(root)
    prepared = prepare(root)
    service = TrainingExecutionService(hooks())
    run = service._resolve(prepared)
    state = TrainingRunState(
        run_id=prepared.plan.run_id, family="static", action="incremental",
        plan_fingerprint=prepared.plan_fingerprint, execution_fingerprint=run.execution_fingerprint,
        resume_fingerprint=run.resume_fingerprint, anchor_date=run.anchor_date,
        target_keys=("finite@static",), outcomes={}, phase="executing",
    )
    TrainingStateRepository(root / "data/run_state.json").save(state)
    changed = copy.deepcopy(RESEARCH)
    changed["backtest"]["account"] *= 2
    path.write_text(yaml.safe_dump(changed))
    resumed = prepare(root, resume=True)
    def forbidden(*args):
        pytest.fail("resume cannot adopt changed research conditions")
    with pytest.raises(TrainingStateConflictError):
        TrainingExecutionService(hooks(prepare_cache=forbidden, run_static_target=forbidden)).execute(resumed)
    assert not (root / "data/training_transactions").exists()
    # A distinct new run may use the new conditions and same model source.
    fresh = prepare(root, run_id="different-research-run")
    assert fresh.source_snapshot == prepared.source_snapshot
    assert service._resolve(fresh).params["account"] == changed["backtest"]["account"]


@pytest.mark.parametrize("action", ["full", "incremental", "predict_only"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_service_runner_helper_analysis_consume_selected_research(
        finite_signal_training, monkeypatch, action, use_cache):
    import pandas as pd
    from qlib.contrib.eva.alpha import calc_ic
    from quantpits.training.service import prepare_default_cache

    observed = []
    def backtest(*, strategy, **kwargs):
        observed.append((strategy, kwargs))
        return {"1day": (pd.DataFrame({"return": [.1]}), {})}, {}
    monkeypatch.setattr("qlib.workflow.record_temp.normal_backtest", backtest)
    def forbidden(*a, **k):
        pytest.fail("selected research must bypass both merged loaders")
    monkeypatch.setattr("quantpits.utils.config_loader.load_workspace_config", forbidden)
    monkeypatch.setattr("quantpits.utils.strategy.load_strategy_config", forbidden)

    def execute(case):
        root, params = case["workspace"], case["params"]
        model = dict(params, train_date_mode="fixed", current_date=params["anchor_date"], data_slice_mode="fixed")
        (root / "config/model_config.json").write_text(json.dumps(model))
        (root / "config/model_registry.yaml").write_text(
            "models:\n  finite: {enabled: true, yaml_file: config/finite_signal.yaml}\n")
        (root / "config/finite_signal.yaml").write_text(yaml.safe_dump(case["task_config"]))
        (root / "config/prod_config.json").write_text("{broken account JSON")
        (root / "config/strategy_config.yaml").write_text("[broken default YAML")
        path = research_file(root)
        if action == "predict_only":
            (root / "source.json").write_text(json.dumps({
                "experiment_name": "finite_signal", "models": {"finite@static": case["source"].id},
            }))
        prepared = prepare(root, action=action, source_records="source.json",
                           experiment_name="finite_signal", cache_size_mb=16 if use_cache else 0)
        case["prepared"] = prepared
        summary = TrainingExecutionService(hooks(prepare_cache=prepare_default_cache)).execute(prepared)
        manifest = root / summary.manifest_path
        history = root / "data" / ("prediction_history.jsonl" if action == "predict_only" else "training_history.jsonl")
        case["manifest"] = json.loads(manifest.read_text())
        case["history"] = [json.loads(line) for line in history.read_text().splitlines()]
        saved = manifest.read_bytes(), history.read_bytes()
        path.write_text(path.read_text() + "\n# changed after completion\n")
        with pytest.raises(TrainingPlanError, match="no training state"):
            prepare(root, action=action if action != "full" else "incremental", resume=True,
                    source_records="source.json", experiment_name="finite_signal")
        assert (manifest.read_bytes(), history.read_bytes()) == saved
        return summary

    case = finite_signal_training("DatasetH", predict_only=action == "predict_only", use_cache=use_cache,
                                  record_classes=["SignalRecord", "SigAnaRecord", "PortAnaRecord"],
                                  fit_validation_steps=2, execute=execute)
    summary, model = case["result"], case["model"]
    assert summary.publication_applied and summary.outcomes[0]["outcome"] == "success"
    assert model.predict_segments.count("test") == len(observed) == 1
    prediction = model.returned_predictions[-1].to_frame("score")
    pd.testing.assert_frame_equal(observed[0][0]["kwargs"]["signal"], prediction)
    pd.testing.assert_frame_equal(case["recorder"].load_object("pred.pkl"), prediction)
    backtest_options = observed[0][1]
    assert backtest_options["account"] == RESEARCH["backtest"]["account"]
    assert backtest_options["exchange_kwargs"] == RESEARCH["backtest"]["exchange_kwargs"]
    assert backtest_options["start_time"] == case["params"]["test_start_time"]
    assert observed[0][0]["kwargs"]["topk"] == 2
    assert observed[0][0]["kwargs"]["n_drop"] == 1
    assert observed[0][0]["kwargs"]["only_tradable"] is True
    labels = case["recorder"].load_object("label.pkl")
    assert labels.loc[(slice(case["dates"][-6], None), slice(None))].isna().all().all()
    ic, _ = calc_ic(prediction.iloc[:, 0], labels.iloc[:, 0])
    pd.testing.assert_series_equal(case["recorder"].load_object("sig_analysis/ic.pkl"), ic)
    assert case["recorder"].load_object("portfolio_analysis/report_normal_1day.pkl").iloc[0, 0] == .1
    if action != "predict_only":
        assert model.topk == 2 and model.n_drop == 1
        assert model.fit_calls == 1 and model.predict_segments == ["valid", "valid", "test"]
    else:
        assert model.fit_dataset is None
        assert case["source_path"].read_bytes() == case["source_bytes"]
    description = case["manifest"]["records"]["research_config"]
    assert description == case["prepared"].plan.metadata["research_config"]
    assert description["account_source"] == "backtest.account"
    assert case["manifest"]["config_fingerprints"]["config/research.yaml"] == description["fingerprint"]
    assert len(case["history"]) == 1


def test_cash_and_fee_changes_keep_model_task_source_and_cache_identity(tmp_path, mock_env_constants, monkeypatch):
    from quantpits.utils.handler_cache import build_cache_key
    from tests.quantpits.training.static_config_cases import configuration_cases
    train_utils, _ = mock_env_constants
    root = workspace(tmp_path)
    path = research_file(root)
    first = prepare(root, action="predict_only")
    run = TrainingExecutionService(hooks())._resolve(first)
    template = configuration_cases()[0]["template"]
    # Exercise an actual shared YAML params alias, retained as fit input.
    template["task"]["model"]["kwargs"]["strategy_params"] = template["strategy"]["params"]
    monkeypatch.setattr(train_utils, "_read_static_template", lambda _: template)
    model_a, records_a = train_utils.prepare_static_tasks("unused", run.params, research_config=first.research_config.to_dict())
    changed = copy.deepcopy(RESEARCH)
    changed["backtest"]["account"] *= 2
    changed["backtest"]["exchange_kwargs"]["open_cost"] *= 2
    path.write_text(yaml.safe_dump(changed))
    second = prepare(root, action="predict_only", run_id="another-run")
    run_b = TrainingExecutionService(hooks())._resolve(second)
    model_b, records_b = train_utils.prepare_static_tasks("unused", run_b.params, research_config=second.research_config.to_dict())
    assert model_a == model_b
    assert first.source_snapshot == second.source_snapshot
    assert build_cache_key(model_a["task"]["dataset"]["kwargs"]["handler"]) == build_cache_key(model_b["task"]["dataset"]["kwargs"]["handler"])
    assert records_a["port_analysis_config"]["backtest"]["account"] == 123000
    assert records_b["port_analysis_config"]["backtest"]["account"] == 246000
    assert records_b["port_analysis_config"]["backtest"]["exchange_kwargs"]["open_cost"] == .004
    assert model_b["task"]["model"]["kwargs"]["strategy_params"]["topk"] == 2
    changed["strategy"]["params"].update(topk=3, n_drop=2)
    model_c, _ = train_utils.prepare_static_tasks("unused", run_b.params, research_config=changed)
    assert model_c["task"]["model"]["kwargs"]["strategy_params"]["topk"] == 3
    assert model_c != model_b
    # Cache-only preparation does not construct research records or load accounts.
    def forbidden(*a, **k):
        pytest.fail("cache-only builder must not generate portfolio records")
    monkeypatch.setattr("quantpits.utils.strategy.generate_port_analysis_config", forbidden)
    from quantpits.training.static_config import build_model_task
    assert build_model_task(template, run.params) == model_a


def test_static_cli_json_plan_accepts_research_without_qlib_or_account(tmp_path, monkeypatch, capsys):
    from quantpits.scripts import static_train
    root = workspace(tmp_path)
    research_file(root)
    (root / "config/prod_config.json").write_text("{broken account JSON")
    before = tree(root)
    def forbidden(*a, **k):
        pytest.fail("JSON plan must not initialize Qlib or execute")
    monkeypatch.setattr(static_train.env, "init_qlib", forbidden)
    monkeypatch.setattr(static_train.env, "safeguard", forbidden)
    assert static_train.main([
        "--workspace", str(root), "--models", "finite", "--json-plan",
        "--research-config", "config/research.yaml", "--anchor", "2024-05-17",
    ]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert plan["metadata"]["research_config"]["backtest"] == RESEARCH["backtest"]
    assert tree(root) == before


@pytest.mark.parametrize("predict_only", [False, True])
@pytest.mark.parametrize("failure", ["close", "interrupt"])
def test_explicit_helper_preserves_failure_and_interrupt_boundaries(
        finite_signal_training, mock_env_constants, predict_only, failure):
    train_utils, _ = mock_env_constants
    captured = []
    def execute(case):
        captured.append(case)
        common = dict(history_file=False, workspace_root=str(case["workspace"]), research_config=RESEARCH)
        if predict_only:
            return train_utils.predict_single_model(
                "finite", {"yaml_file": str(case["workspace"] / "config/finite_signal.yaml")},
                case["params"], "finite_signal", {"experiment_name": "finite_signal",
                "models": {"finite@static": case["source"].id}}, **common)
        return train_utils.train_single_model("finite", str(case["workspace"] / "config/finite_signal.yaml"),
                                               case["params"], "finite_signal", **common)
    arguments = dict(dataset_class="DatasetH", predict_only=predict_only, execute=execute)
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            finite_signal_training(save_error=KeyboardInterrupt("stop saving"), **arguments)
    else:
        case = finite_signal_training(recorder_exit_error=RuntimeError("close failed"), **arguments)
        assert not case["result"]["success"]
        assert case["result"]["record_id"] == case["recorder"].id
        assert "record_entry" not in case["result"] and "history_entry" not in case["result"]
    assert captured[0]["model"].predict_segments == ["test"]
    assert not (captured[0]["workspace"] / "data/prediction_history.jsonl").exists()
    assert not (captured[0]["workspace"] / "data/training_history.jsonl").exists()


def test_research_input_module_imports_without_workspace_or_qlib(tmp_path):
    import os
    import subprocess
    import sys
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("QLIB_WORKSPACE_DIR", None)
    environment.pop("MLFLOW_TRACKING_URI", None)
    result = subprocess.run([sys.executable, "-c",
        "import sys; import quantpits.training.research; "
        "assert 'quantpits.utils.env' not in sys.modules; "
        "assert 'qlib' not in sys.modules; assert 'mlflow' not in sys.modules"],
        env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
