"""Static computation and requested research settings have separate builders."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from quantpits.training.static_config import build_model_task, build_record_config
from quantpits.utils.handler_cache import build_cache_key
from tests.quantpits.training.static_config_cases import configuration_cases, explicit_portfolio

BASELINE = json.loads((Path(__file__).parent / "fixtures/static_config_baseline.json").read_text())


@pytest.mark.parametrize("case", configuration_cases(), ids=lambda c: c["name"])
def test_legacy_facade_and_separate_tasks_match_frozen_old_outputs(case, mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    path = tmp_path / "workflow.yaml"
    raw = yaml.safe_dump(case["template"])
    path.write_text(raw)
    original = copy.deepcopy(case)
    with patch("quantpits.utils.strategy.generate_port_analysis_config", side_effect=lambda *, freq: explicit_portfolio(freq)), \
         patch.object(train_utils, "resolve_pretrained_path", return_value=case["resolved_path"]) as resolve, \
         patch.object(train_utils, "validate_pretrain_compatibility") as validate:
        facade = train_utils.inject_config(str(path), case["params"], case["model_name"], case["no_pretrain"])
        model, records = train_utils.prepare_static_tasks(str(path), case["params"], case["model_name"], case["no_pretrain"])
    expected = BASELINE["cases"][case["name"]]
    assert facade == expected["config"]
    assert model["task"]["model"] == facade["task"]["model"]
    assert model["task"]["dataset"] == facade["task"]["dataset"]
    assert model["data_handler_config"] == facade["data_handler_config"]
    assert records["task"]["record"] == facade["task"]["record"]
    assert records["strategy"] == facade["strategy"]
    assert model["custom_root"] == {"keep": [1, 2]} and model["task"]["custom_task"] == {"keep": True}
    assert "record" not in model["task"] and "strategy" not in model and "port_analysis_config" not in model
    assert "model" not in records["task"] and "dataset" not in records["task"]
    assert build_cache_key(model["task"]["dataset"]["kwargs"]["handler"]) == expected["handler_key"]
    assert (facade["data_handler_config"] is facade["task"]["dataset"]["kwargs"]["handler"]["kwargs"]) == expected["handler_alias"]
    assert facade["port_analysis_config"] is facade["task"]["record"][2]["kwargs"]["config"]
    assert records["port_analysis_config"] is records["task"]["record"][2]["kwargs"]["config"]
    assert path.read_text() == raw and case == original
    if case["name"] in ("week_alias", "day_override", "multi_label", "non_alias", "no_pretrain", "configured_path"):
        resolve.assert_not_called()
    if case["name"] == "pretrain_found":
        assert validate.call_args.args == ("synthetic", case["resolved_path"], 20)


def test_record_changes_do_not_change_computation_or_handler_key():
    case = configuration_cases()[0]
    template, params = case["template"], case["params"]
    model = build_model_task(template, params)
    unchanged = copy.deepcopy(model)
    changed = copy.deepcopy(params)
    changed["account"] = 1
    portfolio = explicit_portfolio("week")
    portfolio["backtest"]["exchange_kwargs"]["open_cost"] = .5
    records = build_record_config(template, changed, port_analysis_config=portfolio)
    assert records["port_analysis_config"]["backtest"]["account"] == 1
    assert records["port_analysis_config"]["backtest"]["exchange_kwargs"]["open_cost"] == .5
    assert model == unchanged == build_model_task(template, changed)
    assert build_cache_key(model["task"]["dataset"]["kwargs"]["handler"]) == BASELINE["cases"]["week_alias"]["handler_key"]
    records["task"]["record"][0]["kwargs"]["dataset"] = "mutated consumer"
    assert model == unchanged
    assert model["task"]["model"]["kwargs"]["topk"] == 7  # constructor value retained; helper owns fit attribute injection
    assert model["task"]["model"]["kwargs"]["metric"] == "ir"


@pytest.mark.parametrize("field", ["features", "label", "fit"])
def test_computational_inputs_reach_the_actual_handler(field):
    case = configuration_cases()[0]
    template, params = case["template"], case["params"]
    if field == "features":
        template["data_handler_config"]["feature_extension"]["fields"].append("$volume")
    elif field == "label":
        params["label_formula"] = "changed-label"
    else:
        params["fit_start_time"] = "2024-01-03"
    model = build_model_task(template, params)
    effective = model["task"]["dataset"]["kwargs"]["handler"]
    assert build_cache_key(effective) != BASELINE["cases"]["week_alias"]["handler_key"]
    assert effective["kwargs"] is model["data_handler_config"]
    if field == "label": assert effective["kwargs"]["label"] == ["changed-label"]
    if field == "fit": assert effective["kwargs"]["fit_start_time"] == "2024-01-03"


def test_model_task_ignores_unrequested_invalid_record_configuration():
    case = configuration_cases()[0]
    case["template"]["strategy"] = None
    case["template"]["port_analysis_config"] = "invalid unused research context"
    case["template"]["task"]["record"] = None
    params = {k: v for k, v in case["params"].items() if k not in ("account", "benchmark")}
    model = build_model_task(case["template"], params)
    assert model["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] == ["Ref($close, -6) / Ref($close, -1) - 1"]
    with pytest.raises((TypeError, ValueError)):
        build_record_config(case["template"], case["params"])


def test_record_builder_requires_explicit_context_and_preserves_inputs():
    case = configuration_cases()[0]
    with pytest.raises(ValueError, match="explicit"):
        build_record_config(case["template"], case["params"])
    original = copy.deepcopy(case)
    context = explicit_portfolio("week")
    context_before = copy.deepcopy(context)
    result = build_record_config(case["template"], case["params"], port_analysis_config=context)
    assert result["task"]["record"][1]["kwargs"]["ann_scaler"] == 52
    assert case == original and context == context_before


def test_report_options_are_not_model_inputs():
    case = configuration_cases()[0]
    original = build_model_task(case["template"], case["params"])
    case["template"]["task"]["record"][1]["kwargs"] = {"custom_report": {"enabled": False}}
    changed = build_model_task(case["template"], case["params"])
    assert changed == original
    records = build_record_config(case["template"], case["params"], port_analysis_config=explicit_portfolio("week"))
    assert records["task"]["record"][1]["kwargs"]["custom_report"] == {"enabled": False}


def test_legacy_fit_parameter_aliases_and_facade_alias_graph_survive():
    from quantpits.training.static_config import compose_static_config
    case = configuration_cases()[0]
    case["template"]["task"]["model"]["kwargs"] = case["template"]["strategy"]["params"]
    before = copy.deepcopy(case)
    model = build_model_task(case["template"], case["params"])
    assert model["task"]["model"]["kwargs"]["topk"] == 33
    assert model["task"]["model"]["kwargs"]["n_drop"] == 4
    facade = compose_static_config(case["template"], case["params"], port_analysis_config=explicit_portfolio("week"))
    assert facade["strategy"]["params"] is facade["task"]["model"]["kwargs"]
    assert case == before


def test_pretrain_resolution_and_compatibility_stay_at_the_adapter(mock_env_constants, tmp_path):
    train_utils, _ = mock_env_constants
    case = configuration_cases()[4]
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(case["template"]))
    with patch("quantpits.utils.strategy.generate_port_analysis_config", return_value=explicit_portfolio("week")), \
         patch.object(train_utils, "resolve_pretrained_path", return_value="/synthetic/base.pkl"), \
         patch.object(train_utils, "validate_pretrain_compatibility", side_effect=ValueError("incompatible d_feat")):
        with pytest.raises(ValueError, match="incompatible"):
            train_utils.prepare_static_tasks(str(path), case["params"], model_name="synthetic")
    pure = build_model_task(case["template"], case["params"], pretrain_path="/explicit/selected.pkl")
    assert pure["task"]["model"]["kwargs"]["model_path"] == "/explicit/selected.pkl"


def test_requested_record_error_precedes_model_initialization(mock_env_constants, tmp_path, monkeypatch):
    train_utils, _ = mock_env_constants
    case = configuration_cases()[0]
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(case["template"]))
    def fail(*args, **kwargs): raise RuntimeError("requested analysis configuration is invalid")
    monkeypatch.setattr("quantpits.utils.strategy.generate_port_analysis_config", fail)
    monkeypatch.setattr("qlib.utils.init_instance_by_config", lambda *a, **k: pytest.fail("fit/model initialization must not start"))
    with pytest.raises(RuntimeError, match="requested analysis"):
        train_utils.train_single_model("synthetic", str(path), case["params"], "unused")


def test_real_static_cache_and_default_cache_never_build_records(tmp_path, monkeypatch):
    from quantpits.training.service import prepare_default_cache
    from quantpits.utils.handler_cache import enumerate_tasks_static
    from quantpits.utils.workspace import WorkspaceContext
    case = configuration_cases()[0]
    case["template"]["task"]["record"] = None  # invalid for a requested full run, unused by cache
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(case["template"]))
    def forbidden(*args, **kwargs): pytest.fail("unused records/model factory must not run")
    monkeypatch.setattr("quantpits.utils.strategy.generate_port_analysis_config", forbidden)
    monkeypatch.setattr("qlib.utils.init_instance_by_config", forbidden)
    monkeypatch.setattr("quantpits.utils.handler_cache.estimate_handler_memory", lambda cfg: 0)
    # Prove that the guard actually detects the forbidden factory.
    with pytest.raises(pytest.fail.Exception): forbidden()
    tasks = enumerate_tasks_static(["synthetic"], {"synthetic": str(path)}, case["params"])
    assert build_cache_key(tasks[0][1]) == BASELINE["cases"]["week_alias"]["handler_key"]
    run = SimpleNamespace(prepared=SimpleNamespace(options=SimpleNamespace(cache_size_mb=16, family="static"),
        ctx=WorkspaceContext.from_root(tmp_path)), targets=[SimpleNamespace(target=SimpleNamespace(model_name="synthetic",workflow_path="workflow.yaml"))], params=case["params"])
    cache = prepare_default_cache(run)
    assert cache.stats["handlers_registered"] == 1 and cache.stats["handlers_cached"] == 0


def test_builders_import_and_build_without_workspace_or_runtime(tmp_path):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for key in ("QLIB_WORKSPACE_DIR", "MLFLOW_TRACKING_URI", "QLIB_DATA_DIR", "QLIB_REGION"):
        environment.pop(key, None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    code = '''import sys
from quantpits.training.static_config import build_model_task, build_record_config
from tests.quantpits.training.static_config_cases import configuration_cases, explicit_portfolio
case=configuration_cases()[0]
model=build_model_task(case["template"],case["params"])
records=build_record_config(case["template"],case["params"],port_analysis_config=explicit_portfolio("week"))
assert model["task"]["model"] and records["task"]["record"]
assert not any(name in sys.modules for name in ("quantpits.utils.env","quantpits.utils.train_utils","quantpits.utils.strategy","qlib","mlflow"))
'''
    completed = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=environment,
        capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []
